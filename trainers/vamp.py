import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.utils import load_pretrained_weights, count_num_param
from dassl.metrics import compute_accuracy
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import set_random_seed
from clip.clip import tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from trainers.baseda import *
from utils.clip_part import *
from utils.templates import DATASETs_DOMAINs
from utils.my_data_manager import MyDataManager
import numpy as np
import os
import math
import gc
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

_tokenizer = _Tokenizer()


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.VAMP.N_CTX
        ctx_init = cfg.TRAINER.VAMP.CTX_INIT

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.VAMP.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.VAMP.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:  # and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = " ".join(["X"] * n_ctx) + " " + ctx_init.replace("_", " ")
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)  
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)  # + " ".join(["X"] * n_ctx)
            # self.nctx_init = n_ctx

        print('VAMP design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of VAMP context words (tokens): {n_ctx}")

        self.proj = nn.Linear(768, ctx_dim).half()

        ctx_vectors_v = torch.empty(n_ctx, 768, dtype=dtype)  
        nn.init.normal_(ctx_vectors_v, std=0.02)
        self.ctx_v = nn.Parameter(ctx_vectors_v).half()
     

        self.compound_prompts_visual = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 768).half())
                                                         for _ in range(self.compound_prompts_depth - 1)])

        for single_para in self.compound_prompts_visual:
            nn.init.normal_(single_para, std=0.02)

        # Also make corresponding projection layers, for each prompt
        
        self.shared_proj = cfg.TRAINER.VAMP.SHARED_PROJ

        if not self.shared_proj:
            single_layer = nn.Linear(768, ctx_dim).half()
            self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
      
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :]) 

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )
     
        return prompts

    def forward(self):
        ctx = self.proj(self.ctx_v)  # domain_ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix.to(ctx.device)
        suffix = self.token_suffix.to(ctx.device)
        prompts = self.construct_prompts(ctx, prefix, suffix)
       
        text_deep_prompts = []
        if not self.shared_proj:
            for index, layer in enumerate(self.compound_prompt_projections):
                text_deep_prompts.append(layer(self.compound_prompts_visual[index]))
        else:
            for index in range(self.compound_prompts_depth - 1):
                text_deep_prompts.append(self.proj(self.compound_prompts_visual[index]))

        return prompts, self.ctx_v, self.compound_prompts_visual, text_deep_prompts


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MultiPromptLearnerCustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.n_source = len(cfg.DATASET.SOURCE_DOMAINS)
        prompt_learner_list = []
        for i in range(self.n_source):
            prompt_learner_list.append(MultiModalPromptLearner(cfg, classnames, clip_model))
            print(f"Initialize the source{i} prompt learner! ")

        self.prompt_learner_list = nn.ModuleList(prompt_learner_list)

        self.tokenized_prompts = self.prompt_learner_list[0].tokenized_prompts
        self.image_encoder = clip_model.visual 
        self.text_encoder = TextEncoderForV2T(clip_model)

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, domain_id):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_visual_ctx, deep_compound_prompts_visual, deep_compound_prompts_text = self.prompt_learner_list[
            domain_id]()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)

        image_features = self.image_encoder(image.type(self.dtype), shared_visual_ctx, deep_compound_prompts_visual)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        return logits, text_features, image_features


class CLIPModel(nn.Module): #for zero-shot CLIP 
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.text_encoder = Simple_TextEncoder(clip_model)

        prompt_prefix = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_u = [prompt_prefix.format(c.replace("_", " ")) for c in classnames]
        self.tokenized_prompts_u = torch.cat([clip.tokenize(p) for p in prompts_u])

        self.image_encoder = VisionEncoder_Trans(clip_model)

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image_u):
        with torch.no_grad():
            logit_scale = self.logit_scale.exp()
            text_features_u = self.text_encoder(self.tokenized_prompts_u.to(self.logit_scale.device))
            text_features_u = text_features_u / text_features_u.norm(dim=-1, keepdim=True)

            image_features_u = self.image_encoder(image_u.type(self.dtype))
            image_features_u = image_features_u / image_features_u.norm(dim=-1, keepdim=True)

            logits_u = logit_scale * image_features_u @ text_features_u.t()

            pseudo_label = torch.softmax(logits_u, dim=-1)
            max_probs, label_p = torch.max(pseudo_label, dim=-1)

        return max_probs, label_p


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul  
        self.fix_sigma = fix_sigma

    def guassian_kernel(self, source, target, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma: 
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        if source.size(0) != target.size(0):
            min_n = min(source.size(0), target.size(0))
            source = source[0:int(min_n), :]
            target = target[0:int(min_n), :]

        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss


@TRAINER_REGISTRY.register()
class VAMP(TrainerXU):
    def build_data_loader(self):
        dm = MyDataManager(self.cfg)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.source_domains = cfg.DATASET.SOURCE_DOMAINS
        self.target_domains = cfg.DATASET.TARGET_DOMAINS
        self.n_cls = len(classnames)
        self.save = cfg.SAVE_MODEL
        self.batch_size_x = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        self.batch_size_u = cfg.DATALOADER.TRAIN_U.BATCH_SIZE

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.VAMP.PREC == "fp32" or cfg.TRAINER.VAMP.PREC == "amp":
            clip_model.float()  # CLIP's default precision is fp16

        self.dtype = clip_model.dtype

        print("Building custom CLIP...")

        self.model = MultiPromptLearnerCustomCLIP(cfg, classnames, clip_model)

        self.CLIP_ZS = CLIPModel(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder...")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
            if "prompt_learner" in name:
                param.requires_grad_(True)
            # if "share_visual_ctx" in name:
            #     param.requires_grad_(True)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)

        print(f"Parameters to be updated: {sorted(enabled)}")

        self.ids_domain = DATASETs_DOMAINs[cfg.DATASET.NAME.split("_")[1]]["ids_domain"]
        self.domain_ids = DATASETs_DOMAINs[cfg.DATASET.NAME.split("_")[1]]["domain_ids"]
        self.seed = cfg.SEED

        self.source_prompt_domain2ids = {} 
        for i, domain in enumerate(self.source_domains):
            self.source_prompt_domain2ids[domain] = i

        self.alpha1 = cfg.TRAINER.VAMP.ALPHA1
        self.alpha2 = cfg.TRAINER.VAMP.ALPHA2

        print(f"alpha1: {self.alpha1}, alpha2: {self.alpha2}")  
       
        num_batches = len(self.train_loader_x)
        self.total_steps = num_batches * cfg.OPTIM.MAX_EPOCH
        print(f"total steps: {self.total_steps}")

        self.consistency = cfg.TRAINER.VAMP.CONSISTENCY
        self.TSD = cfg.TRAINER.VAMP.TSD
        self.TCC = cfg.TRAINER.VAMP.TCC
        self.DDA = cfg.TRAINER.VAMP.DDA
  
        self.confi = cfg.TRAINER.VAMP.CONFI
        self.warmup_epoch = cfg.OPTIM.WARMUP_EPOCH

        self.mmd_loss = MMD_loss(fix_sigma=cfg.TRAINER.VAMP.SIGMA)

        self.model.to(self.device)
        self.CLIP_ZS.to(self.device)
        self.CLIP_ZS.eval()

        self.zs_flag = True

        param_dict = [{'params': [p for p in self.model.parameters() if p.requires_grad]}]
       
        self.optim = build_optimizer(model=self.model, param_groups=param_dict, optim_cfg=cfg.OPTIM)

        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        self.register_model("multi_prompt_learner", self.model, self.optim, self.sched)

    def train(self):
        """Generic training loops."""

        self.before_train()
        self.test()
        print("Training the domain-invariant maple model on source dataset...")
        for self.epoch in range(self.start_epoch, self.max_epoch):
            set_random_seed(self.epoch + self.seed)  # Make sure the dataloader loads a different batch for each epoch
            self.run_epoch()
            gc.collect()
            self.after_epoch()
        self.after_train()

    def model_backward_and_update(self, loss, names=None):
        self.model_zero_grad(names)
        self.model_backward(loss)
        self.model_update(names)

    def forward_backward(self, batch, batch_t=None, TSD=True, TCC=True, DDA=True):
        image_x, label, domain_x, image_u, domain_u, image_t = self.parse_batch_train(batch, batch_t)

        cur_domain = self.source_prompt_domain2ids[self.ids_domain[int(domain_x[0])]]

        image = torch.cat([image_x, image_u, image_t], dim=0)  
        logits, text_features_t_c, image_features = self.model(image, cur_domain)

        logits_x = logits[0:image_x.size(0), :]  
        loss_ce = F.cross_entropy(logits_x, label).to(self.device)

        # ce_u
        loss_ce_t = torch.tensor(0.0)
        logits_t_c = logits[(image_x.size(0) + image_u.size(0)):, :]  
        if self.zs_flag or self.epoch < self.warmup_epoch:
            max_probs, label_p = self.CLIP_ZS(image_t)
 
            mask = max_probs.ge(self.confi).float()

            if mask.sum() != 0:
                loss_ce_t = (F.cross_entropy(logits_t_c, label_p, reduction="none") * mask).sum() / mask.sum()

        mmd_loss = torch.tensor(0.0)
        consist_loss = torch.tensor(0.0)
        diversity_loss = torch.tensor(0.0)

        if DDA:
            image_features_u = image_features[image_x.size(0):(image_x.size(0) + image_u.size(0)), :]
            image_features_t = image_features[(image_x.size(0) + image_u.size(0)):, :]  
            mmd_loss = self.mmd_loss(image_features_u, image_features_t)

        remain_domain = []
        for k, v in self.source_prompt_domain2ids.items():
            if v != cur_domain:
                remain_domain.append(v)
        
        logits_r_l = [] 
        text_features_r_l = []  
        for i in remain_domain:
            logits_t_r, text_features_t_r_x, _ = self.model(image_t, i)
            logits_r_l.append(logits_t_r)
            text_features_r_l.append(text_features_t_r_x)

        if TCC:
            logits_r_l = torch.stack(logits_r_l, dim=0)  
            
            logits_c_l = logits[(image_x.size(0) + image_u.size(0)):, :].unsqueeze(0).repeat(logits_r_l.size(0), 1,
                                                                                                1)  
            if len(remain_domain) == 1:
                consist_loss = (logits_r_l - logits_c_l).abs().mean()
            
            elif len(remain_domain) == 2:
                consist_loss = torch.sum(torch.mean(torch.mean((logits_r_l - logits_c_l).abs(), dim=2), dim=1),
                                            dim=0)
                
                consist_loss += torch.mean(torch.mean((logits_r_l[0, :, :] - logits_r_l[1, :, :]).abs(), dim=1), dim=0) 
                
                consist_loss /= 3
            
        if TSD:
            text_features_r_l = torch.stack(text_features_r_l, dim=0)  
            text_features_t_c = text_features_t_c.unsqueeze(0).repeat(text_features_r_l.size(0), 1, 1) 

            if len(remain_domain) == 1:
                dis = text_features_t_c @ text_features_r_l.permute(0, 2, 1) 
                matrix_eye = [torch.eye(self.n_cls, dtype=torch.bool) for _ in range(text_features_r_l.size(0))]
                matrix_eye = torch.stack(matrix_eye, dim=0) 
                diversity_loss = dis[matrix_eye].abs().mean()

            if len(remain_domain) == 2:
                dis1 = text_features_t_c @ text_features_r_l.permute(0, 2, 1)  
                matrix_eye1 = [torch.eye(self.n_cls, dtype=torch.bool) for _ in range(text_features_r_l.size(0))]
                matrix_eye1 = torch.stack(matrix_eye1, dim=0)  
                diversity_loss1 = torch.mean(dis1[matrix_eye1].abs().view(len(remain_domain), -1), dim=1)  # 2
                diversity_loss = torch.sum(diversity_loss1, dim=0)
                dis_2 = text_features_r_l[0, :, :] @ text_features_r_l[1, :, :].permute(1, 0)  
                matrix_eye2 = torch.eye(self.n_cls, dtype=torch.bool)
                diversity_loss += dis_2[matrix_eye2].abs().mean()  
                diversity_loss /= 3
  
        gamma = (2 / (1 + math.exp(-10 * (self.n_iter) / (self.total_steps))) - 1)

        loss = loss_ce + loss_ce_t + gamma * self.alpha1 * (mmd_loss + consist_loss) + gamma * self.alpha2 * diversity_loss 

        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "gamma": gamma,
            "loss_ce": loss_ce.item(),
            "loss_ce_t": loss_ce_t.item(),
            "dda_loss": mmd_loss.item(),
            "tcc_loss": consist_loss.item(),
            "tsd_loss": diversity_loss.item(),
            "acc_x": compute_accuracy(logits_x, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch, batch_t):

        input = batch["img"][0:self.batch_size_x]
        label = batch["label"][0:self.batch_size_x]
        domain = batch["domain"][0:self.batch_size_x]

        input_u = batch["img"][self.batch_size_x:]
        domain_u = batch["domain"][self.batch_size_x:]

        input_t = batch_t["img"]

        input = input.to(self.device)
        label = label.to(self.device)
        input_u = input_u.to(self.device)
        domain_u = domain_u.to(self.device)
        input_t = input_t.to(self.device)
        return input, label, domain, input_u, domain_u, input_t

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = ((self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
                                if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False)

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test()
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                if self.save:
                    self.save_model(self.epoch, self.output_dir, model_name="model-best.pth.tar")
            self.set_model_mode("train")

        if self.save and (meet_checkpoint_freq or last_epoch):
            self.save_model(self.epoch, self.output_dir)

    def save_model(self, epoch, directory, is_best=False, model_name=""):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        len_train_loader = len(self.train_loader_x)
        self.num_batches = len_train_loader

        train_loader_iter = iter(self.train_loader_x)
        test_loader_iter = iter(self.test_loader)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(self.train_loader_x)
                batch = next(train_loader_iter)

            try:
                batch_t = next(test_loader_iter)
            except StopIteration:
                test_loader_iter = iter(self.test_loader)
                batch_t = next(test_loader_iter)

            self.n_iter = self.epoch * self.num_batches + self.batch_idx
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch, batch_t,
                                                TSD=self.TSD,
                                                TCC=self.TCC, DDA=self.DDA)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0 \
                    or self.num_batches < self.cfg.TRAIN.PRINT_FREQ:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch -
                              1) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print("epoch [{0}/{1}][{2}/{3}]\t"
                      "time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                      "data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                      "eta {eta}\t"
                      "{losses}\t"
                      "lr {lr:.6e}".format(
                    self.epoch + 1,
                    self.max_epoch,
                    self.batch_idx + 1,
                    self.num_batches,
                    batch_time=batch_time,
                    data_time=data_time,
                    eta=eta,
                    losses=losses,
                    lr=self.get_current_lr(),
                ))

            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, self.n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), self.n_iter)

            end = time.time()

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        data_loader = self.test_loader
        print("Do evaluation on test set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)

            output_l = []
            for i in range(len(self.source_domains)):
                output, _, _ = self.model(input, i)  
                output_l.append(output)
              

            output_l = torch.stack(output_l, dim=0) 
            output_l = output_l.mean(dim=0)

            self.evaluator.process(output_l, label)

        results = self.evaluator.evaluate()
        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        results_all = results["accuracy"]

        return results_all
    
            
    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
