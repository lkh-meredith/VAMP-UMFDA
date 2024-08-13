import copy

from clip import clip

import torch
import torch.nn as nn
from torch.nn import functional as F


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    if cfg.TRAINER.NAME == "VAMP":
        design_details={
            "trainer": cfg.TRAINER.NAME,
            "vision_depth": 0,
            "language_depth": 0, "vision_ctx": 0,
            "language_ctx": 0,
            "prompt_length": cfg.TRAINER.VAMP.N_CTX,
        }
    else:
        design_details={
            "trainer": cfg.TRAINER.NAME,
        }

    model = clip.build_model(state_dict=state_dict or model.state_dict(), design_details=design_details)

    return model


class Simple_TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.text_transformer2
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # print(x.size()) #77 31 512
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x


class TextEncoderForV2T(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.text_transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        # print(combined)
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class VisionEncoder_Trans(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        # visual
        self.conv1_visual = clip_model.visual.conv1
        self.class_embedding_visual = clip_model.visual.class_embedding
        self.positional_embedding_visual = clip_model.visual.positional_embedding
        self.ln_pre_visual = clip_model.visual.ln_pre
        self.ln_post_visual = clip_model.visual.ln_post
        self.proj_visual = clip_model.visual.proj
        self.transformer = clip_model.visual2.transformer

    def forward(self, x):
        # visual pre-pro
        x = self.conv1_visual(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
         [self.class_embedding_visual.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                       dtype=x.dtype,device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding_visual.to(x.dtype)
        x = self.ln_pre_visual(x)
        x = x.permute(1, 0, 2)  # NLD -> LND # 209 4 768

        x = self.transformer(x)

        # visual post_pro
        x = x.permute(1, 0, 2)  # LND -> NLD  4 213 768
        x = self.ln_post_visual(x[:, 0, :])

        if self.proj_visual is not None:
            x = x @ self.proj_visual.half()  # 4 512

        return x