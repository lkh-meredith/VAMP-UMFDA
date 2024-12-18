import argparse

import torch
import os
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

from datasets import officehome, domainnet

from trainers import vamp


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    # if args.output_dir:
    #     cfg.OUTPUT_DIR = args.output_dir

    if args.model_dir:
        cfg.MODEL_DIR = args.model_dir
        if args.trainer == 'CLIP_ZS' or args.trainer == 'CLIP_LR':
            cfg.MODEL_DIR = None

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.gpu:
        cfg.GPU = args.gpu

    if args.save:
        cfg.SAVE_MODLE = args.save

    if cfg.DATASET.NAME == "MS_OfficeHome":
        DOMAINS = {'a': "Art", 'c': "Clipart", 'p': "Product", 'r': "Real World"}
        # DOMAINs_IDs = {"art": 0, "clipart": 1, "product": 2, "real_world": 3}

    elif cfg.DATASET.NAME == "MS_DomainNet":
        DOMAINS = {'c': "clipart", 'p': "painting", 'r': "real", 's': "sketch"}
        # DOMAINs_IDs = {"clipart": 0, "painting": 1, "real": 2, "sketch": 3}

    if args.domains:  # single source
        
        cfg.DATASET.SOURCE_DOMAINS = [DOMAINS[args.domains]]

    if args.source_domains: #multi-source
        cfg.DATASET.SOURCE_DOMAINS = [DOMAINS[s] for s in args.source_domains]

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = [DOMAINS[args.target_domains]]

    if args.source_domains and args.target_domains and args.output_dir:
        file_names = ""
        for s in args.source_domains:
            file_names += s
        file_names += "_"
        for t in args.target_domains:
            file_names += t

        insert_pos = args.output_dir.find('maple_md/') + len('maple_md/')
        cfg.OUTPUT_DIR = args.output_dir[:insert_pos] + file_names + '/' + args.output_dir[insert_pos:]
        print(f"save path: {cfg.OUTPUT_DIR}")

def extend_cfg(cfg, args):
    """
    Add new config variables for your method.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.SAVE_MODEL = True
    cfg.DATASET.DOMAINs = []
    cfg.DATASET.NUM_SHOTS = 1
    cfg.TRAINER.NAME = ""

    if args.trainer == 'CLIP_ZS' or args.trainer == 'CLIP_LR' or args.trainer == 'CLIP_FC' or args.trainer == 'CLIP_FT':
        cfg.TRAINER.CLIP = CN()
        cfg.TRAINER.CLIP.PREC = "fp16"  # fp16, fp32, amp

    elif args.trainer == 'CoOp':
        cfg.TRAINER.COOP = CN()
        cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp

        cfg.TRAINER.COOP.CSC = False  # class-specific context
        cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
        cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
        cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
        cfg.TRAINER.COOP.DROPOUT = 0.0

    elif args.trainer == 'VPT':
        cfg.TRAINER.VPT = CN()
        cfg.TRAINER.VPT.PREC = "fp16"

        cfg.TRAINER.VPT.VP = True
        cfg.TRAINER.VPT.NUM_TOKENS = 10
        cfg.TRAINER.VPT.LOCATION = "middle"
        cfg.TRAINER.VPT.V_DEEP = False
        cfg.TRAINER.VPT.DEEP_LAYERS = None  # if set to be an int, then do partial-deep prompt tuning
        cfg.TRAINER.VPT.DROPOUT = 0.0

        cfg.TRAINER.VPT.ENABLE_CONV = False
        cfg.TRAINER.VPT.TYPE = "random"  # conv

    elif args.trainer == 'MaPLe':
        cfg.TRAINER.MAPLE = CN()
        cfg.TRAINER.MAPLE.PREC = "fp16"
        cfg.TRAINER.MAPLE.DROPOUT = 0.0
        cfg.TRAINER.MAPLE.DEEP_LAYERS = None
        cfg.TRAINER.MAPLE.SHARE_LAYER = cfg.TRAINER.MAPLE.DEEP_LAYERS

        cfg.TRAINER.MAPLE.TP = True
        cfg.TRAINER.MAPLE.T_DEEP = True
        cfg.TRAINER.MAPLE.CSC = False
        cfg.TRAINER.MAPLE.N_CTX = 2  # number of text context vectors
        cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"
        cfg.TRAINER.MAPLE.CLASS_TOKEN_POSITION = "end"

        cfg.TRAINER.MAPLE.VP = True
        cfg.TRAINER.MAPLE.V_DEEP = cfg.TRAINER.MAPLE.T_DEEP
        cfg.TRAINER.MAPLE.NUM_TOKENS = cfg.TRAINER.MAPLE.N_CTX  # number of visual context vectors
        cfg.TRAINER.MAPLE.LOCATION = "middle"

    
    elif args.trainer == "VAMP":
        cfg.TRAINER.PROMAPLE = CN()
        cfg.TRAINER.PROMAPLE.PREC = "fp16"
        cfg.TRAINER.PROMAPLE.N_CTX = 16  # number of text context vectors
        cfg.TRAINER.PROMAPLE.N_CTX_AG = 2
        cfg.TRAINER.PROMAPLE.CTX_INIT = "a photo of a"
        cfg.TRAINER.PROMAPLE.PROMPT_DEPTH = 12
        cfg.TRAINER.PROMAPLE.ALPHA1 = 0.1
        cfg.TRAINER.PROMAPLE.ALPHA2 = 0.01
        cfg.TRAINER.PROMAPLE.SIGMA = 1.0

        cfg.TRAINER.PROMAPLE.CONSISTENCY = True
        cfg.TRAINER.PROMAPLE.TSD = True
        cfg.TRAINER.PROMAPLE.TCC = True
        cfg.TRAINER.PROMAPLE.DDA = True
        cfg.TRAINER.PROMAPLE.SHARED_PROJ = False

        cfg.TRAINER.PROMAPLE.CONFI = 0.8


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg, args)
    print(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    setup_logger(cfg.OUTPUT_DIR)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(cfg.MODEL_DIR, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default="./data", help="path to dataset")
    parser.add_argument("--output-dir", type=str,
                        default="",
                        help="output directory")
    parser.add_argument("--config-file", type=str, default="",
                        help="path to config file")
    parser.add_argument("--dataset-config-file", type=str, default="",
                        help="path to config file for dataset setup")
    parser.add_argument("--model-dir", type=str, default="",
                        help="load model from this directory for eval-only mode")

    parser.add_argument("--domains", type=str, help="domains for DA/DG")  
    parser.add_argument("--source-domains" , type=str, nargs="+",
                        help="source domains for DA/DG") 
    parser.add_argument("--target-domains" type=str,
                        help="target domains for DA/DG") 

    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="ViT-B/16", help="name of CNN backbone")  
    parser.add_argument("--head", type=str, default="", help="name of head")

    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")

    parser.add_argument("--resume", type=str, default="",
                        help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--load-epoch", type=int,
                        help="load model weights at this epoch for evaluation")

    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")

    parser.add_argument("--gpu", type=str, default="0", help="which gpu to use")  # if you use this hyperpameter, you need modify the source code of dassl library.
    # i.e., in dassl.engine.trainer line 314: self.device = torch.device("cuda:{}".format(cfg.GPU))
    parser.add_argument("--seed", type=int, default=3,
                        help="only positive value enables a fixed seed")
    parser.add_argument("--save", type=str, default=True, help="need to save model")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="modify config options using the command-line")

    args = parser.parse_args()

    main(args)
