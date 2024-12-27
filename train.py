import argparse
import torch
import os
import numpy as np

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet
import datasets.svhn
import datasets.resisc45
import datasets.clevr

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.bar
import trainers.blackvip
import trainers.bptvlm
import trainers.zip
import trainers.zsclip


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

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.best_or_last == "best":     
        cfg.TEST.FINAL_MODEL = "best_val"


def extend_cfg(cfg):
    from yacs.config import CfgNode as CN

    #! Dataset
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"

    #! MODEL basic
    cfg.TRAINER.BASIC = CN()
    cfg.TRAINER.BASIC.OPT_TYPE = "spsa-gc"
    cfg.TRAINER.BASIC.SP_AVG = 5
    cfg.TRAINER.BASIC.MAX_API_CALLS = 0

    #! BAR
    cfg.TRAINER.BAR = CN()
    cfg.TRAINER.BAR.METHOD = 'reprogramming'
    cfg.TRAINER.BAR.PREC = "fp32"
    cfg.TRAINER.BAR.LRS = [0.01, 0.0001]
    cfg.TRAINER.BAR.FRAME_SIZE = 224
    cfg.TRAINER.BAR.SMOOTH = 0.01
    cfg.TRAINER.BAR.SIMGA = 1.0
    cfg.TRAINER.BAR.DECAYING = 0.9
    cfg.TRAINER.BAR.FOCAL_G = 2.0

    #! BlackVIP
    cfg.TRAINER.BLACKVIP = CN()
    cfg.TRAINER.BLACKVIP.METHOD = 'coordinator'
    cfg.TRAINER.BLACKVIP.PT_BACKBONE = 'vit-mae-base'
    cfg.TRAINER.BLACKVIP.SRC_DIM = 1568
    cfg.TRAINER.BLACKVIP.E_OUT_DIM = 0
    cfg.TRAINER.BLACKVIP.SPSA_PARAMS = [1.0,0.005,0.01,0.4,0.2]
    cfg.TRAINER.BLACKVIP.MOMS = 0.9
    cfg.TRAINER.BLACKVIP.P_EPS = 1.0
    cfg.TRAINER.BLACKVIP.PREC = "fp32"

    #! BPT-VLM
    cfg.TRAINER.BPTVLM = CN()
    cfg.TRAINER.BPTVLM.PREC = "fp32"
    cfg.TRAINER.BPTVLM.OPT_TYPE = "shallow_cma"
    cfg.TRAINER.BPTVLM.POP_SIZE = 15
    cfg.TRAINER.BPTVLM.BUDGET = 6000
    cfg.TRAINER.BPTVLM.BOUND = 0
    cfg.TRAINER.BPTVLM.TOKENS_L = 5
    cfg.TRAINER.BPTVLM.TOKENS_V = 5
    cfg.TRAINER.BPTVLM.SIGMA = 1
    cfg.TRAINER.BPTVLM.INTRINSIC_DIM_L = 2000
    cfg.TRAINER.BPTVLM.INTRINSIC_DIM_V = 2000

    #! ZIP
    cfg.TRAINER.ZIP = CN()
    cfg.TRAINER.ZIP.PREC = "fp32" 
    cfg.TRAINER.ZIP.N_CTX = 8
    cfg.TRAINER.ZIP.CTX_INIT = ""
    cfg.TRAINER.ZIP.SPSA_PARAMS = [1.0,0.005,0.01,0.4,0.2]
    cfg.TRAINER.ZIP.MOMS = 0.9
    cfg.TRAINER.ZIP.P_EPS = 1.0
    cfg.TRAINER.ZIP.INTRINSIC_DIM = 1000
    cfg.TRAINER.ZIP.RANK = 5    


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

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
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        acc = trainer.test()
        print("Test acc: {:.2f}".format(acc))
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("--resume",type=str,default="",help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--seed", type=int, default=-1, help="only positive value enables a fixed seed")
    parser.add_argument("--source-domains", type=str, nargs="+", help="source domains for DA/DG")
    parser.add_argument("--target-domains", type=str, nargs="+", help="target domains for DA/DG")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument("--config-file", type=str, default="", help="path to config file")
    parser.add_argument("--dataset-config-file",type=str,default="",help="path to config file for dataset setup",)
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--model-dir",type=str,default="",help="load model from this directory for eval-only mode",)
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")

    #! extension
    parser.add_argument("--visual-backbone", type=str, default="", help="name of CNN visual backbone")
    parser.add_argument("opts",default=None,nargs=argparse.REMAINDER,help="modify config options using the command-line",)
    parser.add_argument("--best_or_last", type=str, default="last", help="best or last model")
    args = parser.parse_args()
    main(args)