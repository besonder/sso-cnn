# -*- coding: utf-8 -*-
import os
import json
import argparse
import shutil
from .utils import Logger

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_name',   default='',                    help='experiment name')
    parser.add_argument('--gpu',              default='0',                   help='gpu id')
    parser.add_argument('--batch_size',       default=256,       type=int,   help='mini-batch size')
    parser.add_argument('--loss',             default="margin",  type=str,   help='margin or ce', choices=['margin', 'ce'])
    parser.add_argument('--opt',              default="adam",    type=str,   help='adam or sgd', choices=['adam', 'sgd'])
    parser.add_argument('--lr_max',           default=0.01,      type=float)
    parser.add_argument('--lr_schedule',      default="tri",     type=str,   help='piecewise triangle or multi step', choices=['tri', 'step'])
    parser.add_argument('--weight_decay',     default=0.0,       type=float, help='optimizer weight decay')
    parser.add_argument('--epochs',           default=100,       type=int,   help='epochs')

    parser.add_argument('--save_dir',         default='./exps',  type=str,   help='save directory for checkpoint')
    parser.add_argument('--dataset',          default='cifar10', type=str,   help='cifar10 or cifar100', choices=['cifar10', 'cifar100'])

    parser.add_argument('--seed',             default=777,       type=int,   help='random seed')
    parser.add_argument('--num_workers',      default=0,         type=int,   help='number of workers in data loader')

    parser.add_argument('--backbone',         default='ResNet9', choices=['KWLarge', 'ResNet9', 'WideResNet', 'LipConvNet'])
    parser.add_argument('--conv',             default='SESConv', 
                        choices=['PlainConv', 'BCOP', 'CayleyConv', 'SOC', 'ECO', 
                                'SESConv', 'SESConv1x1'])
    parser.add_argument('--linear',           default='none', help='linear ftn. If linear is "none", then use the linear ftn corresponding to chosen conv',
                        choices=['none', 'Linear', 'BjorckLinear', 'CayleyLinear', 'SESLinear'])
    parser.add_argument('--eps',              default=36.0,      type=float)

    args, unknown_args = parser.parse_known_args()

    if 'SES' in args.conv:
        parser.add_argument('--lam',         default=1.7,  type=float, help='the lambda of additional loss')
        parser.add_argument('--scale',       default=4.0,  type=float, help='the scale of loss in SESLinear')
        args, unknown_args = parser.parse_known_args()
    if args.backbone == 'LipConvNet':
        parser.add_argument('--n_lip',       default=1,    type=int,   help='the number of blocks in LipConvNet. 1, 2, 3, 4, 5, 6, 7, 8')
        args, unknown_args = parser.parse_known_args()

    return args, unknown_args

class Config():
    def __init__(self, opt) -> None:
        self.exp_name: str = opt.exp_name
        self.gpu_id: str = opt.gpu
        self.batch_size: int = opt.batch_size
        self.loss: str = opt.loss.lower()
        self.opt: str = opt.opt.lower()
        self.lr_max: float = opt.lr_max
        self.lr_schedule: str = opt.lr_schedule.lower()
        self.weight_decay: float = opt.weight_decay
        self.epochs: int = opt.epochs

        self.save_dir: str = opt.save_dir
        self.dataset: str = opt.dataset

        self.seed: int = int(opt.seed)
        self.num_workers: int = opt.num_workers

        self.backbone: str = opt.backbone
        self.conv: str = opt.conv
        self.linear: str = opt.linear
        self.eps: float = opt.eps

        if 'SES' in self.conv:
            self.lam = opt.lam
            self.scale = opt.scale
        if self.backbone == 'LipConvNet':
            self.n_lip = opt.n_lip

        assert len(self.__dict__) == len(opt.__dict__), "Check argparse"

        conv_linear = {
            'PlainConv'    : 'Linear',         
            'BCOP' : 'BCOP', 'SOC' : 'SOC', 'ECO': 'ECO',
            'CayleyConv'   : 'CayleyLinear',
            'SESConv'  : 'SESLinear',   'SESConv1x1': 'SESLinear',
        }

        if self.linear == 'none':
            self.linear = conv_linear[self.conv]

        self.num_classes = {'cifar10': 10, 'cifar100': 100}[self.dataset]
        if self.backbone == 'LipConvNet':
            self.backbone += f"_N{self.n_lip*5}"

        self.hyper_param = {
            'dataset': '',
            'backbone': '',
            'conv': '',
            'linear': '',
        }

        if "SES" in self.conv:
            self.hyper_param.update({'lam': 'Lam', 'scale': 'Scale'})

        self.hyper_param.update({
            'loss' : 'loss_',
            'lr_max': 'LRMAX',
            # 'epochs': 'Ep',
            # 'batch_size': 'B',
            'seed': 'SEED',
        })

        self._build()

    def _build(self):
        # Set exp name
        for k, v in self.hyper_param.items():
            self.exp_name += f"_{v}{self.__getattribute__(k)}"

        if self.exp_name[0] == '_': self.exp_name = self.exp_name[1:]

        self._save()

    def _save(self):
        log_dir = os.path.join(self.save_dir, self.exp_name)
        if os.path.exists(log_dir):
            if 'debug' in self.exp_name: 
                isdelete = "y"
            else:
                isdelete = input("delete exist exp dir (y/n): ")
            if isdelete == "y":
                shutil.rmtree(log_dir, ignore_errors=True)
            elif isdelete == "n":
                raise FileExistsError
            else:
                raise FileExistsError

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        option_path = os.path.join(self.save_dir, self.exp_name, "options.json")

        with open(option_path, 'w') as fp:
            json.dump(self.__dict__, fp, indent=4, sort_keys=True)

        self.logger = Logger(os.path.join(log_dir, "log.txt"))
        self.logger.log_time()
        self.logger(f"[NAME] {self.exp_name}")
        self.log_dir = log_dir

def get_option() -> Config:
    args, unknown_args = get_args()
    if len(unknown_args) and unknown_args[0] == '-f' and 'jupyter' in unknown_args[1]:
        unknown_args = unknown_args[2:]
    # assert len(unknown_args) == 0, f"Invalid Arguments: {str(unknown_args)}"
    return Config(args)
