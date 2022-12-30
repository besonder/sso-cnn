# -*- coding: utf-8 -*-
import os
import json
import argparse
import shutil
from .utils import Logger

parser = argparse.ArgumentParser()

parser.add_argument('-e', '--exp_name',   default='',                 help='experiment name')
parser.add_argument('--gpu',              default='0',                help='gpu id')
parser.add_argument('--batch_size',       default=256,        type=int,   help='mini-batch size')
parser.add_argument('--opt',              default="Adam",    type=str,   help='adam or sgd')
parser.add_argument('--weight_decay',     default=0.0,    type=float, help='optimizer weight decay')
parser.add_argument('--epochs',           default=100,       type=int,   help='epochs')

parser.add_argument('--log_step',         default=50,        type=int,   help='step for logging in iteration')
parser.add_argument('--save_step',        default=1,         type=int,   help='step for saving in epoch')
parser.add_argument('--data_dir',         default='./',                  help='data directory')
parser.add_argument('--save_dir',         default='./exps',              help='save directory for checkpoint')

parser.add_argument('--seed',             default=777,       type=int,   help='random seed')
parser.add_argument('--num_workers',      default=4,         type=int,   help='number of workers in data loader')

parser.add_argument('--backbone',         default='KWLarge', choices=['KWLarge', 'ResNet9', 'WideResNet', 'LipConvNet'])
parser.add_argument('--conv',             default='CayleyConvED', 
                    choices=['CayleyConv', 'BCOP', 'RKO', 'SVCM', 'OSSN', 'PlainConv', 'CayleyConvED', 'CayleyConvED2', 'ECO', 'SOC'])
parser.add_argument('--linear',           default='CayleyLinear', 
                    choices=['CayleyLinear', 'BjorckLinear', 'Linear'])
parser.add_argument('--lr_max',           default=0.01,      type=float)
parser.add_argument('--eps',              default=36.0,      type=float)
parser.add_argument('--stddev',           action='store_true')

class Config():
    def __init__(self, opt) -> None:
        self.exp_name: str = opt.exp_name
        self.gpu_id: str = opt.gpu
        self.batch_size: int = opt.batch_size
        self.opt: str = opt.opt.lower()
        self.weight_decay: float = opt.weight_decay
        self.epochs: int = opt.epochs

        self.log_step: int = opt.log_step
        self.save_step: int = opt.save_step
        self.data_dir: str = opt.data_dir
        self.save_dir: str = opt.save_dir

        self.seed: int = int(opt.seed)
        self.num_workers: int = opt.num_workers

        self.backbone: str = opt.backbone
        self.conv: str = opt.conv
        self.linear: str = opt.linear
        self.lr_max: float = opt.lr_max
        self.stddev: bool = opt.stddev
        self.eps: float = opt.eps

        assert len(self.__dict__) == len(opt.__dict__), "Check argparse"

        if self.conv in ['BCOP', 'SOC', 'ECO']:
            self.linear = self.conv

        self.hyper_param = {
            'backbone': '',
            'conv': '',
            'linear': '',
            'stddev': 'STD',
            'lr_max': 'LRMAX',
            'epochs': 'Ep',
            'batch_size': 'B',
            'seed': 'SEED',
        }

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
    option, unknown_args = parser.parse_known_args()
    if len(unknown_args) and unknown_args[0] == '-f' and 'jupyter' in unknown_args[1]:
        unknown_args = unknown_args[2:]
    assert len(unknown_args) == 0, f"Invalid Arguments: {str(unknown_args)}"
    return Config(option)
