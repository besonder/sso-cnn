# -*- coding: utf-8 -*-
from typing import Dict, Optional
import random
import numpy as np
import time
from datetime import datetime, timezone, timedelta
import torch

class Logger():
    def __init__(self, logfile, isCurrTime=False) -> None:
        self.logfile = logfile
        self.isCurrTime = isCurrTime
        f = open(self.logfile, 'w')
        f.close()

    def print_args(self, args):
        self.log(f'Strat time : {current_time(easy=True)}')
        for key in args.__dict__.keys():
            self.log(f'{key} : {args.__dict__[key]}')
    
    def log(self, text: str, consol: bool = True) -> None:
        if self.isCurrTime: text = f"[{current_time(easy=True)}]\t" + text
        with open(self.logfile, 'a') as f:
            print(text, file=f)
        if consol:
            print(text)

    def log_time(self):
        self.log(f"[{current_time(easy=True)}]")

    def __call__(self, text: str, consol: bool = True) -> None:
        self.log(text, consol)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name='', fmt=':f', mode='avg'):
        """
        Args:
            fmt: print format. .4f
            mode: 'avg', 'sum', 'val'
        """
        self.name = name
        self.fmt = fmt
        self.mode = mode
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        if self.mode == 'avg':
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        elif self.mode == 'sum':
            fmtstr = '{name} {val' + self.fmt + '} ({sum' + self.fmt + '})'
        elif self.mode == 'val':
            fmtstr = '{name} {val' + self.fmt + '}'
        elif self.mode == 'avg_only':
            fmtstr = '{name} {avg' + self.fmt + '}'
        else:
            raise NotImplemented(f"{self.mode} Mode not implemented")
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters: Optional[Dict[str, AverageMeter]] = None, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters: Dict[str, AverageMeter] = {} if meters is None else meters
        self.prefix = prefix
        self.num_batchs = num_batches

    def __getitem__(self, key) -> AverageMeter:
        return self.meters[key]
    
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    def display(self, batch, isPrint=True):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters.values()]
        if isPrint:
            print('\t'.join(entries))
        return '\t'.join(entries)
    
    def add(self, name='', fmt=':f', mode='avg'):
        self.meters.update({name: AverageMeter(name, fmt, mode)})

    def update(self, name: str, val: int, n: int = 1):
        self.meters[name].update(val, n)

    def reset(self):
        for key in self.meters.keys():
            self.meters[key].reset()

    def keys(self):
        return self.meters.keys()

def get_log_meters(nums, prefix='EPOCH'):
    meters = {
        'loss' : AverageMeter('Loss', ':.4f', mode='avg_only'),
        'train_acc' : AverageMeter('Train Acc', ':.4f', mode='avg_only')
    }
    progress = ProgressMeter(nums, meters, prefix=prefix)

    return progress

def current_time(easy=False):
    """
    return : 
        if easy==False, '20190212_070531'
        if easy==True, '2019-02-12 07:05:31'
    """
    tzone = timezone(timedelta(hours=9)) if 'UTC' in time.tzname else None
    now = datetime.now(tzone)
    if not easy:
        current_time = '{0.year:04}{0.month:02}{0.day:02}_{0.hour:02}{0.minute:02}{0.second:02}'.format(now)
    else:
        current_time = '{0.year:04}-{0.month:02}-{0.day:02} {0.hour:02}:{0.minute:02}:{0.second:02}'.format(now)

    return current_time


def cal_num_parameters(parameters, file=None):
    """
    Args:
        parameters : model.parameters()
    """
    model_parameters = filter(lambda p: p.requires_grad, parameters)
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'The number of parameters : {num_params/1000000:.2f}M')
    if file is not None:
        with open(file, 'a') as f:
            print(f'The number of parameters : {num_params/1000000:.2f}M', file=f)
    return num_params

def do_seed(seed_num, cudnn_ok=True):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num) # if use multi-GPU
    # It could be slow
    if cudnn_ok:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class PieceTriangleLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, epochs, num_batches, last_epoch=-1) -> None:
        self.lr_max = optimizer.param_groups[0]['lr']
        self.total_epochs = epochs
        self.num_batches = num_batches
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        get_lr_func = lambda t: np.interp([t], 
                                            [0, self.total_epochs*2//5, self.total_epochs*4//5, self.total_epochs], 
                                            [0, self.lr_max, self.lr_max/20.0, 0])
        return get_lr_func(self._step_count/self.num_batches)