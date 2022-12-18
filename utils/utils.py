# -*- coding: utf-8 -*-
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
    def __init__(self, name='', fmt=':f'):
        self.name = name
        self.fmt = fmt
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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.num_batchs = num_batches

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

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