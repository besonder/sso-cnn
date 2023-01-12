import os, sys
import numpy as np
import time
import warnings
warnings.filterwarnings(action='ignore')

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from model import get_model, margin_loss, extract_SESLoss
from dataset.load_data import get_dataset
from utils.evaluate import accuracy, rob_acc, empirical_local_lipschitzity, cert_stats
from utils.option import get_option
from utils.utils import do_seed, cal_num_parameters, get_log_meters
import ray
from ray import tune, air
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from main import main
from functools import partial
import warnings   

def net_tune(config):
    args = get_option()
    if 'SES' in args.conv:
        args.lam = float(config["lam"])
        args.scale = float(config["scale"])
        args.gpu_id = config["gpu"]
    else:
        warnings.warn("This tuning only works for SES. Set args conv to 'SES'.")
    main(args, hyperparam_tune=True)

if __name__ == '__main__':
    ###################
    # PARAMETER SETTING
    NUM_SAMPLES = 10        # total trial number
    CPU_RESOURCE = 4        # number of CPU cores per trial
    GPU_RESOURCE = 0.5      # number of GPU cores per trial
    
    config = {
        "lam": tune.sample_from(lambda _: 0.1*float(np.random.randint(10,50))),
        "scale": tune.sample_from(lambda _: np.random.randint(2,6)),
        "gpu": tune.choice(['0']),
        # "gpu": tune.choice(['0','1']),    # For multi gpus
    }
    ###################

    scaling_config = air.ScalingConfig(
        trainer_resources={"CPU": CPU_RESOURCE, "GPU": GPU_RESOURCE},
        placement_strategy="SPREAD"
    )
    reporter = CLIReporter(metric="test_acc", mode="max", sort_by_metric=True)
    trainable_with_resources = tune.with_resources(net_tune, resources=scaling_config)
    tuner = tune.Tuner(
        trainable_with_resources, 
        param_space=config,
        run_config=air.RunConfig(
            local_dir="./ray_results", 
            name="test_experiment", 
            progress_reporter=reporter
            ),
        tune_config=tune.TuneConfig(
            num_samples=NUM_SAMPLES,
            ),
        )
    result = tuner.fit()
