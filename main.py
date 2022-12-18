import sys
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from model.backbone import KWLarge, ResNet9, WideResNet
from model.cayley import Normalize, CayleyConv, CayleyLinear
from model.ed import CayleyConvED, CayleyConvED2
from dataset.load_data import train_batches, test_batches, mu, std
from utils.evaluate import accuracy, rob_acc, empirical_local_lipschitzity, cert_stats, margin_loss
from utils.option import get_option

if __name__ == '__main__':
    args = get_option()
    logger = args.logger
    logger(' '.join(sys.argv))
    
    eps = args.eps / 255.0
    alpha = eps / 4.0

    _model = eval(args.backbone)
    conv = eval(args.conv)
    linear = eval(args.linear)

    model = nn.Sequential(
        Normalize(mu, std if args.stddev else 1.0),
        _model(conv=conv, linear=linear)
    ).cuda()

    model_name = args.backbone
    epochs = args.epochs
    lr_max = args.lr_max

    # for SVCM projections
    proj_nits = 100

    # lr schedule: superconvergence
    lr_schedule = lambda t: np.interp([t], [0, epochs*2//5, epochs*4//5, epochs], [0, lr_max, lr_max/20.0, 0])[0]

    # optimizer: Adam
    opt = optim.Adam(model.parameters(), lr=lr_max, weight_decay=0)

    # loss: multi-margin loss
    criterion = lambda yhat, y: margin_loss(yhat, y, 0.5, 1.0, 1.0)

    for epoch in range(1, epochs+1):
        start = time.time()
        train_loss, acc, n = 0, 0, 0
        for i, batch in enumerate(train_batches):
            X, y = batch['input'], batch['target']
            
            lr = lr_schedule(epoch + (i + 1)/len(train_batches))
            opt.param_groups[0].update(lr=lr)
            
            output = model(X)
            loss = criterion(output, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            
            train_loss += loss.item() * y.size(0)
            acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
            
            # for SVCM projections
            if i % proj_nits == 0 or i == len(train_batches) - 1:
                for m in model.modules():
                    if hasattr(m, '_project'):
                        m._project()
        
        if (epoch + 1) % 10 == 0:
            l_emp = empirical_local_lipschitzity(model, test_batches, early_stop=True).item()
            logger(f"[{args.backbone}] --- Empirical Lipschitzity: {l_emp}")

        logger(f"[{args.backbone}] Epoch: {epoch} | Train Acc: {acc/n:.4f}, Test Acc: {accuracy(model, test_batches):.4f}, Time: {time.time() - start:.1f}, lr: {lr:.4f}")

    if not args.stddev:
        vals = cert_stats(model, test_batches, eps * 2**0.5, full=True)
        logger(f"[{args.backbone}] (PROVABLE) Certifiably Robust (eps: {eps:.4f}): {vals[0]:.4f}, Cert. Wrong: {vals[1]:.4f}, Insc. Right: {vals[2]:.4f}, Insc. Wrong: {vals[3]:.4f}")
    
    val_rob_acc = rob_acc(test_batches, model, eps, alpha, opt, False, 10, 1, linf_proj=False, l2_grad_update=True)[0]
    logger(f"[{args.backbone}] (EMPIRICAL) Robust accuracy (eps: {eps:.4f}): {val_rob_acc}")
    logger.log_time()
                