import os, sys
import numpy as np
import time
import warnings
warnings.filterwarnings(action='ignore')

import torch
import torch.optim as optim

from model import get_model, margin_loss
from dataset.load_data import get_dataset
from utils.evaluate import accuracy, rob_acc, empirical_local_lipschitzity, cert_stats
from utils.option import get_option
from utils.utils import do_seed

args = get_option()

if __name__ == '__main__':
    start_main = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    logger = args.logger
    logger(' '.join(sys.argv))
    do_seed(args.seed)
    
    eps = args.eps / 255.0
    alpha = eps / 4.0

    model = get_model(args)

    # lr schedule: superconvergence
    lr_schedule = lambda t: np.interp([t], [0, args.epochs*2//5, args.epochs*4//5, args.epochs], [0, args.lr_max, args.lr_max/20.0, 0])[0]

    # optimizer: Adam
    opt = optim.Adam(model.parameters(), lr=args.lr_max, weight_decay=0)

    # loss: multi-margin loss
    criterion = lambda yhat, y: margin_loss(yhat, y, 0.5, 1.0, 1.0)

    # load dataset
    train_batches, test_batches = get_dataset(args)

    for epoch in range(args.epochs):
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
            
        if (epoch + 1) % 10 == 0:
            l_emp = empirical_local_lipschitzity(model, test_batches, early_stop=True).item()
            logger(f"[{args.backbone}] --- Empirical Lipschitzity: {l_emp}")

        logger(f"[{args.backbone}] Epoch: {epoch+1} | Train Acc: {acc/n:.4f}, Test Acc: {accuracy(model, test_batches):.4f}, Time: {time.time() - start:.1f}, lr: {lr:.4f}")

    if not args.stddev:
        vals = cert_stats(model, test_batches, eps * 2**0.5, full=True)
        logger(f"[{args.backbone}] (PROVABLE) Certifiably Robust (eps: {eps:.4f}): {vals[0]:.4f}, Cert. Wrong: {vals[1]:.4f}, Insc. Right: {vals[2]:.4f}, Insc. Wrong: {vals[3]:.4f}")
    
    val_rob_acc = rob_acc(test_batches, model, eps, alpha, opt, False, 10, 1, linf_proj=False, l2_grad_update=True)[0]
    logger(f"[{args.backbone}] (EMPIRICAL) Robust accuracy (eps: {eps:.4f}): {val_rob_acc}")
    logger.log_time()

    torch.save({
        'args': args,
        'epoch': epoch+1,
        'state_dict': model.state_dict()
    }, os.path.join(args.log_dir, "model.pth"))

    logger(f"Elapsed Time: {(time.time() - start_main)/60:.1f} Min")
