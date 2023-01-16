import os, sys
import time
import warnings
warnings.filterwarnings(action='ignore')

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from model import get_model, margin_loss, extract_SESLoss
from dataset.load_data import get_dataset
from utils.evaluate import accuracy, rob_acc, empirical_local_lipschitzity, cert_stats
from utils.option import get_option, Config
from utils.utils import do_seed, cal_num_parameters, get_log_meters, PieceTriangleLR
from ray import tune

def main_for_tune(args: Config):
    start_main = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    logger = args.logger
    logger(' '.join(sys.argv))
    do_seed(args.seed)
    
    eps = args.eps / 255.0
    alpha = eps / 4.0

    model = get_model(args)
    logger(f"The number of parameters : {cal_num_parameters(model.parameters())/1000000:.2f}M", consol=False)

    # whether the model is SES or not
    sesmode = False
    for _, layer in model.named_modules():
        if "SES" in layer.__class__.__name__:
            sesmode = True
            break

    # load dataset
    train_batches, test_batches = get_dataset(args)

    # optimizer: Adam
    if args.opt == "adam":
        opt = optim.Adam(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        opt = optim.SGD(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)

    # loss: multi-margin loss
    if args.loss == 'margin':
        criterion = lambda yhat, y: margin_loss(yhat, y, 0.5, 1.0, 1.0)
    elif args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()

    # learning rate scheduler
    if args.lr_schedule == 'tri':
        lr_scheduler = PieceTriangleLR(opt, args.epochs, len(train_batches))
    elif args.lr_schedule == 'step':
        lr_steps = args.epochs * len(train_batches)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps // 2, 
            (3 * lr_steps) // 4], gamma=0.1)

    # average meters
    progress = get_log_meters(args.epochs, prefix=f'[{args.backbone}] EPOCH')
    # tensorboard writer
    writer = SummaryWriter(log_dir=args.log_dir)
    global_step = 0

    cert_right = 0.
    val_rob_acc = 0.
    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        for i, batch in enumerate(train_batches):
            X, y = batch['input'], batch['target']
            
            lr = opt.param_groups[0]['lr']
            writer.add_scalar("lr", lr, global_step=global_step)
            
            output = model(X)
            loss = criterion(output, y)
            
            # SESLoss
            if sesmode:
                loss += args.lam *extract_SESLoss(model, scale=args.scale)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()
            
            correct = (output.max(1)[1] == y).sum().item()
            progress.update('loss', loss.item(), y.size(0))
            progress.update('train_acc', correct/y.size(0), y.size(0))

            global_step += 1
            
        # test
        test_acc = accuracy(model, test_batches)
        # log
        msg = f"\tTest Acc {test_acc:.4f}\tlr {lr:.4f}\tTime {time.time() - start:.1f}"
        logger(progress.display(epoch+1, isPrint=False) + msg)
        # write on Tensorboard
        writer.add_scalar("train/loss", progress['loss'].avg, global_step=epoch+1)
        writer.add_scalar("train/acc", progress['train_acc'].avg, global_step=epoch+1)
        writer.add_scalar("test/acc", test_acc, global_step=epoch+1)

        if (epoch+1) % 10 == 0 or (epoch+1) == args.epochs:
            l_emp = empirical_local_lipschitzity(model, test_batches, early_stop=True).item()
            logger(f"[{args.backbone}] EPOCH {epoch+1} : --- Empirical Lipschitzity: {l_emp}")
            writer.add_scalar("Lipschitz", l_emp, global_step=epoch+1)
        if (epoch+1) % 10 == 0 or (epoch+1) == args.epochs:
            cert_right, cert_wrong, insc_right, insc_wrong = cert_stats(model, test_batches, eps * 2**0.5, full=True)
            logger(f"[{args.backbone}] (PROVABLE) Certifiably Robust (eps: {eps:.4f}): {cert_right:.4f}, " + 
                    f"Cert. Wrong: {cert_wrong:.4f}, Insc. Right: {insc_right:.4f}, Insc. Wrong: {insc_wrong:.4f}"
            )
        if (epoch+1) % 10 == 0 or (epoch+1) == args.epochs:
            val_rob_acc = rob_acc(test_batches, model, eps, alpha, opt, False, 10, 1, linf_proj=False, l2_grad_update=True)[0]
            logger(f"[{args.backbone}] (EMPIRICAL) Robust accuracy (eps: {eps:.4f}): {val_rob_acc:.4f}")
        
        ### Report for Hyper-Parmater Tunning
        tune.report(train_loss=progress['loss'].avg, train_acc=progress['train_acc'].avg, test_acc=test_acc, cert_right=cert_right, val_rob_acc=val_rob_acc)
        progress.reset()
    logger.log_time()

    torch.save({
        'args': args,
        'epoch': epoch+1,
        'state_dict': model.state_dict()
    }, os.path.join(args.log_dir, "model.pth"))

    logger(f"Elapsed Time: {(time.time() - start_main)/60:.1f} Min")


if __name__ == '__main__':
    args = get_option()
    main_for_tune(args=args)
