import torch
from torch import Tensor

class Trainer(object):
    def __init__(self, option: Config):
        self.option = option
        self.image_size = option.image_size
        self.logger = logger_setting(option.exp_name, option.save_dir)
        self._build_model()
        self._set_optimizer()
        self._set_loss()
        self._print_net()
        if self.option.cuda:
            self._set_cuda()
        
        self.global_step = 0
        self.tb_writer: SummaryWriter

    def _build_model(self):
        if self.option.data in ['mnist', 'mnist-biased']:
            self.logger.info("[MODEL] Simple CNN for mnist")
            self.net = SimpleConvNet()
        elif self.option.data in ['cifar10', 'cifar10-c']:
            self.logger.info(f"[MODEL] ResNet-18 for cifar10-c")
            self.net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        elif self.option.data == 'celebA_hq':
            self.logger.info(f"[MODEL] ResNet-18 for celebA_hq")
            self.net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=2)

    def _set_optimizer(self):
        # parse optimizer args
        optimizer_arg = {
            'params':self.net._get_params(),
            'lr':self.option.lr,
            'weight_decay':self.option.weight_decay
        }

        if self.option.opt in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.option.momentum
        elif self.option.opt in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.option.opt == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.option.opt = 'Adam'
        elif self.option.opt == 'Adam':
            optimizer_arg['betas'] = (self.option.momentum, 0.999)

        # create optimizers
        self.optim = torch.optim.__dict__[self.option.opt](**optimizer_arg)

        return optimizer_arg

    def _set_loss(self):
        self.loss = nn.CrossEntropyLoss()

    def _print_net(self):
        self.logger.info(f"[PARAMETER:BASELINE]: {self._count_parameters(self.net)}")

    def _set_cuda(self):
        self.net.cuda()

    def train_task(self, train_loader, val_loader=None):
        self._mode_setting(is_train=True)
        start_epoch = 1
        for step in range(start_epoch, self.option.epoch+1):
            logs = self._train_one_epoch(train_loader, step)
            self._log_tensorboard(logs, self.global_step, tag='train')

            if step == 1 or step % self.option.save_step == 0 or step == self.option.epoch:
                test_acc = self._validate(val_loader, step=step, msg='[TEST]')
                if self.option.epoch == step:
                    self._save_model(step)

        return test_acc

    def _train_one_epoch(self, data_loader, step):
        self._mode_setting(is_train=True)
        
        loss_sum = 0.
        total_num_train = 0
        for images, labels in tqdm(data_loader):
            images = self._get_variable(images)
            labels = self._get_variable(labels)
            bsize = images.shape[0]
            total_num_train += bsize

            """
            Training Baseline
            """
            self.optim.zero_grad()
            pred_label = self.net(images)
            loss: Tensor = self.loss(pred_label, labels)
            loss_sum += bsize*loss.item()
            loss.backward()
            self.optim.step()

        avg_loss = loss_sum/total_num_train
        msg = f"[TRAIN][{step:>3}] BASE LOSS : {avg_loss:.4f}"
        self.logger.info(msg)
        self.global_step += 1

        return {'loss': avg_loss}

    @torch.no_grad()
    def _validate(self, data_loader, valid_type=None, step=None, msg="Validation"):
        self.logger.info(msg)
        self._mode_setting(is_train=False)

        total_num_correct = 0.
        total_num_test = 0.
        total_loss = 0.

        for images, labels in tqdm(data_loader):
            images = self._get_variable(images)
            labels = self._get_variable(labels)
            
            batch_size = images.shape[0]
            total_num_test += batch_size

            # self.optim.zero_grad()
            pred_label = self.net(images)
            loss = self.loss(pred_label, labels)
            
            total_num_correct += self._num_correct(pred_label,labels,topk=1).data
            total_loss += loss.data*batch_size

        avg_loss = total_loss/total_num_test
        avg_acc = float(total_num_correct)/total_num_test
        if valid_type != None:
            msg = f"[EVALUATION - {valid_type}] LOSS : {avg_loss:.4f}, ACCURACY : {avg_acc:.4f}"
        else:
            msg = f"[EVALUATION][{step:>3}] LOSS : {avg_loss:.4f}, ACCURACY : {avg_acc:.4f}"
        
        self.logger.info(msg)

        return avg_acc

    def _init_weights(self):
        self.net._init_weights()

    def _count_parameters(self, *models):
        num = 0
        for model in models:
            num += sum(p.numel() for p in model.parameters() if p.requires_grad)
        msg = f"{num/(1000000):.4f} M"
        return msg

    def _mode_setting(self, is_train=True):
        if is_train:
            self.net.train()
        else:
            self.net.eval()

    def _num_correct(self, outputs: Tensor, labels: Tensor, topk=1):
        _, preds = outputs.topk(k=topk, dim=1)
        preds = preds.t()
        correct = preds.eq(labels.view(1, -1).expand_as(preds))
        correct = correct.view(-1).sum()
        return correct

    def _accuracy(self, outputs: Tensor, labels: Tensor):
        batch_size = labels.size(0)
        _, preds = outputs.topk(k=1, dim=1)
        preds = preds.t()
        correct = preds.eq(labels.view(1, -1).expand_as(preds))
        correct = correct.view(-1).float().sum(0, keepdim=True)
        accuracy = correct.mul_(100.0 / batch_size)
        return accuracy

    def _save_model(self, step, task=0):
        if task > 0:
            save_path = os.path.join(self.option.save_dir, self.option.exp_name, 
                                    f'checkpoint_step_{step}_task_{task}.pth')
        else:
            save_path = os.path.join(self.option.save_dir, self.option.exp_name, 
                                    f'checkpoint_step_{step}.pth')
        torch.save({
            'step': step,
            'task': task,
            'optim_state_dict': self.optim.state_dict(),
            'net_state_dict': self.net.state_dict()
        }, save_path)

        print(f'[SAVE] checkpoint step: {step}')

    def _get_variable(self, inputs: Tensor) -> Tensor:
        if self.option.cuda:
            return Variable(inputs.cuda())
        return Variable(inputs)

    def _log_tensorboard(self, logs: Dict[str, float], step: int, tag=""):
        for key in logs.keys():
            name = f"{tag}/{key}" if tag else f"{key}"
            self.tb_writer.add_scalar(name, logs[key], global_step=step)
        self.tb_writer.flush()