import math


class WarmupLinearAnnealingLR(object):
    def __init__(self, optimizer, T_warmup, T_max, eta_min=0, last_epoch=-1):
        self.T_warmup = T_warmup
        self.T_max = T_max
        self.eta_min = eta_min

        self.optimizer = optimizer

        self.base_lr = optimizer.param_groups[0]["lr"]

    def step(self, epoch):
        if epoch < self.T_warmup:
            lr = self.base_lr * epoch / self.T_warmup
        elif self.T_max - epoch > 5:
            lr = self.base_lr * (self.T_warmup - 5 - epoch) / (self.T_warmup - 5)
        else:
            lr = self.base_lr * (self.T_warmup - 5 - epoch) / ((self.T_warmup - 5)*5)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
