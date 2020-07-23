import time

import torch
import torch.nn as nn

import cnas.torchutils as utils
from cnas.torchutils.common import device
from cnas.torchutils.distributed import world_size
from cnas.torchutils.metrics import AverageMetric, AccuracyMetric, EstimatedTimeArrival


class SpeedTester():
    def __init__(self):
        self.reset()

    def reset(self):
        self.batch_size = 0
        self.start = time.perf_counter()

    def update(self, tensor):
        batch_size, *_ = tensor.shape
        self.batch_size += batch_size
        self.end = time.perf_counter()

    def compute(self):
        if self.batch_size == 0:
            return 0
        else:
            return self.batch_size/(self.end-self.start)


def train(epoch, max_epochs, model, loader, criterion, optimizer, scheduler,
          drop_path_prob, auxiliary_weight, grad_clip, report_freq):
    model.train()

    loader_len = len(loader)
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    loss_metric = AverageMetric()
    accuracy_metric = AccuracyMetric(topk=(1, 5))
    ETA = EstimatedTimeArrival(loader_len)
    speed_tester = SpeedTester()

    if scheduler is not None:
        scheduler.step(epoch)

    utils.logger.info(f"Train start, epoch={epoch:04d}, lr={optimizer.param_groups[0]['lr']:.6f}")

    for iter_, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        logits, logits_aux = model(inputs, drop_path_prob*epoch/max_epochs)
        loss = criterion(logits, targets)
        if auxiliary_weight is not None:
            loss += auxiliary_weight * criterion(logits_aux, targets)

        optimizer.zero_grad()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_metric.update(loss)
        accuracy_metric.update(logits, targets)
        ETA.step()
        speed_tester.update(inputs)

        if iter_ % report_freq == 0 or iter_ == loader_len-1:
            utils.logger.info(", ".join([
                "Train",
                f"epoch={epoch:04d}",
                f"iter={iter_:05d}/{loader_len:05d}",
                f"speed={speed_tester.compute()*world_size():.2f} images/s",
                f"loss={loss_metric.compute():.4f}",
                f"top1-accuracy={accuracy_metric.at(1).rate*100:.2f}%",
                f"top5-accuracy={accuracy_metric.at(5).rate*100:.2f}%",
                f"ETA={ETA.remaining_time}",
                f"cost={ETA.cost_time}" if iter_ == loader_len - 1 else "",
            ]))
            speed_tester.reset()

    return loss_metric.compute(), (accuracy_metric.at(1).rate, accuracy_metric.at(5).rate)


def evaluate(epoch, max_epochs, model, loader, critirion, optimizer, scheduler,
             drop_path_prob, auxiliary_weight, grad_clip, report_freq):
    model.eval()

    loader_len = len(loader)
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    loss_metric = AverageMetric()
    accuracy_metric = AccuracyMetric(topk=(1, 5))
    ETA = EstimatedTimeArrival(loader_len)
    speed_tester = SpeedTester()

    for iter_, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            logits, _ = model(inputs, drop_path_prob)
            loss = critirion(logits, targets)

        loss_metric.update(loss)
        accuracy_metric.update(logits, targets)
        ETA.step()
        speed_tester.update(inputs)

        if iter_ % report_freq == 0 or iter_ == loader_len-1:
            utils.logger.info(", ".join([
                "EVAL",
                f"epoch={epoch:04d}",
                f"iter={iter_:05d}/{loader_len:05d}",
                f"speed={speed_tester.compute()*world_size():.2f} images/s",
                f"loss={loss_metric.compute():.4f}",
                f"top1-accuracy={accuracy_metric.at(1).rate*100:.2f}%",
                f"top5-accuracy={accuracy_metric.at(5).rate*100:.2f}%",
                f"ETA={ETA.remaining_time}",
                f"cost={ETA.cost_time}" if iter_ == loader_len - 1 else "",
            ]))
            speed_tester.reset()

    return loss_metric.compute(), (accuracy_metric.at(1).rate, accuracy_metric.at(5).rate)
