import os
import random
import subprocess
import numpy
import torch
import torch.nn as nn
import torch.optim as optim

from .distributed import torchsave


device = "cuda:0" if torch.cuda.is_available() else "cpu"

def generate_random_seed():
    return int.from_bytes(os.urandom(2), byteorder="little", signed=False)


def set_reproducible(seed=0):
    '''
    To ensure the reproducibility, refer to https://pytorch.org/docs/stable/notes/randomness.html.
    Note that completely reproducible results are not guaranteed.
    '''
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_cudnn_auto_tune():
    torch.backends.cudnn.benchmark = True


def compute_nparam(module: nn.Module, skip_pattern):
    n_param = 0
    for name, p in module.named_parameters():
        if skip_pattern not in name:
            n_param += p.numel()
    return n_param


def compute_flops(module: nn.Module, size, skip_pattern):
    def size_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        *_, h, w = output.shape
        module.output_size = (h, w)
    hooks = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(size_hook))
    with torch.no_grad():
        training = module.training
        module.eval()
        module(torch.rand(size))
        module.train(mode=training)
    for hook in hooks:
        hook.remove()

    flops = 0
    for name, m in module.named_modules():
        if skip_pattern in name:
            continue
        if isinstance(m, nn.Conv2d):
            h, w = m.output_size
            kh, kw = m.kernel_size
            flops += h * w * m.in_channels * m.out_channels * kh * kw / m.groups
        if isinstance(module, nn.Linear):
            flops += m.in_features * m.out_features
    return flops


def get_last_commit_id():
    try:
        commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8")
        commit_id = commit_id.strip()
        return commit_id
    except subprocess.CalledProcessError as e:
        out_bytes = e.output
        code = e.returncode
        return None


def save_checkpoint(output_directory, epoch, model: nn.Module,
                    optimizer: optim.Optimizer, best_acc1, best_acc5, best_epoch):
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model_without_parallel = model.module
    else:
        model_without_parallel = model
    ckpt = dict(
        epoch=epoch,
        state_dict=model_without_parallel.state_dict(),
        optimizer=optimizer.state_dict(),
        best_acc1=best_acc1,
        best_acc5=best_acc5,
    )
    torchsave(ckpt, os.path.join(output_directory, "checkpoint.pth"))
    if epoch == best_epoch:
        torchsave(ckpt, os.path.join(output_directory, "best.pth"))

class GradientAccumulator:
    def __init__(self, steps=1):
        self.steps = steps
        self._counter = 0

    @property
    def counter(self):
        return self._counter

    def inc_counter(self):
        self._counter += 1
        self._counter %= self.steps

    @property
    def is_start_cycle(self):
        return self._counter == 0

    @property
    def is_end_cycle(self):
        return self._counter == self.steps - 1

    def bw_step(self, loss: torch.Tensor, optimizer: optim.Optimizer):
        if optimizer is None:
            return

        loss.backward(gradient=1/self.steps)
        if self.is_start_cycle:
            optimizer.zero_grad()
        if self.is_end_cycle:
            optimizer.step()

        self.inc_counter()
