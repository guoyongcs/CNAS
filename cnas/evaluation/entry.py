import functools
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import cnas.torchutils as utils
from cnas.evaluation.core.config import args
from cnas.model import genotypes
from cnas.model.eval import NetworkCIFAR, NetworkImageNet
from cnas.evaluation.core.transforms import get_cifar10_transforms, get_imagenet_transforms
from cnas.evaluation.core.criterion import LabelSmoothCrossEntropyLoss
from cnas.evaluation.core.engine import train, evaluate
from cnas.evaluation.scheduler import WarmupCosineAnnealingLR, WarmupLinearAnnealingLR
from cnas.torchutils.common import compute_flops, compute_nparam
from cnas.torchutils.distributed import is_dist_avail_and_init, local_rank
from cnas.torchutils.common import save_checkpoint
from cnas.torchutils.metrics import EstimatedTimeArrival


if args.dataset.lower() == "cifar10":
    NUM_CLASSES = 10
    build_train_dataset = functools.partial(datasets.CIFAR10, train=True)
    build_val_dataset = functools.partial(datasets.CIFAR10, train=False)
    build_transforms = get_cifar10_transforms
    build_network = NetworkCIFAR
    input_size = 32
elif args.dataset.lower() == "imagenet":
    NUM_CLASSES = 1000
    build_train_dataset = functools.partial(datasets.ImageNet, split="train")
    build_val_dataset = functools.partial(datasets.ImageNet, split="val")
    build_transforms = get_imagenet_transforms
    build_network = NetworkImageNet
    input_size = args.eval_size
else:
    raise ValueError(f"The dataset {args.dataset} is not registered.")


def eval_entry(args):
    # build model
    model_without_parallel = build_network(
        C=args.init_channels,
        num_classes=NUM_CLASSES,
        layers=args.layers,
        auxiliary=args.auxiliary_weight is not None,
        genotype=getattr(genotypes, args.arch)
    )
    n_params = compute_nparam(model_without_parallel, skip_pattern="auxiliary")
    flops = compute_flops(model_without_parallel, (1, 3, input_size, input_size),
                          skip_pattern="auxiliary")
    utils.logger.info(f"n_params={n_params/1000**2:.2f}M, "
                      f"Madds={flops/1000**2:.2f}M.")

    torch.cuda.set_device(local_rank())
    model_without_parallel = model_without_parallel.cuda()
    model = DDP(model_without_parallel, device_ids=[local_rank()]) \
        if is_dist_avail_and_init() else model_without_parallel

    criterion = LabelSmoothCrossEntropyLoss(NUM_CLASSES) if args.label_smooth \
        else nn.CrossEntropyLoss()

    # build optimizer
    group_weight = []
    group_bias = []
    for name, param in model.named_parameters():
        if 'bias' in name:
            group_bias.append(param)
        else:
            group_weight.append(param)
    assert len(list(model.parameters())) == len(group_weight) + len(group_bias)
    optimizer = torch.optim.SGD([
        {'params': group_weight},
        {'params': group_bias, 'weight_decay': 0 if args.no_bias_decay else args.weight_decay}
    ], lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    # build data loader
    train_transform, val_transform = build_transforms()

    train_dataset = build_train_dataset(root=args.data, transform=train_transform)
    val_dataset = build_val_dataset(root=args.data, transform=val_transform)
    
    if is_dist_avail_and_init():
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), pin_memory=False,
        sampler=train_sampler, num_workers=args.num_workers)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=(val_sampler is None), pin_memory=False,
        sampler=val_sampler, num_workers=args.num_workers)

    epoch = 0
    max_epochs = args.max_epochs
    ETA = EstimatedTimeArrival(max_epochs)

    top1_accuracy, top5_accuracy = 0.0, 0.0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu")
        start_epoch = ckpt["epoch"]
        best_acc1 = ckpt["best_acc1"]
        best_acc5 = ckpt["best_acc5"]
        best_epoch = ckpt["best_epoch"]
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        utils.logger(f"Load chckpoint from {args.resume}, epoch={epoch}, best_acc1={best_acc1*100:.2f}%.")
    else:
        start_epoch = 0
        best_epoch = 0
        best_acc1, best_acc5 = 0.0, 0.0
        
    if args.scheduler =="cosine":
        scheduler = WarmupCosineAnnealingLR(optimizer, args.warmup_epochs, max_epochs)
    elif args.scheduler =="linear":
        scheduler = WarmupLinearAnnealingLR(optimizer, args.warmup_epochs, max_epochs)
    else:
        raise ValueError("Unkown schduler type.")

    for epoch in range(start_epoch+1, max_epochs+1):
        for func, loader in zip((train, evaluate), (train_loader, val_loader)):
            name = func.__name__.upper()
            loss, (top1_accuracy, top5_accuracy) = func(epoch, max_epochs, model, loader, criterion,
                                                        optimizer, scheduler, args.drop_path_prob,
                                                        args.auxiliary_weight, args.grad_clip,
                                                        args.report_freq)
            utils.summary_writer.add_scalar(f"{name}/loss", loss, epoch)
            utils.summary_writer.add_scalar(f"{name}/acc_1", top1_accuracy, epoch)
            utils.summary_writer.add_scalar(f"{name}/acc_5", top5_accuracy, epoch)
            utils.summary_writer.add_scalar(f"{name}/lr", optimizer.param_groups[0]['lr'], epoch)
            utils.logger.info(", ".join([
                f"{name} Complete",
                f"epoch={epoch:04d}",
                f"loss={loss:.4f}",
                f"top1-accuracy={top1_accuracy*100:.2f}%",
                f"top5-accuracy={top5_accuracy*100:.2f}%",
            ]))
        ETA.step()
        utils.logger.info(", ".join([
            f"Epoch Complete",
            f"epoch={epoch:03d}",
            f"best acc={best_acc1*100:.2f}%/{best_acc5*100:.2f}%(epoch={best_epoch:03d})",
            f"eta={ETA.remaining_time}",
            f"arrival={ETA.arrival_time}",
            f"cost={ETA.cost_time}",
        ]))
        if best_acc1 < top1_accuracy:
            best_acc1 = top1_accuracy
            best_acc5 = top5_accuracy
            best_epoch = epoch
        save_checkpoint(utils.output_directory, epoch, model, optimizer,
                        best_acc1, best_acc5, best_epoch)
