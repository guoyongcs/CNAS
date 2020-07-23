import os
import math
import random
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets

import cnas.model.genotypes as genotypes
import cnas.torchutils as utils

from cnas.search.core.metric import MovingAverageMetric
from cnas.search.core.engine import update_w, update_theta, derive_test
from cnas.search.core.config import args
from cnas.evaluation.core.transforms import get_cifar10_transforms
from cnas.model.search import NASNetwork
from cnas.model.master import MasterPairs, ArchMaster
from cnas.torchutils import output_directory
from cnas.torchutils import summary_writer as writer
from cnas.torchutils.common import get_last_commit_id, set_reproducible, generate_random_seed


device = "cuda:0" if torch.cuda.is_available() else "cpu"


def search_entry(args):
    utils.logger.info(f"commit id: {get_last_commit_id()}")
    if args.seed is None:
        seed = generate_random_seed()
    else:
        seed = args.seed
    utils.logger.info(f"set random seed to {seed}")
    set_reproducible(seed=seed)
    NUM_CLASSES = 10
    train_transform, val_transform = get_cifar10_transforms(cutout=None)
    trainset = torchvision.datasets.CIFAR10(root=args.data, train=True,
                                            download=True, transform=train_transform)
    valset = torchvision.datasets.CIFAR10(root=args.data, train=False,
                                          download=True, transform=val_transform)
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(len(trainset), generator=g).tolist()
    split = int(math.floor(args.train_portion * len(trainset)))
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=False, num_workers=args.num_workers
    )

    val_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:len(trainset)]),
        pin_memory=False, num_workers=args.num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size,
        shuffle=False, pin_memory=False, num_workers=args.num_workers
    )
    if args.debug:
        test_loader = val_loader = train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:96]),
            pin_memory=False, num_workers=args.num_workers
        )

    criterion = nn.CrossEntropyLoss()
    primitives = getattr(genotypes, args.search_space)

    mode = args.mode
    base_op, *remain = primitives
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(len(remain), generator=g).tolist()
    primitives = [base_op] + [remain[i] for i in indices]
    # if args.n_ops is not None:
    #     premitives = primitives[:args.n_ops]
    utils.logger.info(f"All avalible ops are {primitives}.")

    architecture: nn.Module = NASNetwork(
        C=args.init_channels,
        num_classes=NUM_CLASSES,
        layers=args.layers,
        n_nodes=args.n_nodes,
        multiplier=args.multiplier,
        stem_multiplier=args.stem_multiplier,
        loose_end=args.loose_end,
        op_type=primitives
    )
    architecture = architecture.to(device=device)

    cur_nodes = 1 if args.mode == "CNAS_NODE" else args.n_nodes
    cur_ops = 1 if args.mode == "CNAS_OP" else args.n_ops
    master = MasterPairs(
        normal_arch_master=ArchMaster(n_nodes=args.n_nodes,
                                      cur_nodes=cur_nodes,
                                      n_ops=args.n_ops,
                                      cur_ops=cur_ops,
                                      device=device,
                                      hidden_size=args.hidden_size,
                                      temperature=args.temperature,
                                      tanh_constant=args.tanh_constant,
                                      op_tanh_reduce=args.op_tanh_reduce),
        reduce_arch_master=ArchMaster(n_nodes=args.n_nodes,
                                      cur_nodes=cur_nodes,
                                      n_ops=args.n_ops,
                                      cur_ops=cur_ops,
                                      device=device,
                                      hidden_size=args.hidden_size,
                                      temperature=args.temperature,
                                      tanh_constant=args.tanh_constant,
                                      op_tanh_reduce=args.op_tanh_reduce),
    )
    master = master.to(device=device)

    arch_optimizer = optim.SGD(
        architecture.parameters(),
        lr=args.arch_learning_rate,
        momentum=args.arch_momentum,
        weight_decay=args.arch_weight_decay,
        nesterov=True
    )
    master_optimizer = optim.Adam(
        master.parameters(),
        lr=args.controller_learning_rate,
        weight_decay=args.controller_weight_decay,
    )

    if mode == "CNAS_NODE":
        n_stages = args.n_nodes
    elif mode == "CNAS_OP":
        n_stages = args.n_ops
    elif mode == "CNAS_FIX":
        n_stages = 1
    else:
        raise ValueError()

    baseline = MovingAverageMetric(gamma=args.baseline_moving_gamma)

    conduct_pipeline(
        mode,
        n_stages,
        args.epochs_for_warmup,
        args.epoch_per_stage,
        args.master_start_traning_epoch,
        device,
        train_loader,
        val_loader,
        test_loader,
        primitives,
        master,
        architecture,
        criterion,
        arch_optimizer,
        master_optimizer,
        args.update_w_force_uniform,
        baseline,
        args.entropy_coeff,
        args.grad_clip,
        writer,
        args.report_freq,
    )


def conduct_pipeline(mode, n_stages, epochs_for_warmup, epoch_per_stage,
                     master_start_traning_epoch, device,
                     train_loader, val_loader, test_loader, primitives,
                     master, architecture, criterion, arch_optimizer, master_optimizer,
                     update_w_force_uniform, baseline, entropy_coeff, grad_clip,
                     writer, log_frequency):
    total_epoch = 0
    for stage in range(1, n_stages+1, 1):
        # warmup
        for epoch in range(1, epochs_for_warmup+1, 1):
            update_w(epoch=epoch, data_loader=train_loader, device=device,
                     master_pair=master, architecture=architecture,
                     criterion=criterion, optimizer=arch_optimizer,
                     force_uniform=True,
                     writer=writer, log_frequency=log_frequency)
        # normal training
        for epoch in range(1, epoch_per_stage+1, 1):
            total_epoch += 1
            is_stage_end = epoch == epoch_per_stage
            update_w(epoch=total_epoch, data_loader=train_loader, device=device,
                     master_pair=master, architecture=architecture,
                     criterion=criterion, optimizer=arch_optimizer,
                     force_uniform=update_w_force_uniform,
                     writer=writer, log_frequency=log_frequency)
            if epoch > master_start_traning_epoch:
                update_theta(epoch=total_epoch, baseline=baseline, entropy_coeff=entropy_coeff,
                             grad_clip=grad_clip, data_loader=val_loader, device=device,
                             master_pair=master, architecture=architecture,
                             optimizer=master_optimizer, writer=writer,
                             log_frequency=log_frequency)
                if is_stage_end:
                    os.makedirs(os.path.join(output_directory, "supernet",), exist_ok=True)
                    torch.save(architecture.state_dict(),
                               os.path.join(output_directory, "supernet",
                                            f"supernet-{stage}-{epoch}.pth"))
                    os.makedirs(os.path.join(output_directory, "controller",), exist_ok=True)
                    torch.save(master.state_dict(),
                               os.path.join(output_directory, "controller",
                                            f"controller-{stage}-{epoch}.pth"))
                    derive_test(stage=stage, epoch=total_epoch, save_path=output_directory,
                                n_node=master.normal_arch_master.cur_nodes, primitives=primitives,
                                data_loader=test_loader, device=device,
                                master_pair=master, architecture=architecture,
                                force_uniform=False, writer=writer)
            if mode == "CNAS_NODE":
                master.inc_n_nodes()
            elif mode == "CNAS_OP":
                master.inc_n_ops()
            elif mode == "CNAS_FIX":
                pass
            else:
                raise ValueError()
