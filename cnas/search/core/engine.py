import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from cnas.model.genotypes import Genotype
from cnas.model.master import MasterPairs
from cnas.model.search import NASNetwork
from cnas.search.core.metric import AverageMetric, AccuracyMetric, MovingAverageMetric, Statistics
from cnas.search.core.arch_sample import ArchSample
from cnas.search.core.visualization import draw_genotype

import cnas.torchutils as utils


class SaveID:
    id = 0


def update_w(epoch: int, data_loader, device: str, master_pair: MasterPairs, architecture: NASNetwork,
             criterion: nn.Module, optimizer: optim.Optimizer, force_uniform: bool,
             writer: SummaryWriter, log_frequency: int):
    start = datetime.now()
    loss_metric = AverageMetric()
    accuracy_metric = AccuracyMetric(topk=(1, 5))
    normal_logp_metric = AverageMetric()
    node_normal_entropy_metric = AverageMetric()
    op_normal_entropy_metric = AverageMetric()
    reduced_logp_metric = AverageMetric()
    node_reduced_entropy_metric = AverageMetric()
    op_reduced_entropy_metric = AverageMetric()

    master_pair.set_force_uniform(force_uniform=force_uniform)

    for iter_, (datas, targets) in enumerate(data_loader, start=1):
        datas, targets = datas.to(device=device), targets.to(device=device)
        with torch.no_grad():
            (normal_arch, normal_logp, node_normal_entropy, op_normal_entropy), \
                (reduced_arch, reduced_logp, node_reduced_entropy,
                 op_reduced_entropy) = master_pair()

        outputs = architecture(datas, normal_arch, reduced_arch)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update metrics
        loss_metric.update(loss)
        accuracy_metric.update(targets, outputs)
        normal_logp_metric.update(normal_logp)
        node_normal_entropy_metric.update(node_normal_entropy)
        op_normal_entropy_metric.update(op_normal_entropy)
        reduced_logp_metric.update(reduced_logp)
        node_reduced_entropy_metric.update(node_reduced_entropy)
        op_reduced_entropy_metric.update(op_reduced_entropy)

        # iteration log
        if iter_ % log_frequency == 0 or iter_ == len(data_loader):
            message = f"UPDATE W, epoch={epoch:03d}, iter={iter_}/{len(data_loader)}, "
            message += f"celoss={loss_metric.last:.4f}({loss_metric.value:.4f}), "
            message += f"accuracy@1={accuracy_metric.last_accuracy(1).rate*100:.2f}%"
            message += f"({accuracy_metric.accuracy(1).rate*100:.2f}%), "
            message += f"accuracy@5={accuracy_metric.last_accuracy(5).rate*100:.2f}%"
            message += f"({accuracy_metric.accuracy(5).rate*100:.2f}%), "
            message += f"normal_logp={normal_logp_metric.last:.4f}({normal_logp_metric.value:.4f}), "
            message += f"node_normal_entropy={node_normal_entropy_metric.last:.4f}({node_normal_entropy_metric.value:.4f}), "
            message += f"op_normal_entropy={op_normal_entropy_metric.last:.4f}({op_normal_entropy_metric.value:.4f}), "
            message += f"reduced_logp={reduced_logp_metric.last:.4f}({reduced_logp_metric.value:.4f}), "
            message += f"node_reduced_entropy={node_reduced_entropy_metric.last:.4f}({node_reduced_entropy_metric.value:.4f}), "
            message += f"op_reduced_entropy={op_reduced_entropy_metric.last:.4f}({op_reduced_entropy_metric.value:.4f})."
            if iter_ == len(data_loader):
                message += f" Eplased time={datetime.now()-start}."
            utils.logger.info(message)

    writer.add_scalar("update_w/celoss", loss_metric.value, epoch)
    writer.add_scalar("update_w/accuracy@1",
                      accuracy_metric.accuracy(1).rate, epoch)
    writer.add_scalar("update_w/accuracy@5",
                      accuracy_metric.accuracy(5).rate, epoch)
    writer.add_scalar("update_w/normal_logp",
                      normal_logp_metric.value, epoch)
    writer.add_scalar("update_w/node_normal_entropy",
                      node_normal_entropy_metric.value, epoch)
    writer.add_scalar("update_w/op_normal_entropy",
                      op_normal_entropy_metric.value, epoch)
    writer.add_scalar("update_w/reduced_logp",
                      reduced_logp_metric.value, epoch)
    writer.add_scalar("update_w/node_reduced_entropy",
                      node_reduced_entropy_metric.value, epoch)
    writer.add_scalar("update_w/op_reduced_entropy",
                      op_reduced_entropy_metric.value, epoch)


def update_theta(epoch: int, baseline: MovingAverageMetric,
                 entropy_coeff, grad_clip: int,
                 data_loader, device: str,
                 master_pair: MasterPairs, architecture: NASNetwork,
                 optimizer: optim.Optimizer, writer: SummaryWriter, log_frequency: int):
    start = datetime.now()
    policy_loss_metric = AverageMetric()
    accuracy_metric = AccuracyMetric(topk=(1, 5))
    normal_logp_metric = AverageMetric()
    node_normal_entropy_metric = AverageMetric()
    op_normal_entropy_metric = AverageMetric()
    reduced_logp_metric = AverageMetric()
    node_reduced_entropy_metric = AverageMetric()
    op_reduced_entropy_metric = AverageMetric()

    node_normal_entropy_coeff, node_reduced_entropy_coeff, \
        op_normal_entropy_coeff, op_reduced_entropy_coeff = [entropy_coeff, ]*4

    master_pair.unset_force_uniform()

    for iter_, (datas, targets) in enumerate(data_loader, start=1):
        datas, targets = datas.to(device=device), targets.to(device=device)
        (normal_arch, normal_logp, node_normal_entropy, op_normal_entropy), \
            (reduced_arch, reduced_logp, node_reduced_entropy,
             op_reduced_entropy) = master_pair()
        with torch.no_grad():
            outputs = architecture(datas, normal_arch, reduced_arch)
        accuracy_metric.update(targets, outputs)
        accuracy_1 = accuracy_metric.last_accuracy(1).rate
        baseline.update(accuracy_1)
        reward = accuracy_1 - baseline.value
        policy_loss = -(normal_logp + reduced_logp) * reward \
            - (node_normal_entropy*node_normal_entropy_coeff
               + op_normal_entropy*op_normal_entropy_coeff
                + node_reduced_entropy*node_reduced_entropy_coeff
                + op_reduced_entropy*op_reduced_entropy_coeff)

        optimizer.zero_grad()
        policy_loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(master_pair.parameters(), grad_clip)
        optimizer.step()

        # update metrics
        policy_loss_metric.update(policy_loss)
        normal_logp_metric.update(normal_logp)
        node_normal_entropy_metric.update(node_normal_entropy)
        op_normal_entropy_metric.update(op_normal_entropy)
        reduced_logp_metric.update(reduced_logp)
        node_reduced_entropy_metric.update(node_reduced_entropy)
        op_reduced_entropy_metric.update(op_reduced_entropy)

        # iteration log
        if iter_ % log_frequency == 0 or iter_ == len(data_loader):
            message = f"UPDATE THETA, epoch={epoch:03d}, Iter={iter_}/{len(data_loader)}, "
            message += f"reward={reward:.4f}, "
            message += f"pocily loss={policy_loss_metric.last:.4f}({policy_loss_metric.value:.4f}), "
            message += f"moving accuracy={baseline.value*100:.2f}%, "
            message += f"normal_logp={normal_logp_metric.last:.4f}({normal_logp_metric.value:.4f}), "
            message += f"node_normal_entropy={node_normal_entropy_metric.last:.4f}({node_normal_entropy_metric.value:.4f}), "
            message += f"op_normal_entropy={op_normal_entropy_metric.last:.4f}({op_normal_entropy_metric.value:.4f}), "
            message += f"reduced_logp={reduced_logp_metric.last:.4f}({reduced_logp_metric.value:.4f}), "
            message += f"node_reduced_entropy={node_reduced_entropy_metric.last:.4f}({node_reduced_entropy_metric.value:.4f}), "
            message += f"op_reduced_entropy={op_reduced_entropy_metric.last:.4f}({op_reduced_entropy_metric.value:.4f})."
            if iter_ == len(data_loader):
                message += f" Eplased time={datetime.now()-start}."
            utils.logger.info(message)

    writer.add_scalar("update_theta/policy_loss",
                      policy_loss_metric.value, epoch)
    writer.add_scalar("update_theta/baseline", baseline.value, epoch)
    writer.add_scalar("update_theta/accuracy@1",
                      accuracy_metric.accuracy(1).rate, epoch)
    writer.add_scalar("update_theta/accuracy@5",
                      accuracy_metric.accuracy(5).rate, epoch)
    writer.add_scalar("update_theta/normal_logp",
                      normal_logp_metric.value, epoch)
    writer.add_scalar("update_theta/node_normal_entropy",
                      node_normal_entropy_metric.value, epoch)
    writer.add_scalar("update_theta/op_normal_entropy",
                      op_normal_entropy_metric.value, epoch)
    writer.add_scalar("update_theta/reduced_logp",
                      reduced_logp_metric.value, epoch)
    writer.add_scalar("update_theta/node_reduced_entropy",
                      node_reduced_entropy_metric.value, epoch)
    writer.add_scalar("update_theta/op_reduced_entropy",
                      op_reduced_entropy_metric.value, epoch)


TIMES = (3, 10)


def derive_test(stage: int, epoch: int, save_path: str,
                n_node, primitives, data_loader, device: str,
                master_pair: MasterPairs, architecture: NASNetwork,
                force_uniform: bool, writer: SummaryWriter):
    utils.logger.info(f"DERIVE TEST, epoch={epoch}, times={TIMES}")
    repeat_times, derive_times = TIMES
    start = datetime.now()

    acc_1_statistics = Statistics()
    acc_5_statistics = Statistics()
    normal_logp_statistics = Statistics()
    node_normal_entropy_statistics = Statistics()
    op_normal_entropy_statistics = Statistics()
    reduce_logp_statistics = Statistics()
    node_reduce_entropy_statistics = Statistics()
    op_reduce_entropy_statistics = Statistics()

    master_pair.set_force_uniform(force_uniform=force_uniform)

    with open(os.path.join(save_path, "visulization.md"), "a+") as f:
        for i in range(1, repeat_times+1, 1):
            best_candidate_index = 0
            best_acc = 0.
            best_arch = None
            for j in range(1, derive_times+1, 1):
                with torch.no_grad():
                    (normal_arch, normal_logp, node_normal_entropy, op_normal_entropy),\
                        (reduced_arch, reduced_logp, node_reduced_entropy, op_reduced_entropy) = master_pair()

                normal_logp_statistics.update(normal_logp)
                node_normal_entropy_statistics.update(node_normal_entropy)
                op_normal_entropy_statistics.update(op_normal_entropy)
                reduce_logp_statistics.update(reduced_logp)
                node_reduce_entropy_statistics.update(node_reduced_entropy)
                op_reduce_entropy_statistics.update(op_reduced_entropy)

                accuracy_metric = AccuracyMetric(topk=(1, 5))
                for _, (datas, targets) in enumerate(data_loader):
                    datas, targets = datas.to(device=device), targets.to(device=device)
                    with torch.no_grad():
                        outputs = architecture(datas, normal_arch, reduced_arch)
                    accuracy_metric.update(targets, outputs)
                acc_1_statistics.update(accuracy_metric.accuracy(1).rate)
                acc_5_statistics.update(accuracy_metric.accuracy(5).rate)
                acc = accuracy_metric.accuracy(1).rate
                if acc > best_acc:
                    best_acc = acc
                    best_candidate_index = i
                    best_arch = Genotype.from_ordinal_arch(ordinal_normal_arch=normal_arch,
                                                           ordinal_reduced_arch=reduced_arch,
                                                           primitives=primitives)
                    best_normal_logp = normal_logp.item()
                    best_reduced_logp = reduced_logp.item()
                    best_node_normal_entropy = node_normal_entropy.item()
                    best_op_normal_entropy = op_normal_entropy.item()
                    best_node_reduced_entropy = node_reduced_entropy.item()
                    best_op_reduced_entropy = op_reduced_entropy.item()
            if save_path is not None:
                sample_saved_path = os.path.join(save_path, "derive",
                                                 f"stage-{stage}", f"epoch-{epoch}")
                arch_path = os.path.join("derive", f"stage-{stage}", f"epoch-{epoch}")
                os.makedirs(sample_saved_path, exist_ok=True)
                ArchSample(arch=str(best_arch), derived_acc=acc,
                           normal_logp=best_normal_logp,
                           reduced_logp=best_reduced_logp,
                           node_normal_entropy=best_node_normal_entropy,
                           op_normal_entropy=best_op_normal_entropy,
                           node_reduced_entropy=best_node_reduced_entropy,
                           op_reduced_entropy=best_op_reduced_entropy
                           ).to_file(os.path.join(sample_saved_path, f"arch-{i}.json"), transform=True)
                draw_genotype(best_arch.named_normal_arch, n_node,
                              os.path.join(sample_saved_path, f"normal_arch-{i}"))
                draw_genotype(best_arch.named_reduced_arch, n_node,
                              os.path.join(sample_saved_path, f"reduced_arch-{i}"))
                f.write(f"## Satge={stage}, Epoch={epoch}, Sample ID={i}\n\n")
                f.write(f"### Normal Architecture\n\n")
                f.write(f'<p align="center">\n')
                f.write(f'<img src={arch_path}/normal_arch-{i}.svg>\n')
                f.write(f'</p>\n\n')
                f.write(f"### Reduced Architecture\n\n")
                f.write(f'<p align="center">\n')
                f.write(f'<img src={arch_path}/reduced_arch-{i}.svg>\n')
                f.write(f'</p>\n\n')

            utils.logger.info(f"DERIVE TEST, epoch={epoch:03d}, "
                              f"candidate architecture {i:02d}, "
                              f"accuracy@1={accuracy_metric.accuracy(1).rate*100:.2f}%, "
                              f"accuracy@5={accuracy_metric.accuracy(5).rate*100:.2f}%, "
                              f"normal_logp={normal_logp.item():.4f}, "
                              f"node_normal_entropy={node_normal_entropy.item():.4f}, "
                              f"op_normal_entropy={op_normal_entropy.item():.4f}, "
                              f"reduced_logp={reduced_logp.item():.4f}, "
                              f"node_reduced_entropy={node_reduced_entropy.item():.4f}, "
                              f"op_reduced_entropy={op_reduced_entropy.item():.4f}.")
    utils.logger.info(f"DERIVE TEST Complete, epoch={epoch:03d}, "
                      f"best candidate architecture is No.{best_candidate_index:02d}, "
                      f"best accuracy@1={best_acc*100:.2f}%, "
                      f"accuracy@1={acc_1_statistics.avg*100:.2f}±{acc_1_statistics.std*100:.2f}%, "
                      f"accuracy@5={acc_5_statistics.avg*100:.2f}±{acc_5_statistics.std*100:.2f}%, "
                      f"normal_logp={normal_logp_statistics.avg:.4f}±{normal_logp_statistics.std:.4f}, "
                      f"node_normal_entropy={node_normal_entropy_statistics.avg:.4f}±{node_normal_entropy_statistics.std:.4f}, "
                      f"op_normal_entropy={op_normal_entropy_statistics.avg:.4f}±{op_normal_entropy_statistics.std:.4f}, "
                      f"reduced_logp={reduce_logp_statistics.avg:.4f}±{reduce_logp_statistics.std:.4f}, "
                      f"node_reduced_entropy={node_reduce_entropy_statistics.avg:.4f}±{node_reduce_entropy_statistics.std:.4f}, "
                      f"op_reduced_entropy={op_reduce_entropy_statistics.avg:.4f}±{op_reduce_entropy_statistics.std:.4f}, "
                      f"eplased time={datetime.now()-start}.")
    writer.add_scalar("derive_test/best_accuracy@1", best_acc, epoch)
    writer.add_scalar("derive_test/avg_accuracy@1", acc_1_statistics.avg, epoch)
    writer.add_scalar("derive_test/avg_accuracy@5", acc_5_statistics.avg, epoch)
    writer.add_scalar("derive_test/normal_logp", normal_logp_statistics.avg, epoch)
    writer.add_scalar("derive_test/node_normal_entropy", node_normal_entropy_statistics.avg, epoch)
    writer.add_scalar("derive_test/op_normal_entropy", op_normal_entropy_statistics.avg, epoch)
    writer.add_scalar("derive_test/reduced_logp", reduce_logp_statistics.avg, epoch)
    writer.add_scalar("derive_test/node_reduced_entropy", node_reduce_entropy_statistics.avg, epoch)
    writer.add_scalar("derive_test/op_reduced_entropy", op_reduce_entropy_statistics.avg, epoch)
