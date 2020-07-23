'''
The cell and network definition of neural network search mode.
'''

import torch
import torch.nn as nn

from cnas.model.operation import OPERATIONS, FactorizedReduce, ReLUConvBN


class NASOp(nn.Module):
    def __init__(self, C, stride, op_type):
        super(NASOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in op_type:
            op = OPERATIONS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, index):
        return self._ops[index](x)


class NASCell(nn.Module):
    def __init__(self, n_nodes,
                 multiplier,
                 C_prev_prev, C_prev, C, reduction,
                 reduction_prev, loose_end=False, concat=None, op_type=None):
        super(NASCell, self).__init__()
        self.n_nodes = n_nodes
        self.multiplier = multiplier
        self.C = C
        self.reduction = reduction
        self.loose_end = loose_end

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)

        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

        self._ops = nn.ModuleList()
        for i in range(self.n_nodes):
            for j in range(i + 2):
                stride = 2 if reduction and j < 2 else 1
                op = NASOp(C, stride, op_type)
                self._ops.append(op)

        self._concat = concat

    def forward(self, s0, s1, arch):
        s0 = self.preprocess0.forward(s0)
        s1 = self.preprocess1.forward(s1)
        states = {0: s0, 1: s1}
        used_nodes = set()
        for op, f, t in arch:
            edge_id = int((t - 2) * (t + 1) / 2 + f)
            if t in states:
                states[t] = states[t] + self._ops[edge_id](states[f], op)
            else:
                states[t] = self._ops[edge_id](states[f], op)
            used_nodes.add(f)

        for i in range(2, self.n_nodes+2):
            if i in states:
                continue
            else:
                states[i] = torch.zeros_like(states[2])

        if self._concat is not None:
            return torch.cat([states[i] if i in self._concat else states[i]*0
                              for i in range(2, self.n_nodes+2)], dim=1)
        else:
            if self.loose_end:
                return torch.cat([states[i]*0 if i in used_nodes else states[i]*0
                                  for i in range(2, self.n_nodes+2)], dim=1)
            else:
                return torch.cat([states[i] for i in range(2, self.n_nodes+2)], dim=1)


class NASNetwork(nn.Module):
    def __init__(self, C, num_classes, layers,
                 n_nodes=4, multiplier=4, stem_multiplier=3,
                 loose_end=False,
                 normal_concat=None, reduced_concat=None, op_type=None):
        super(NASNetwork, self).__init__()
        self.C = C
        self.num_classes = num_classes
        self.layers = layers
        self.n_nodes = n_nodes
        self.multiplier = multiplier

        self.stem_multiplier = stem_multiplier
        self.op_type = op_type

        self.loose_end = loose_end

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        _concat = None
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
                if reduced_concat is not None:
                    _concat = reduced_concat
            else:
                reduction = False
                if normal_concat is not None:
                    _concat = normal_concat
            cell = NASCell(self.n_nodes, multiplier, C_prev_prev, C_prev, C_curr,
                           reduction, reduction_prev,
                           loose_end=loose_end, concat=_concat, op_type=self.op_type)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, inputs, normal_arch, reduced_arch):
        s0 = s1 = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                archs = reduced_arch
            else:
                archs = normal_arch
            s0, s1 = s1, cell(s0, s1, archs)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits
