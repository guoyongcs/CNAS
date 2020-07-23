import torch
import torch.nn as nn
import torch.nn.functional as F

import functools

import torch.distributions

import cnas.model.genotypes
from cnas.model.genotypes import Genotype
from torch.distributions.utils import probs_to_logits


class ArchMaster(nn.Module):
    def __init__(self, n_nodes, cur_nodes=None, n_ops=None, cur_ops=None,
                 device="cpu", hidden_size=None,
                 temperature=None, tanh_constant=None,
                 op_tanh_reduce=None, force_uniform=False,):
        super(ArchMaster, self).__init__()

        self.n_ops = n_ops
        self.n_nodes = n_nodes
        self.device = device

        self.cur_nodes = n_nodes if cur_nodes is None else cur_nodes
        self.cur_ops = n_ops if cur_ops is None else cur_ops

        self.hidden_size = hidden_size
        self.attention_hidden_size = hidden_size
        self.temperature = temperature
        self.tanh_constant = tanh_constant
        self.op_tanh_reduce = op_tanh_reduce
        self.force_uniform = force_uniform

        # Embedding of (n_nodes+1) nodes
        # Note that the (n_nodes+2)-th node will not be used
        self.node_op_embedding = nn.Embedding(
            n_nodes + 1 + n_ops, self.hidden_size)

        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)

        self.emb_attn = nn.Linear(
            self.hidden_size, self.attention_hidden_size, bias=False)
        self.hid_attn = nn.Linear(
            self.hidden_size, self.attention_hidden_size, bias=False)
        self.v_attn = nn.Linear(self.hidden_size, 1, bias=False)

        self.w_soft = nn.Linear(self.hidden_size, n_nodes)

        self.batch_size = 1
        self.n_prev_nodes = 2

        self.reset_parameters()

    def inc_n_ops(self):
        '''
        progressive growing search space
        '''
        self.cur_ops += 1

    def inc_n_nodes(self):
        self.cur_nodes += 1

    @functools.lru_cache(maxsize=128)
    def _zeros(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size),
                           device=self.device, requires_grad=False)

    def reset_parameters(self, init_range=0.1):
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)

    def _scale_attention(self, logits, temperature, tanh_constant, constant_reduce=None):
        if temperature is not None:
            logits /= temperature
        if tanh_constant is not None:
            if constant_reduce is not None:
                tanh_constant /= constant_reduce
            logits = tanh_constant * torch.tanh(logits)
        return logits

    def _impl(self, probs):
        m = torch.distributions.Categorical(probs=probs)
        action = m.sample().view(-1)
        select_log_p = m.log_prob(action)
        entropy = m.entropy()
        return action, select_log_p, entropy

    def forward(self, logger=None):
        node_log_ps = []
        op_log_ps = []
        node_entropys = []
        op_entropys = []
        nodes = []
        ops = []

        inputs = self._zeros(self.batch_size)
        hidden = self._zeros(self.batch_size), self._zeros(self.batch_size)
        embed = None
        for node_idx in range(self.cur_nodes):
            for select in (("node", "op")):
                for i in range(self.n_prev_nodes):
                    embed = inputs if embed is None else self.node_op_embedding(inputs)
                    if self.force_uniform:
                        size = node_idx+2 if select == "node" else self.cur_ops
                        probs = torch.full((size,), 1/size, device=self.device)
                    else:
                        hx, cx = self.lstm(embed, hidden)
                        hidden = (hx, cx)
                        if select == "node":
                            # (node_idx+2, hidden_size)
                            query = self.node_op_embedding.weight[:node_idx+2, :]
                            # (node_idx+2, attention_hidden_size)
                            query = torch.tanh(self.emb_attn(query) + self.hid_attn(hx))
                            logits = self.v_attn(query).view(-1)  # (node_idx+2,)
                            logits = self._scale_attention(logits, self.temperature, self.tanh_constant)
                        else:
                            logits = self.w_soft(hx).view(-1)
                            logits = logits[:self.cur_ops]
                            logits = self._scale_attention(logits, self.temperature,
                                                           self.tanh_constant,
                                                           self.op_tanh_reduce)
                        probs = F.softmax(logits, dim=-1)

                    action, select_log_p, entropy = self._impl(probs)
                    if select == "node":
                        node_log_ps.append(select_log_p)
                        node_entropys.append(entropy)
                        inputs = action
                        nodes.append(action)
                    else:
                        op_log_ps.append(select_log_p)
                        op_entropys.append(entropy)
                        inputs = action + (self.n_nodes + 1)
                        ops.append(action)

        ordinal_arch = Genotype.lstm_output_to_ordinal(self.cur_nodes,
                                                       torch.cat(nodes).tolist(),
                                                       torch.cat(ops).tolist())
        return ordinal_arch, sum(node_log_ps)+sum(op_log_ps), sum(node_entropys), sum(op_entropys)


class MasterPairs(nn.Module):
    def __init__(self, normal_arch_master, reduce_arch_master):
        super(MasterPairs, self).__init__()
        self.normal_arch_master = normal_arch_master
        self.reduce_arch_master = reduce_arch_master

    def forward(self, logger=None):
        return self.normal_arch_master(logger), self.reduce_arch_master(logger)

    def set_force_uniform(self, force_uniform=True):
        self.normal_arch_master.force_uniform = force_uniform
        self.reduce_arch_master.force_uniform = force_uniform

    def unset_force_uniform(self):
        self.set_force_uniform(force_uniform=False)

    def inc_n_ops(self):
        self.normal_arch_master.inc_n_ops()
        self.reduce_arch_master.inc_n_ops()

    def inc_n_nodes(self):
        self.normal_arch_master.inc_n_nodes()
        self.reduce_arch_master.inc_n_nodes()
