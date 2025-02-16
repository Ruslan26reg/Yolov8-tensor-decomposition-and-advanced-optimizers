import math

import torch
from torch.optim.optimizer import Optimizer

from pytorch_optimizer.base_optimizer import BaseOptimizer
from pytorch_optimizer.types import BETAS, CLOSURE, DEFAULTS, LOSS, PARAMETERS


class DiffPNM(Optimizer, BaseOptimizer):

    def __init__(
        self,
        params: PARAMETERS,
        lr: float = 1e-2,
        betas: BETAS = (0.99, 0.999, 1.0),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        amsgrad: bool = True,
        eps: float = 1e-5,
    ):
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.weight_decouple = weight_decouple
        self.amsgrad = amsgrad
        self.eps = eps

        self.validate_parameters()

        defaults: DEFAULTS = dict(
            lr=lr, betas=betas, weight_decay=weight_decay, weight_decouple=weight_decouple, amsgrad=amsgrad, eps=eps
        )
        super().__init__(params, defaults)

    def validate_parameters(self):
        self.validate_learning_rate(self.lr)
        self.validate_betas(self.betas)
        self.validate_weight_decay(self.weight_decay)
        self.validate_epsilon(self.eps)

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['neg_exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['previous_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['previous_data'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if group['amsgrad']:
                    state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, closure: CLOSURE = None) -> LOSS:
        loss: LOSS = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                data = p.data
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('DiffPNM does not support sparse gradients')

                if group['weight_decouple']:
                    p.mul_(1.0 - group['lr'] * group['weight_decay'])
                else:
                    grad.add_(p, alpha=group['weight_decay'])

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['neg_exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['previous_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['previous_data'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['step'] += 1
                beta1, beta2, beta3 = group['betas']

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                previous_grad = state['previous_grad']
                preserve_data = state['previous_data']

                diff = abs(previous_grad - grad)
                dfc = 1.0 / (1.0 + torch.exp(-diff))
                diff_data = previous_data - data

                state['previous_grad'] = grad.clone()
                state['previous_data'] = data.clone()

                exp_avg_sq = state['exp_avg_sq']
                if state['step'] % 2 == 1:
                    exp_avg, neg_exp_avg = state['exp_avg'], state['neg_exp_avg']
                else:
                    exp_avg, neg_exp_avg = state['neg_exp_avg'], state['exp_avg']

                dg = grad
                if state['step'] > 1:
                    dg = (dg + diff_data * dg * dg)/2
                exp_avg.mul_(beta1 ** 2).add_(dg, alpha=1 - beta1 ** 2)
                noise_norm = math.sqrt((1 + beta3) ** 2 + beta3 ** 2)

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                if group['amsgrad']:
                    exp_avg_sq = torch.max(state['max_exp_avg_sq'], exp_avg_sq)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                pn_momentum = exp_avg.mul(1 + beta3).add(neg_exp_avg, alpha=-beta3).mul(1.0 / noise_norm)
                p.addcdiv_(pn_momentum, denom, value=-step_size)

        return loss
