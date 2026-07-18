import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


def cosine_lambda(epoch, total_epochs, max_l=10.0, min_l=-10.0):

    cos = (1 + math.cos(math.pi * epoch / total_epochs)) / 2
    return min_l + (max_l - min_l) * cos


class Architect(object):

    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model

        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled, epoch,
             n_epochs):
        self.optimizer.zero_grad()
        if unrolled:
      
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
          
            self._backward_step(input_valid, target_valid, epoch, n_epochs)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid, epoch, n_epochs):
        """


        loss = self.model._loss(input_valid, target_valid)


        alphas = list(self.model.arch_parameters())
        grads = torch.autograd.grad(loss, alphas, create_graph=True, retain_graph=True)


        grads_data = [g.detach() for g in grads]

        total_kl = 0
        for g in grads:
            # g shape: [num_edges, num_ops]
            g_probs = F.softmax(torch.abs(g), dim=-1)
            num_ops = g.size(-1)

            g_entropy = -torch.sum(g_probs * torch.log(g_probs + 1e-9), dim=-1).mean()

            # KL(P || U) = log(N) - H(P)
            kl_div = math.log(num_ops) - g_entropy
            total_kl += kl_div

        avg_kl = total_kl / len(grads)

        lambda_k = cosine_lambda(epoch, n_epochs)


        reg_term = lambda_k * avg_kl
        reg_term.backward()


        del reg_term, grads, loss, avg_kl, g_probs


        with torch.no_grad():
            for v, g_d in zip(alphas, grads_data):
                if v.grad is None:
                    v.grad = g_d
                else:
                    v.grad += g_d


        del grads_data

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss = self.model._loss(input, target)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))
        return unrolled_model

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        loss = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]