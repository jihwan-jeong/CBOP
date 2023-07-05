"""
Contain some self-contained modules.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence

import rlkit.torch.pytorch_util as ptu
from rlkit.util.common import LOG_STD_MAX, LOG_STD_MIN
from rlkit.util.eval_util import create_stats_ordered_dict

from collections import OrderedDict


class HuberLoss(nn.Module):
    def __init__(self, delta=1):
        super().__init__()
        self.huber_loss_delta1 = nn.SmoothL1Loss()
        self.delta = delta

    def forward(self, x, x_hat):
        loss = self.huber_loss_delta1(x / self.delta, x_hat / self.delta)
        return loss * self.delta * self.delta


class LayerNorm(nn.Module):
    """
    Simple 1D LayerNorm.
    """

    def __init__(self, features, center=True, scale=False, eps=1e-6):
        super().__init__()
        self.center = center
        self.scale = scale
        self.eps = eps
        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(features))
        else:
            self.scale_param = None
        if self.center:
            self.center_param = nn.Parameter(torch.zeros(features))
        else:
            self.center_param = None

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = (x - mean) / (std + self.eps)
        if self.scale:
            output = output * self.scale_param
        if self.center:
            output = output + self.center_param
        return output


class GaussianMixtureCluster(nn.Module):
    """
    Clustering with the centroids updated by variational inference over the Gaussian mixture model.
    Code copied from https://github.com/LunjunZhang/world-model-as-a-graph.
    """
    def __init__(
            self,
            num_mixture_models: int,
            dim: int,
            learned_prior: bool = False,
            std_reg: float = 0.0,
            elbo_beta: float = 1.0,
            embed_epsilon: float = 0.1,
            embed_op: str = 'mean',
    ):
        super().__init__()

        self.num_mixture_models = num_mixture_models
        self.dim = dim
        self.comp_mean = nn.Parameter(
            torch.randn(self.num_mixture_models, self.dim) * np.sqrt(1.0 / self.num_mixture_models))
        self.comp_logstd = nn.Parameter(
            torch.randn(1, self.dim) * 1 / np.e, requires_grad=True
        )
        self.mixture_logit = nn.Parameter(
            torch.ones(self.num_mixture_models), requires_grad=learned_prior,
        )

        self.std_reg = std_reg
        self._initialized = False
        self.elbo_beta = elbo_beta
        self.embed_epsilon = embed_epsilon
        self.embed_op = embed_op

    @property
    def initialized(self):
        return self._initialized
    
    @initialized.setter
    def initialized(self, val):
        self._initialized = val

    def component_log_prob(self, x):
        """
        Args:

        Returns:

        """
        if x.ndim == 1:
            x = x.repeat(1, self.num_mixture_models, 1)
        elif x.ndim == 2:
            x = x.unsqueeze(1).repeat(1, self.num_mixture_models, 1)
        assert x.ndim == 3 and x.size(1) == self.num_mixture_models and x.size(2) == self.dim
        comp_logstd = torch.clamp(self.comp_logstd, LOG_STD_MIN, LOG_STD_MAX)
        comp_dist = Normal(self.comp_mean, torch.exp(comp_logstd))
        comp_log_prob = comp_dist.log_prob(x).sum(dim=-1)   # (nbatch, n_mix)
        return comp_log_prob

    def forward(self, x, with_elbo=True):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        assert x.ndim == 2 and x.size(1) == self.dim
        log_mix_probs = torch.log_softmax(self.mixture_logit, dim=-1).unsqueeze(0)  # (1, n_mix)
        assert log_mix_probs.size(0) == 1 and log_mix_probs.size(1) == self.num_mixture_models

        prior_prob = torch.softmax(self.mixture_logit, dim=0).unsqueeze(0)
        log_comp_probs = self.component_log_prob(x)     # (nbatch, n_mix)

        log_prob_x = torch.logsumexp(log_mix_probs + log_comp_probs, dim=-1, keepdim=True)   # (nbatch, 1)
        log_posterior = log_comp_probs + log_mix_probs - log_prob_x    # (nbatch, n_mix)
        posterior = torch.exp(log_posterior)

        if with_elbo:
            kl_from_prior = kl_divergence(Categorical(probs=posterior), Categorical(probs=prior_prob))
            return posterior, dict(
                comp_log_prob=log_comp_probs,
                log_data=(posterior * log_comp_probs).sum(dim=-1),
                kl_from_prior=kl_from_prior
            )
        else:
            return posterior

    def centroids(self):
        with torch.no_grad():
            return self.comp_mean.clone().detach()

    def circles(self):
        with torch.no_grad():
            return torch.exp(self.comp_logstd).clone().expand_as(self.comp_mean).detach()

    def std_mean(self):
        return torch.exp(self.comp_logstd).mean()

    def assign_centroids(self, x):
        self.comp_mean.data.copy_(x)

    def embed_loss(self, embedding):
        posterior, elbo = self(embedding, with_elbo=True)
        log_data = elbo['log_data']
        kl_from_prior = elbo['kl_from_prior']
        if ptu.has_nan(log_data) or ptu.has_nan(kl_from_prior):
            pass
        loss_elbo = - (log_data - self.elbo_beta * kl_from_prior).mean()
        std_mean = self.std_mean()
        loss_std = self.std_reg * std_mean
        loss_embed_total = loss_elbo + loss_std

        info = OrderedDict()
        info['ELBO Loss'] = np.mean(ptu.get_numpy(loss_elbo))
        info['Cluster Std Loss'] = np.mean(ptu.get_numpy(loss_std))
        info['Embed Total Loss'] = np.mean(ptu.get_numpy(loss_embed_total))
        info.update(create_stats_ordered_dict(
            'Cluster Log Data',
            ptu.get_numpy(log_data)
        ))
        info.update(create_stats_ordered_dict(
            'Cluster KL',
            ptu.get_numpy(kl_from_prior)
        ))
        info.update(create_stats_ordered_dict(
            'Cluster Post Std',
            ptu.get_numpy(posterior.std(dim=-1))
        ))
        info['Cluster Std Mean'] = ptu.get_numpy(std_mean)
        return loss_embed_total, info
