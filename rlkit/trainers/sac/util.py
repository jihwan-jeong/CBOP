import torch

from rlkit.policies.gaussian_policy import TanhGaussianPolicy
import rlkit.torch.pytorch_util as ptu


def get_policy_actions(
        obs: torch.Tensor,
        num_actions: int,
        network: TanhGaussianPolicy,    # Policy class used by SAC and CQL
        detach: bool = True,
):
    obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
    dist = network(obs_temp)
    if detach:
        new_obs_actions, new_obs_log_pi = dist.sample_and_logprob()
    else:
        new_obs_actions, new_obs_log_pi = dist.rsample_and_logprob()
    
    return new_obs_actions, new_obs_log_pi.view(obs.shape[0], num_actions, 1)


def get_tensor_values(
        obs: torch.Tensor,
        actions: torch.Tensor,
        network: torch.nn.Module,
):
    action_shape = actions.size(0)
    obs_shape = obs.size(0)
    num_repeat = int(action_shape / obs_shape)
    obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.size(0) * num_repeat, obs.size(1))
    preds = network(obs_temp, actions)
    preds = preds.view(-1, obs.size(0), num_repeat, 1)
    return preds


def try_update_target_networks(train_step, target_update_period, tau, networks):
    if train_step % target_update_period == 0:
        update_target_networks(tau, networks)


def update_target_networks(tau, networks):
    for source, target in networks:
        ptu.soft_update_from_to(
            source, target, tau
        )
