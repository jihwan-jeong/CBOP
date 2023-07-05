import torch
import numpy as np
from rlkit.torch import pytorch_util as ptu

def compute_disagreement(
        next_obs,
        logvar,
        sampling_cfg,
        batch_size,
):
    # MOPO
    if sampling_cfg.type == "var":
        std = torch.sqrt(torch.exp(logvar))
        disagreement = torch.amax(torch.linalg.norm(std, dim=2), dim=0).unsqueeze(-1)

    # MOReL
    elif sampling_cfg.type == "mean":
        max_err = np.zeros((batch_size, 1))
        next_obs = ptu.get_numpy(next_obs)
        for i in range(next_obs.shape[0]):      # Iterate models in the ensemble
            pred_1 = next_obs[i, :]
            for j in range(i+1, next_obs.shape[0]):
                pred_2 = next_obs[j, :]
                error = np.linalg.norm((pred_1 - pred_2), axis=-1, keepdims=True)
                max_err = np.maximum(max_err, error)
        disagreement = max_err

    else:
        raise ValueError
    return disagreement
