import numpy as np
from typing import cast, Union, Optional
from rlkit.data_management.replay_buffers.env_replay_buffer import EnvReplayBuffer

def get_policy_training_batch(batch_size, real_data_ratio, env_buffer, model_buffer):
    """
    Mix the real experiences from the true environment as well as the fake experiences from the model environment
    for policy training.
    """
    # When ``batch_size==-1``, return all samples in the model buffer
    if batch_size == -1:
        data = model_buffer.get_transitions_dict()
        n = len(model_buffer)
        idx = np.random.permutation(np.arange(n))
        data = {
            k: val[idx] for k, val in data.items()
        }
        return data

    env_batch_size = int(batch_size * real_data_ratio)
    model_batch_size = batch_size - env_batch_size

    env_batch = env_buffer.random_batch(env_batch_size)
    if model_batch_size > 0 and len(model_buffer) > 0:
        model_batch = model_buffer.random_batch(model_batch_size)
        batch = {k: np.concatenate((env_batch[k], model_batch[k]), axis=0) for k in env_batch.keys()}
    else:
        batch = env_batch
    return batch


def maybe_replace_model_buffer(
    model_buffer: Optional[EnvReplayBuffer],
    env,
    new_capacity: int,
) -> EnvReplayBuffer:
    if model_buffer is None or new_capacity != model_buffer.max_replay_buffer_size():
        new_buffer = EnvReplayBuffer(
            max_replay_buffer_size=new_capacity,
            env=env,
        )
        if model_buffer is None:
            return new_buffer
        n = len(model_buffer)
        new_buffer.add_batch(
            dict(
                observations=model_buffer._observations[:n],
                actions=model_buffer._actions[:n],
                rewards=model_buffer._rewards[:n],
                next_observations=model_buffer._next_obs[:n],
                terminals=model_buffer._terminals[:n],
            )
        )
        return new_buffer
    return model_buffer

