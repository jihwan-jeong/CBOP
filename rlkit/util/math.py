import torch.nn as nn
import torch

# inplace truncated normal function for pytorch.
# credit to https://github.com/Xingyu-Lin/mbpo_pytorch/blob/main/model.py#L64
def truncated_normal_(tensor: torch.Tensor, mean: float = 0, std: float = 1):
    """Samples from a truncated normal distribution in-place.

    Args:
        tensor (torch.Tensor): The tensor in which sampled values will be stored.
        mean (float): The desired mean (default = 0).
        std (float): The desired standard deviation (default = 1).

    Returns:
        (torch.Tensor): The tensor with the values sampled from the truncated normal distribution.
            Note that this modifies the input tensor in place, so this is just a pointer to the same object.
    """
    nn.init.normal_(tensor, mean=mean, std=std)
    while True:
        cond = torch.logical_or(tensor < mean - 2 * std, tensor > mean + 2 * std)
        if not torch.sum(cond):
            break
        tensor = torch.where(
            cond,
            nn.init.normal_(
                torch.ones(tensor.size(), device=tensor.device), mean=mean, std=std
            ),
            tensor,
        )
    return tensor

def truncated_linear(
    min_x: float, max_x: float, min_y: float, max_y: float, x: float
) -> float:
    """Truncated linear function.
    Implements the following function:
        f1(x) = min_y + (x - min_x) / (max_x - min_x) * (max_y - min_y)
        f(x) = min(max_y, max(min_y, f1(x)))
    If max_x - min_x < 1e-10, then it behaves as the constant f(x) = max_y
    """
    assert min_x <= max_x and min_y <= max_y
    if max_x - min_x < 1e-10:
        return max_y
    if x <= min_x:
        y: float = min_y
    else:
        dx = (x - min_x) / (max_x - min_x)
        dx = min(dx, 1.0)
        y = dx * (max_y - min_y) + min_y
    return y
