from typing import Callable, Optional, Tuple, Union, Dict

import numpy as np
import torch

RewardFuncType = Union[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
                       Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]]
TermFuncType = Union[Callable[[torch.Tensor], torch.Tensor],
                     Callable[[np.ndarray, np.ndarray], np.ndarray]]
ObsProcessFnType = Callable[[np.ndarray], np.ndarray]
TensorType = Union[torch.Tensor, np.ndarray]
TrajectoryEvalFnType = Callable[[TensorType, torch.Tensor, Optional[TensorType],
                                 Optional[TensorType], Optional[int], Optional[Dict]],
                                torch.Tensor]
