from collections import OrderedDict
from copy import deepcopy
import gtimer as gt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from typing import Optional, Dict
from rlkit.core.logging.logging import logger
import rlkit.torch.pytorch_util as ptu
from rlkit.core.rl_algorithms.torch_rl_algorithm import TorchTrainer
from rlkit.util.eval_util import create_stats_ordered_dict
import rlkit.torch.models.ensemble


class PolEvalTrainer(TorchTrainer):
    """
    Policy evaluation trainer.
    If two Q functions are given, use the minimum of results of the two Q functions when computing target values.
    If not, a single Q function will be used.

    The evaluation of the policy is based on Fitted Q-Evaluation (FQE) [Le et al., 2019].
    That is, a datset of {(s, a), y} input-output pairs is firstly generated. Then, we use the standard mean squared
    Bellman error as the training loss to train the value network(s).
    """
    def __init__(
            self,
            value_func,
            use_bootstrap=True,                     # Use the value function to bootstrap the return at horizon H
            batch_size=32,
            discount=0.99,                             # Discount factor
            learning_rate=1e-3,
            polyak=1e-2,                            # Parameter for soft target network update
            obs_preproc=None,                          # Pre(post)-processing of observation data
            optimizer=optim.Adam,
            normalize_inputs: bool = False,
            normalize_outputs: bool = False,
            sampling_method: str = 'mean',
            rng: Optional[np.random.Generator] = None,
            **kwargs
    ):
        super().__init__()

        self.value_func = value_func
        self.is_qf = self.value_func.is_qf if hasattr(self.value_func, 'is_qf') else True
        self.ensemble_size = value_func.ensemble_size
        self.value_func_criterion = nn.MSELoss()
        self.value_func_optimizer = optimizer(
            self.value_func.parameters(),
            lr=learning_rate,
        )

        self._use_bootstrap = use_bootstrap

        self.batch_size = batch_size
        self.discount = discount
        self.polyak = polyak

        self.obs_preproc = obs_preproc
        self._normalize_inputs = normalize_inputs
        self._normalize_outputs = normalize_outputs

        self.max_logging = kwargs.get('max_logging', int(1e3))
        self._n_steps_into_future = kwargs.get('num_steps_into_future', 1)
        self.sampling_method = sampling_method

        self.kwargs = kwargs
        self.eval_statistics = OrderedDict()

        if rng is None:
            self._rng: np.random.Generator = np.random.default_rng()
        else:
            self._rng: np.random.Generator = rng

        network_dict = dict(
            value_func=self.value_func,
        )
        for key, module in network_dict.items():
            self.register_module(key, module)        

        self.name = 'policy_eval'

    def train_from_buffer(
            self,
            replay_buffer,
            holdout_pct=0.2,
            max_grad_steps=None,
            max_epochs_since_last_update=5,
            num_total_epochs=100,
            num_repeat=None,
            improvement_threshold=0.01,
            use_best_parameters=False,
            *args,
            **kwargs
    ):
        """
        Retrieve the transition samples from replay_buffer and evaluate the policy that generated the dataset.

        Args:
            replay_buffer           (SimpleReplayBuffer) The replay buffer to sample data from
            holdout_pct             (float) The fraction of data to be used as the held-out validation set
            max_grad_steps          (int)
            max_epochs_since_last_update (int): Max number of epochs since the last epoch with improvement
            num_total_epochs        (int) The number of training epochs
            num_repeat              (int) How many times target data should be recomputed
            improvement_threshold (float)
            use_best_parameters (bool)
        """
        if self.value_func.trained:
            logger.log("Value function already trained! Skipping training..", with_timestamp=True)
            return

        if num_repeat is None:
            num_repeat = self.kwargs.get('num_value_learning_repeat', 1)
        
        # Set up the value function used for constructing the dataset
        value_func = self.value_func
        if not self._use_bootstrap:
            vf, qf = None, None
            self.discount = 1              # No discounting if not bootstrapping
            num_repeat = 1              # Since not bootstrapping, no need to reconstruct dataset repeatedly
        elif self.is_qf:
            qf, vf = value_func, None
        else:
            qf, vf = None, value_func

        # For input normalization
        if self._normalize_inputs:
            obs = replay_buffer._observations[:replay_buffer._size]
            act = replay_buffer._actions[:replay_buffer._size]
            inputs = np.concatenate((obs, act), axis=-1) if self.is_qf else obs
            self.value_func.fit_input_stats(inputs)
        
        # Repeat value function learning with labels being regenerated within each epoch
        for r in gt.timed_for(range(num_repeat)):

            # Keep track of the best model parameters obtained from this round of training epochs
            best_weights: Optional[Dict] = None
            eval_score: Optional[torch.Tensor] = None
            training_losses, val_scores = [], []

            # Construct the dataset using the current target network
            obs, act, targets = self.process_samples(replay_buffer, vf, qf)
            gt.stamp('N-step target data constructed', unique=False)

            # Preprocess observations if needed
            if self.obs_preproc is not None:
                obs = self.obs_preproc(obs)

            # Prepare input data
            x = np.concatenate((obs, act), axis=-1) if self.is_qf else obs
            y = targets.reshape(-1, 1)

            # Generate the holdout set
            perm = self._rng.permutation(x.shape[0])        # Shuffle the entire data
            x, y = x[perm], y[perm]

            n_test = min(int(x.shape[0] * holdout_pct), self.max_logging)
            x_train, x_test = x[n_test:], x[:n_test]
            y_train, y_test = y[n_test:], y[:n_test]
            n_train = x.shape[0] - n_test

            # Standardize network inputs / outputs
            assert hasattr(self.value_func, 'set_output_transforms') and callable(
                getattr(self.value_func, 'set_output_transforms'))
            self.value_func.set_output_transforms(False)
            if self._normalize_outputs:
                y_mean, y_std = self.value_func.fit_output_stats(y_train)
                y_train = (y_train - y_mean) / (y_std + 1e-8)
                y_test = (y_test - y_mean) / (y_std + 1e-8)

            # Initial evaluation
            self.value_func.eval()
            best_val_score = self.evaluate(self.value_func, x_test, y_test)

            num_epochs, num_steps = 0, 0
            num_epochs_since_last_update = 0
            self.prev_params = deepcopy(self.value_func.state_dict())
            best_holdout_loss = float('inf')

            # indices for random sampling;
            if isinstance(self.value_func, rlkit.torch.models.ensemble.Ensemble):
                shape = (self.value_func.ensemble_size, x_train.shape[0])
            else:
                shape = (x_train.shape[0],)

            while num_epochs_since_last_update < max_epochs_since_last_update and \
                (not max_grad_steps or num_steps < max_grad_steps):
                if num_total_epochs and num_epochs == num_total_epochs:
                    break
                idxs = self._rng.integers(x_train.shape[0], size=shape)
                num_batches = int(np.ceil(n_train / self.batch_size))
                self.start_epoch(num_epochs)

                # Generate idx for each model to bootstrap
                self.value_func.train()
                batch_losses = []
                y_pred_total = np.zeros_like(y_train)
                for b in range(num_batches):
                    if isinstance(self.value_func, rlkit.torch.models.ensemble.Ensemble):
                        b_idxs = idxs[:, b * self.batch_size: (b + 1) * self.batch_size]
                        x_batch, y_batch = x_train[b_idxs], y_train[b_idxs]  # (ensemble_size, batch_size, input_dim)
                    else:
                        b_idxs = idxs[b * self.batch_size: (b + 1) * self.batch_size]
                        x_batch, y_batch = x_train[b_idxs], y_train[b_idxs]  # (batch_size, input_dim)
                    x_batch, y_batch = ptu.from_numpy(x_batch), ptu.from_numpy(y_batch)

                    # Compute the MSE loss
                    y_pred = self.value_func(x_batch, use_propagation=False)
                    if isinstance(y_pred, tuple):
                        y_pred = y_pred[0]
                    loss = self.value_func_criterion(y_pred, y_batch)

                    self.value_func_optimizer.zero_grad()
                    loss.backward()
                    self.value_func_optimizer.step()

                    batch_losses.append(loss.detach().cpu())
                    y_pred_total[b * self.batch_size: (b + 1) * self.batch_size] = ptu.get_numpy(y_pred).mean(axis=0)

                    self._num_train_total_steps += 1
                    self._num_timesteps_this_epoch += 1

                avg_batch_loss = np.mean(batch_losses).mean().item()
                training_losses.append(avg_batch_loss)
                num_steps += num_batches

                # Check if the validation score on average has improved
                self.value_func.eval()
                eval_score = self.evaluate(self.value_func, x_test, y_test)
                val_scores.append(eval_score.mean().item())
                maybe_best_weights, best_val_score = self.maybe_get_best_weights(
                    self.value_func, best_val_score, eval_score, self.prev_params, improvement_threshold
                )

                # If there was an improvement, save the model parameters; otherwise, repeat the epochs
                if maybe_best_weights:
                    best_weights = maybe_best_weights
                    num_epochs_since_last_update = 0
                else:
                    num_epochs_since_last_update += 1

                """
                Save some statistics per epoch
                """
                self.eval_statistics['Training Loss'] = avg_batch_loss
                self.eval_statistics['Holdout Loss'] = ptu.get_numpy(eval_score.mean())
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Value Predictions',
                    y_pred_total * (ptu.get_numpy(value_func.output_std) + 1e-8) + ptu.get_numpy(value_func.output_mu),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Value Targets',
                    y_train * (ptu.get_numpy(value_func.output_std) + 1e-8) + ptu.get_numpy(value_func.output_mu),
                ))
                self.eval_statistics['Repeat'] = r

                if not (num_epochs_since_last_update == max_epochs_since_last_update or
                            (max_grad_steps and num_steps == max_grad_steps) or
                            (num_total_epochs is not None and num_epochs == num_total_epochs - 1)
                ):
                    self.end_epoch(num_epochs)
                num_epochs += 1

            # Load the best model parameters per `repeat`
            if use_best_parameters:
                self._set_maybe_best_weights(self.value_func, best_weights, best_val_score)
            self.end_epoch(num_epochs - 1)

    def train_from_torch(self, batch, idx=None):
        raise NotImplementedError

    def process_samples(self, replay_buffer, vf, qf):
        dataset = replay_buffer.construct_nstep_input_target_dataset(
                                self._n_steps_into_future,
                                vf=vf,
                                qf=qf,
                                use_bootstrap=self._use_bootstrap,
                                discount=self.discount,
                                obs_preproc=self.obs_preproc,
                                sampling_method=self.sampling_method,
        )
        obs = dataset['obs'].copy()
        act = dataset['actions'].copy()
        targets = dataset['targets'].copy()
        del dataset
        return obs, act, targets
    
    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._log_stats(epoch, prefix='policy eval/')
        snapshot = self.get_snapshot()
        logger.save_itr_params(epoch, snapshot, prefix='policy eval')

    def configure_logging(self, **kwargs):
        import wandb
        wandb.watch(self.value_func, **kwargs)

    def load(self, state_dict, prefix=''):
        name = 'value_func'
        model = self.value_func
        
        name = f"{prefix}/{name}" if prefix != '' else name
        if name in state_dict:
            try:
                model.load_state_dict(state_dict[name])
                if hasattr(model, 'trained'):
                    model.trained = True
            except RuntimeError:
                print(f"Failed to load state_dict[{name}]")
