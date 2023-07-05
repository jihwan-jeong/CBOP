from collections import OrderedDict
import copy
import numpy as np
import torch
import torch.optim as optim

import gtimer as gt
from rlkit.data_management.replay_buffers.replay_buffer import ReplayBuffer
from rlkit.torch.models.probabilistic_ensemble import ProbabilisticEnsemble
import rlkit.torch.pytorch_util as ptu
from rlkit.util.common import get_epoch_timings
import rlkit.types
from rlkit.core.logging.logging import logger
from rlkit.core.rl_algorithms.torch_rl_algorithm import TorchTrainer
from rlkit.torch.models.dynamics_models.model import DynamicsModel
from typing import Optional, Dict


class MBRLTrainer(TorchTrainer):
    def __init__(
            self,
            dynamics_model: DynamicsModel,
            num_elites: Optional[int] = None,
            learning_rate: float = 1e-3,
            batch_size: int = 256,
            optimizer_class=optim.Adam,
            train_call_freq: int = 1,
            normalize_inputs: bool = True,
            normalize_outputs: bool = False,
            obs_preproc: rlkit.types.ObsProcessFnType = None,
            targ_proc: rlkit.types.ObsProcessFnType = None,
            rng: Optional[np.random.Generator] = None,
            holdout_pct: Optional[float] = None,
            **kwargs,
    ):
        super().__init__()

        self.dynamics_model = dynamics_model
        self.ensemble_size = dynamics_model.ensemble_size
        self.learn_reward = dynamics_model.learn_reward
        self.learn_logstd_min_max = dynamics_model.learn_logstd_min_max
        self.num_elites = min(num_elites, self.ensemble_size) if num_elites \
                          else self.ensemble_size

        self.obs_dim = dynamics_model.obs_dim             # This could be different from ensemble input_size (see obs_preproc)
        self.action_dim = dynamics_model.action_dim
        self.batch_size = batch_size
        self.train_call_freq = train_call_freq
        self.obs_preproc = obs_preproc
        self.targ_proc = targ_proc

        self.optimizer = self.construct_optimizer(
            dynamics_model.model, optimizer_class, learning_rate)
        self.separate_reward_func = self.dynamics_model.separate_reward_func
        if self.separate_reward_func is not None:
            self.rew_optimizer = self.construct_optimizer(
                self.dynamics_model.separate_reward_func, optimizer_class, learning_rate
            )

        self._normalize_inputs = normalize_inputs
        self._normalize_outputs = normalize_outputs
        
        self.holdout_pct = holdout_pct
        
        self._n_train_steps_total = 0
        self.eval_statistics = OrderedDict()
        
        if rng is None:
            self._rng: np.random.Generator = np.random.default_rng()
        else:
            self._rng: np.random.Generator = rng
        self.max_logging = kwargs.get('max_logging', 1000)      # Max number of samples to use as test set during model learning
        
        model_config = self.dynamics_model.config
        self.name = f"dynamics_model_{model_config}" if len(model_config) > 0 else "dynamics_model"
        
        network_dict = dict(
            dynamics=self.dynamics_model,
            dynamics_optimizer=self.optimizer,
            dynamics_elite_models=self.dynamics_model.elite_models,
        )
        for key, module in network_dict.items():
            self.register_module(key, module)

    def construct_optimizer(self, model, optimizer_class, lr):
        return optimizer_class(model.parameters(), lr)

    def train_from_buffer(
            self,
            replay_buffer,
            holdout_pct=0.2,
            max_grad_steps=None,
            max_epochs_since_last_update=5,
            num_total_epochs=None,
            improvement_threshold: float = 0.01,
            use_best_parameters: bool = False,
            max_logging: int = None,
            **kwargs
    ):
        """
        Train the dynamics model with the data in the buffer.

        Args:
            replay_buffer                The replay buffer to sample batches of data from
            holdout_pct                  The portion of data to be used for validation
            max_grad_steps               The maximum gradient steps to take
            max_epochs_since_last_update
            improvement_threshold
        """
        # Reset elite models
        self.dynamics_model.reset_elite_models()

        self._n_train_steps_total += 1

        # Keep track of the best model parameters obtained from this round of training epochs
        best_weights: Optional[Dict] = None
        eval_score: Optional[torch.Tensor] = None
        training_losses, val_scores = [], []

        # Skip training the model periodically (but, train_call_freq is default at 1)
        if self._n_train_steps_total % self.train_call_freq > 0 and self._n_train_steps_total > 1:
            return

        # Get the entire transition data from the buffer (different format for ordinary and goal-conditioned buffers)
        data = self.get_transitions(replay_buffer, holdout_pct=holdout_pct, max_logging=max_logging)

        x_train, y_train, x_test, y_test = tuple(
            map(lambda x: data[x], ['x_train', 'y_train', 'x_test', 'y_test'])
        )
        n_train = x_train.shape[0]

        # Standardize network inputs / outputs
        assert hasattr(self.dynamics_model.model, 'set_output_transforms') and \
               callable(getattr(self.dynamics_model.model, 'set_output_transforms'))
        self.dynamics_model.model.set_output_transforms(False)  # Do not use output transformation (data is normalized)
        if self._normalize_inputs and not self.dynamics_model.trained:
            self.dynamics_model.fit_input_stats(x_train)

        if self._normalize_outputs:
            if self.separate_reward_func:
                if self.dynamics_model.trained:
                    y_r_mean, y_r_std = ptu.get_numpy(self.dynamics_model.separate_reward_func.output_mu.data), \
                                        ptu.get_numpy(self.dynamics_model.separate_reward_func.output_std.data)
                    y_s_mean, y_s_std = ptu.get_numpy(self.dynamics_model.model.output_mu.data), \
                                        ptu.get_numpy(self.dynamics_model.model.output_std.data)
                else:
                    self.separate_reward_func.set_output_transforms(False)
                    y_r_mean, y_r_std = self.dynamics_model.separate_reward_func.fit_output_stats(y_train[:, :1])
                    y_s_mean, y_s_std = self.dynamics_model.fit_output_stats(y_train[:, 1:])
                y_mean, y_std = np.concatenate((y_r_mean, y_s_mean), axis=-1), \
                                np.concatenate((y_r_std, y_s_std), axis=-1)
            elif self.dynamics_model.trained:
                y_mean, y_std = ptu.get_numpy(self.dynamics_model.model.output_mu.data), \
                                ptu.get_numpy(self.dynamics_model.model.output_std.data)
            else:
                y_mean, y_std = self.dynamics_model.fit_output_stats(y_train)
            y_train = (y_train - y_mean) / (y_std + 1e-8)
            y_test = (y_test - y_mean) / (y_std + 1e-8)

        # Initial evaluation
        best_val_score = self.evaluate(self.dynamics_model, x_test, y_test)

        # Skip training if already trained (set elite models if needed)
        if self.dynamics_model.trained:
            logger.log("Dynamics model already trained! Setting the elite models and skipping training..", with_timestamp=True)
            self._set_elite(self.dynamics_model, best_val_score)
            self.eval_statistics['Final Holdout Loss'] = ptu.get_numpy(best_val_score.mean())
            for i in range(self.ensemble_size):
                name = 'M%d' % (i+1)
                self.eval_statistics[name + 'Final Loss'] = \
                    np.mean(ptu.get_numpy(best_val_score[i]))
            return

        # train until holdout set convergence
        num_epochs, num_steps = 0, 0
        num_epochs_since_last_update = 0
        self.prev_params = copy.deepcopy(self.dynamics_model.state_dict())
        best_holdout_loss = float('inf')

        while num_epochs_since_last_update < max_epochs_since_last_update and \
            (not max_grad_steps or num_steps < max_grad_steps):
            if num_total_epochs and num_epochs == num_total_epochs:
                break
            
            # indices for random sampling;
            idxs = self._rng.integers(x_train.shape[0], size=[self.ensemble_size, x_train.shape[0]])
            num_batches = int(np.ceil(n_train / self.batch_size))

            self.dynamics_model.train()
            batch_losses = []
            for b in range(num_batches):
                # generate idx for each model to bootstrap
                b_idxs = idxs[:, b*self.batch_size: (b+1)*self.batch_size]
                x_batch, y_batch = x_train[b_idxs], y_train[b_idxs]         # (ensemble_size, batch_size, input_dim)
                
                # Transform to torch tensors
                x_batch, y_batch = ptu.from_numpy(x_batch), ptu.from_numpy(y_batch)

                # Compute the loss and take the gradient descent step
                self.optimizer.zero_grad()
                if self.separate_reward_func is None:
                    loss, _ = self.dynamics_model.get_loss(x_batch, y_batch, inc_var_loss=True)
                    loss.backward()
                    self.optimizer.step()
                else:
                    # When a separate model is used for learning reward function
                    self.rew_optimizer.zero_grad()
                    (rew_loss, _), (trans_loss, _) = self.dynamics_model.get_loss(x_batch, y_batch, inc_var_loss=True)
                    
                    trans_loss.backward()
                    rew_loss.backward()
                    self.optimizer.step()
                    self.rew_optimizer.step()
                    loss = (trans_loss * self.obs_dim + rew_loss) / (self.obs_dim + 1)  # Aggregate reward and transition losses

                # Keep track of the training loss over time
                batch_losses.append(loss.detach().cpu())

            gt.stamp('training', unique=False)
            
            avg_batch_loss = np.mean(batch_losses).mean().item()
            training_losses.append(avg_batch_loss)

            num_steps += num_batches

            # Check if the validation score has improved
            eval_score = self.evaluate(self.dynamics_model, x_test, y_test)
            val_scores.append(eval_score.mean().item())
            maybe_best_weights, best_val_score = self.maybe_get_best_weights(
                self.dynamics_model, best_val_score, eval_score, self.prev_params, improvement_threshold
            )

            # If there was an improvement, save the model parameters; otherwise, repeat the epochs
            if maybe_best_weights:
                best_weights = maybe_best_weights
                num_epochs_since_last_update = 0        # As in MBPO
            else:
                num_epochs_since_last_update += 1

            self.eval_statistics['Training Losses'] = avg_batch_loss
            self.eval_statistics['Training Epochs'] = num_epochs
            self.eval_statistics['Training Steps'] = num_steps
            self.eval_statistics['Average Holdout Loss'] = ptu.get_numpy(eval_score.mean())

            for i in range(self.ensemble_size):
                name = 'M%d' % (i+1)
                self.eval_statistics[name + ' Loss'] = \
                    np.mean(ptu.get_numpy(eval_score[i]))
            gt.stamp('evaluation', unique=False)
            
            if not (num_epochs_since_last_update == max_epochs_since_last_update or
                        (max_grad_steps and num_steps == max_grad_steps) or
                        (num_total_epochs is not None and num_epochs == num_total_epochs - 1)
            ):  
                self.end_epoch(num_epochs)
            num_epochs += 1

        # Saving the best models
        if use_best_parameters:
            self._set_maybe_best_weights_and_elite(self.dynamics_model, best_weights, best_val_score)
            self.register_module('dynamics_elite_models', self.dynamics_model.elite_models)
            eval_score = self.evaluate(self.dynamics_model, x_test, y_test)
            self.eval_statistics['Final Holdout Loss'] = ptu.get_numpy(eval_score.mean())
            for i in range(self.ensemble_size):
                name = 'M%d' % (i+1)
                self.eval_statistics[name + 'Final Loss'] = \
                    np.mean(ptu.get_numpy(eval_score[i]))
        self.end_epoch(num_epochs - 1)
    
    def _set_maybe_best_weights_and_elite(
            self,
            model,
            best_weights: Optional[dict],
            best_val_score: torch.Tensor
    ):
        self._set_maybe_best_weights(model, best_weights, best_val_score)
        self._set_elite(model, best_val_score)

    def _set_elite(
            self,
            model,
            best_val_score
    ):
        if len(best_val_score) > 1 and hasattr(model, "num_elites"):
            sorted_indices = np.argsort(best_val_score.tolist())
            elite_models = sorted_indices[:model.num_elites]
            model.set_elite(elite_models)

    def get_transitions(self, buffer, holdout_pct: float = 0.2, max_logging: int = None):
        if isinstance(buffer, ReplayBuffer):
            data = buffer.get_transitions()
            obs_, act = data[:, :self.obs_dim], data[:, self.obs_dim: self.obs_dim+self.action_dim]
            rew = data[:, self.obs_dim+self.action_dim].reshape(-1, 1)
            next_obs_ = data[:, -self.obs_dim:]
            if self.obs_preproc is not None:
                obs = self.obs_preproc(obs_)
            else:
                obs = obs_
            if self.targ_proc is not None:
                target = self.targ_proc(obs_, next_obs_)   # targ_proc defaults to getting the state differences
            else:
                target = next_obs_

            # Prepare input data and corresponding targets; when reward is learned (reward, delta_state) is the target
            x = np.concatenate((obs, act), axis=-1)  # inputs x = (s, a)
            y = target if not self.learn_reward else np.concatenate((rew, target), axis=-1)

        else:
            raise NotImplementedError
        
        # Shuffle the dataset and generate the holdout set
        perm = self._rng.permutation(x.shape[0])
        x, y = x[perm], y[perm]

        n_test = min(int(x.shape[0] * holdout_pct), self.max_logging if max_logging is None else max_logging)
        x_train, x_test = x[n_test:], x[:n_test]
        y_train, y_test = y[n_test:], y[:n_test]

        return dict(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )            

    def train_from_torch(self, batch, idx=None):
        raise NotImplementedError

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._log_stats(epoch, prefix='Model Learning/')
        snapshot = self.get_snapshot()
        logger.save_itr_params(epoch, snapshot, prefix='model')

    def load(self, state_dict, prefix=''):
        
        for name, model in self.network_dict.items():
            name = f"{prefix}/{name}" if prefix != '' else name
            if name in state_dict:
                try:
                    model.load_state_dict(state_dict[name])
                    if hasattr(model, 'trained'):
                        model.trained = True
                except RuntimeError:
                    print(f"Failed to load state_dict[{name}]")
        
        if 'dynamics_elite_models' in state_dict:
            self.dynamics_model.set_elite(state_dict['dynamics_elite_models'])

    def configure_logging(self, **kwargs):
        import wandb
        wandb.watch(self.dynamics_model, **kwargs)
