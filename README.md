# Conservative Bayesian Model-Based Value Expansion for Offline Policy Optimization (CBOP)
This is the codebase for reproducing the results presented in the paper [Conservative Bayesian Model-Based Value Expansion for Offline Policy Optimization]() published at ICLR'23.

Note that the codebase originally derived from [rlkit](https://github.com/rail-berkeley/rlkit), but we have made many modifications, specifically the way we handle experiment configurations. We use [Hydra](https://github.com/facebookresearch/hydra) to organize experiments; while we heavily used [W&B](https://github.com/wandb/client) for hyperparameter sweeping and experiment tracking. However, ```wandb``` is optional and is not required to run the code. 

We used the locomotion benchmark datasets/environments provided in [D4RL](https://github.com/rail-berkeley/d4rl). So, you need to download and install, e.g., MuJoCo, gym, D4RL, etc. Follow the instructions below to set up the conda environment.

## Installation
Here, we give step-by-step instructions to set up the Conda environment with required libraries to run our code.

1. Setting up the Conda environment
```
# Create conda env and upgrade pip to up-to-date
conda create -n cbop python=3.8
conda activate cbop

pip install pip --upgrade
```
2. Install MuJoCo210, mujoco-py, gym, dm-control, d4rl and mjrl
```
# If you are using mac, use mujoco210-macos-x86_64.tar.gz instead
mkdir ~/.mujoco
cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xvzf mujoco210-linux-x86_64.tar.gz

# Move to the directory where you will clone D4RL and MJRL repos
# For example, `cd ~/Documents/repos`
# Then, clone the repos
git clone git@github.com:rail-berkeley/d4rl.git
git clone git@github.com:aravindr93/mjrl.git
```

Now, we need to modify a few lines of `d4rl/setup.py` since otherwise there will be errors. Specifically, comment out lines 16-18 of `d4rl/setup.py` (2 lines related to dm_control and one related to mjrl) and make sure to close the bracket. You should have something like below:
```python
setup(
    name='d4rl',
    version='1.1',
    install_requires=['gym',
                      'numpy',
                      'mujoco_py',
                      'pybullet',
                      'h5py',
                      'termcolor',  # adept_envs dependency
                      'click',  # adept_envs dependency
                      #'dm_control' if 'macOS' in platform() else
                      #'dm_control @ git+https://github.com/deepmind/dm_control@main#egg=dm_control',
                      #'mjrl @ git+https://github.com/aravindr93/mjrl@master#egg=mjrl'],
		               ],
   ...
)
```
Now, we install the aforementioned packages:
```
pip install mujoco-py
pip install gym==0.22.0
pip install dm_control==0.0.403778684
cd mjrl/
pip install -e .
cd ../d4rl/
pip install -e .
```
3. Install the current project as a package

Now, move back to the folder containing the CBOP project. Then, do
```
pip install -e .
```

4. Install other necessary packages:
```
pip install hydra-core --upgrade
pip install torch pygame omegaconf matplotlib gtimer opencv-python wandb 
```
Sometimes, the CUDA version may not be compatible with the torch version. In that case, uninstall torch and reinstall a proper version, following the instructions given in the torch website.

5. Finally, update the environment variable 
```
# Manage the environment PATH for MuJoCo210
echo export LD_LIBRARY_PATH='$LD_LIBRARY_PATH':~/.mujoco/mujoco210/bin >> ~/.bashrc

# If you have GPUs on your machine, make sure that LD_LIBRARY_PATH contains something like the following
echo export LD_LIBRARY_PATH='$LD_LIBRARY_PATH':/usr/lib/nvidia >> ~/.bashrc
```

6. Commonly seen errors
```
>> FileNotFoundError: [Errno 2] No such file or directory: 'patchelf'
Install patchelf with command:
$ sudo apt-get install patchelf

```

## Reproducing the results
1. Pretrain the dynamics ensemble
```
# Set log_to_wandb to true to turn wandb on
# Identify the d4rl environment with arguments "env" and "overrides.d4rl_config"
python -m rlkit.examples.main algorithm=pretrain env=hopper overrides.d4rl_config=medium-v2 is_offline=true log_to_wandb=false overrides.policy_eval.train=false overrides.bc_prior.train=false overrides.dynamics.train=true save_dir=data/pretrain/hopper/medium overrides.dynamics.batch_size=1024 overrides.dynamics.ensemble_model.ensemble_size=30 seed=0
```

2. Pretrain the policy and the Q ensemble with behavior clone (BC) and policy evaluation (PE) respectively
```
python -m rlkit.examples.main algorithm=pretrain env=hopper overrides.d4rl_config=medium-v2 is_offline=true log_to_wandb=false overrides.policy_eval.train=true overrides.bc_prior.train=true overrides.dynamics.train=false save_dir=data/pretrain/hopper/medium overrides.bc_prior.batch_size=1024 overrides.policy_eval.batch_size=1024 overrides.policy_eval.num_value_learning_repeat=5 algorithm.algorithm_cfg.num_total_epochs=100 seed=0
```

<!-- 3. Copy the pretrained checkpoints to the target directory
```
cp exp/pretrain/default/<env-name>/<date>/<training-starting-time>/offline_algo_checkpoint.pth data/pretrain/<env-name>/<d4rl-config-name>/
cp exp/pretrain/default/<env-name>/<date>/<training-starting-time>/.pth data/pretrain/<env-name>/<d4rl-config-name>/
``` -->

3. Reproduce CBOP results
```
python -m rlkit.examples.main env=hopper overrides.d4rl_config=medium-v2 algorithm=mvepo is_offline=true log_to_wandb=false overrides.trainer_cfg.horizon=10 overrides.trainer_cfg.num_qfs=50 algorithm.algorithm_cfg.num_epochs=1000 overrides.trainer_cfg.lcb_coeff=3. overrides.trainer_cfg.lr=3e-4 overrides.dynamics.ensemble_model.ensemble_size=30 overrides.dynamics.num_elites=20 cache_dir=data/pretrain/hopper/medium overrides.trainer_cfg.sampling_method=min overrides.offline_cfg.checkpoint_type=behavior overrides.trainer_cfg.indep_sampling=false snapshot_mode=gap_and_last algorithm.algorithm_cfg.save_snapshot_gap=30 overrides.trainer_cfg.eta=1
```
