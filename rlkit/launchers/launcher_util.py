import os
import os.path as osp
import pickle
import random
import sys
import time

import __main__ as main
import numpy as np

from rlkit.core.logging.logging import logger
from rlkit.core.logging.logging_setup import setup_logger
from rlkit.launchers import conf
from rlkit.torch.pytorch_util import set_gpu_mode
import rlkit.util.pythonplusplus as ppp



def recursive_items(dictionary):
    """
    Get all (key, item) recursively in a potentially recursive dictionary.
    Usage:

    ```
    x = {
        'foo' : {
            'bar' : 5
        }
    }
    recursive_items(x)
    # output:
    # ('foo', {'bar' : 5})
    # ('bar', 5)
    ```
    :param dictionary:
    :return:
    """
    for key, value in dictionary.items():
        yield key, value
        if type(value) is dict:
            yield from recursive_items(value)


def save_experiment_data(dictionary, log_dir):
    with open(log_dir + '/experiment.pkl', 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_experiment_here(
        experiment_function,
        variant=None,
        exp_id=0,
        seed=None,
        use_gpu=True,
        # Logger params:
        exp_prefix="default",
        snapshot_mode='last',
        snapshot_gap=1,
        git_infos=None,
        script_name=None,
        base_log_dir=None,
        force_randomize_seed=False,
        log_dir=None,
        **setup_logger_kwargs
):
    """
    Run an experiment locally without any serialization.

    :param experiment_function: Function. `variant` will be passed in as its
    only argument.
    :param exp_prefix: Experiment prefix for the save file.
    :param variant: Dictionary passed in to `experiment_function`.
    :param exp_id: Experiment ID. Should be unique across all
    experiments. Note that one experiment may correspond to multiple seeds,.
    :param seed: Seed used for this experiment.
    :param use_gpu: Run with GPU. By default False.
    :param script_name: Name of the running script
    :param log_dir: If set, set the log directory to this. Otherwise,
    the directory will be auto-generated based on the exp_prefix.
    :return:
    """
    if variant is None:
        variant = {}
    variant['exp_id'] = str(exp_id)

    if force_randomize_seed or seed is None:
        seed = random.randint(0, 100000)
        variant['seed'] = str(seed)
    reset_execution_environment()

    actual_log_dir = setup_logger(
        exp_prefix=exp_prefix,
        variant=variant,
        exp_id=exp_id,
        seed=seed,
        snapshot_mode=snapshot_mode,
        snapshot_gap=snapshot_gap,
        base_log_dir=base_log_dir,
        log_dir=log_dir,
        git_infos=git_infos,
        script_name=script_name,
        **setup_logger_kwargs
    )

    set_seed(seed)
    set_gpu_mode(use_gpu)

    run_experiment_here_kwargs = dict(
        variant=variant,
        exp_id=exp_id,
        seed=seed,
        use_gpu=use_gpu,
        exp_prefix=exp_prefix,
        snapshot_mode=snapshot_mode,
        snapshot_gap=snapshot_gap,
        git_infos=git_infos,
        script_name=script_name,
        base_log_dir=base_log_dir,
        **setup_logger_kwargs
    )
    save_experiment_data(
        dict(
            run_experiment_here_kwargs=run_experiment_here_kwargs
        ),
        actual_log_dir
    )
    return experiment_function(variant)

def set_seed(seed):
    """
    Set the seed for all the possible random number generators.

    :param seed:
    :return: None
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)


def reset_execution_environment():
    """
    Call this between calls to separate experiments.
    :return:
    """
    logger.reset()


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

"""
Below is doodad-specific code
"""
ec2_okayed = False
gpu_ec2_okayed = False
first_sss_launch = True

try:
    import doodad.mount as mount
    from doodad.utils import REPO_DIR
    CODE_MOUNTS = [
        mount.MountLocal(local_dir=REPO_DIR, pythonpath=True),
    ]
    for code_dir in conf.CODE_DIRS_TO_MOUNT:
        CODE_MOUNTS.append(mount.MountLocal(local_dir=code_dir, pythonpath=True))

    NON_CODE_MOUNTS = []
    for non_code_mapping in conf.DIR_AND_MOUNT_POINT_MAPPINGS:
        NON_CODE_MOUNTS.append(mount.MountLocal(**non_code_mapping))

    SSS_CODE_MOUNTS = []
    SSS_NON_CODE_MOUNTS = []
    if hasattr(conf, 'SSS_DIR_AND_MOUNT_POINT_MAPPINGS'):
        for non_code_mapping in conf.SSS_DIR_AND_MOUNT_POINT_MAPPINGS:
            SSS_NON_CODE_MOUNTS.append(mount.MountLocal(**non_code_mapping))
    if hasattr(conf, 'SSS_CODE_DIRS_TO_MOUNT'):
        for code_dir in conf.SSS_CODE_DIRS_TO_MOUNT:
            SSS_CODE_MOUNTS.append(
                mount.MountLocal(local_dir=code_dir, pythonpath=True)
            )
except ImportError:
    print("doodad not detected")

target_mount = None


def run_experiment(
        method_call,
        mode='local',
        exp_prefix='default',
        seed=None,
        variant=None,
        exp_id=0,
        prepend_date_to_exp_prefix=True,
        use_gpu=False,
        snapshot_mode='last',
        snapshot_gap=1,
        base_log_dir=None,
        local_input_dir_to_mount_point_dict=None,  # TODO(vitchyr): test this
        # local settings
        skip_wait=False,
        # ec2 settings
        sync_interval=180,
        region='us-east-1',
        instance_type=None,
        spot_price=None,
        verbose=False,
        num_exps_per_instance=1,
        # sss settings
        time_in_mins=None,
        # ssh settings
        ssh_host=None,
        # gcp
        gcp_kwargs=None,
):
    """
    Usage:
    ```
    def foo(variant):
        x = variant['x']
        y = variant['y']
        logger.log("sum", x+y)
    variant = {
        'x': 4,
        'y': 3,
    }
    run_experiment(foo, variant, exp_prefix="my-experiment")
    ```
    Results are saved to
    `base_log_dir/<date>-my-experiment/<date>-my-experiment-<unique-id>`
    By default, the base_log_dir is determined by
    `config.LOCAL_LOG_DIR/`
    :param method_call: a function that takes in a dictionary as argument
    :param mode: A string:
     - 'local'
     - 'local_docker'
     - 'ec2'
     - 'here_no_doodad': Run without doodad call
    :param exp_prefix: name of experiment
    :param seed: Seed for this specific trial.
    :param variant: Dictionary
    :param exp_id: One experiment = one variant setting + multiple seeds
    :param prepend_date_to_exp_prefix: If False, do not prepend the date to
    the experiment directory.
    :param use_gpu:
    :param snapshot_mode: See rlkit.core.logging
    :param snapshot_gap: See rlkit.core.logging
    :param base_log_dir: Will over
    :param sync_interval: How often to sync s3 data (in seconds).
    :param local_input_dir_to_mount_point_dict: Dictionary for doodad.
    :param ssh_host: the name of the host you want to ssh onto, should correspond to an entry in
    config.py of the following form:
    SSH_HOSTS=dict(
        ssh_host=dict(
            username='username',
            hostname='hostname/ip address',
        )
    )
    - if ssh_host is set to None, you will use ssh_host specified by
    config.SSH_DEFAULT_HOST
    :return:
    """
    try:
        import doodad
        import doodad.mode
        import doodad.ssh
    except ImportError:
        print("Doodad not set up! Running experiment here.")
        mode = 'here_no_doodad'
    global ec2_okayed
    global gpu_ec2_okayed
    global target_mount
    global first_sss_launch

    """
    Sanitize inputs as needed
    """
    if seed is None:
        seed = random.randint(0, 100000)
    if variant is None:
        variant = {}
    if mode == 'ssh' and base_log_dir is None:
        base_log_dir = conf.SSH_LOG_DIR
    if base_log_dir is None:
        if mode == 'sss':
            base_log_dir = conf.SSS_LOG_DIR
        else:
            base_log_dir = conf.LOCAL_LOG_DIR

    for key, value in ppp.recursive_items(variant):
        # This check isn't really necessary, but it's to prevent myself from
        # forgetting to pass a variant through dot_map_dict_to_nested_dict.
        if "." in key:
            raise Exception(
                "Variants should not have periods in keys. Did you mean to "
                "convert {} into a nested dictionary?".format(key)
            )
    if prepend_date_to_exp_prefix:
        exp_prefix = time.strftime("%m-%d") + "-" + exp_prefix
    variant['seed'] = str(seed)
    variant['exp_id'] = str(exp_id)
    variant['exp_prefix'] = str(exp_prefix)
    variant['instance_type'] = str(instance_type)

    try:
        import git
        doodad_path = osp.abspath(osp.join(
            osp.dirname(doodad.__file__),
            os.pardir
        ))
        dirs = conf.CODE_DIRS_TO_MOUNT + [doodad_path]

        git_infos = []
        for directory in dirs:
            # Idk how to query these things, so I'm just doing try-catch
            try:
                repo = git.Repo(directory)
                try:
                    branch_name = repo.active_branch.name
                except TypeError:
                    branch_name = '[DETACHED]'
                git_infos.append(GitInfo(
                    directory=directory,
                    code_diff=repo.git.diff(None),
                    code_diff_staged=repo.git.diff('--staged'),
                    commit_hash=repo.head.commit.hexsha,
                    branch_name=branch_name,
                ))
            except git.exc.InvalidGitRepositoryError:
                pass
    except ImportError:
        git_infos = None
    run_experiment_kwargs = dict(
        exp_prefix=exp_prefix,
        variant=variant,
        exp_id=exp_id,
        seed=seed,
        use_gpu=use_gpu,
        snapshot_mode=snapshot_mode,
        snapshot_gap=snapshot_gap,
        git_infos=git_infos,
        script_name=main.__file__,
    )
    if mode == 'here_no_doodad':
        run_experiment_kwargs['base_log_dir'] = base_log_dir
        return run_experiment_here(
            method_call,
            **run_experiment_kwargs
        )

    """
    Safety Checks
    """

    if mode == 'ec2' or mode == 'gcp':
        if not ec2_okayed and not query_yes_no(
                "{} costs money. Are you sure you want to run?".format(mode)
        ):
            sys.exit(1)
        if not gpu_ec2_okayed and use_gpu:
            if not query_yes_no(
                    "{} is more expensive with GPUs. Confirm?".format(mode)
            ):
                sys.exit(1)
            gpu_ec2_okayed = True
        ec2_okayed = True

    """
    GPU vs normal configs
    """
    if use_gpu:
        docker_image = conf.GPU_DOODAD_DOCKER_IMAGE
        if instance_type is None:
            instance_type = conf.GPU_INSTANCE_TYPE
        else:
            assert instance_type[0] == 'g'
        if spot_price is None:
            spot_price = conf.GPU_SPOT_PRICE
    else:
        docker_image = conf.DOODAD_DOCKER_IMAGE
        if instance_type is None:
            instance_type = conf.INSTANCE_TYPE
        if spot_price is None:
            spot_price = conf.SPOT_PRICE
    if mode == 'sss':
        singularity_image = conf.SSS_IMAGE
    elif mode in ['local_singularity', 'slurm_singularity']:
        singularity_image = conf.SINGULARITY_IMAGE
    else:
        singularity_image = None


    """
    Get the mode
    """
    mode_kwargs = {}
    if use_gpu and mode == 'ec2':
        image_id = conf.REGION_TO_GPU_AWS_IMAGE_ID[region]
        if region == 'us-east-1':
            avail_zone = conf.REGION_TO_GPU_AWS_AVAIL_ZONE.get(region, "us-east-1b")
            mode_kwargs['extra_ec2_instance_kwargs'] = dict(
                Placement=dict(
                    AvailabilityZone=avail_zone,
                ),
            )
    else:
        image_id = None
    if hasattr(conf, "AWS_S3_PATH"):
        aws_s3_path = conf.AWS_S3_PATH
    else:
        aws_s3_path = None

    """
    Create mode
    """
    if mode == 'local':
        dmode = doodad.mode.Local(skip_wait=skip_wait)
    elif mode == 'local_docker':
        dmode = doodad.mode.LocalDocker(
            image=docker_image,
            gpu=use_gpu,
        )
    elif mode == 'ssh':
        if ssh_host == None:
            ssh_dict = conf.SSH_HOSTS[conf.SSH_DEFAULT_HOST]
        else:
            ssh_dict = conf.SSH_HOSTS[ssh_host]
        credentials = doodad.ssh.credentials.SSHCredentials(
            username=ssh_dict['username'],
            hostname=ssh_dict['hostname'],
            identity_file=conf.SSH_PRIVATE_KEY
        )
        dmode = doodad.mode.SSHDocker(
            credentials=credentials,
            image=docker_image,
            gpu=use_gpu,
        )
    elif mode == 'local_singularity':
        dmode = doodad.mode.LocalSingularity(
            image=singularity_image,
            gpu=use_gpu,
        )
    elif mode == 'slurm_singularity' or mode == 'sss':
        assert time_in_mins is not None, "Must approximate/set time in minutes"
        if use_gpu:
            kwargs = conf.SLURM_GPU_CONFIG
        else:
            kwargs = conf.SLURM_CPU_CONFIG
        if mode == 'slurm_singularity':
            dmode = doodad.mode.SlurmSingularity(
                image=singularity_image,
                gpu=use_gpu,
                time_in_mins=time_in_mins,
                skip_wait=skip_wait,
                pre_cmd=conf.SINGULARITY_PRE_CMDS,
                **kwargs
            )
        else:
            dmode = doodad.mode.ScriptSlurmSingularity(
                image=singularity_image,
                gpu=use_gpu,
                time_in_mins=time_in_mins,
                skip_wait=skip_wait,
                pre_cmd=conf.SSS_PRE_CMDS,
                **kwargs
            )
    elif mode == 'ec2':
        # Do this separately in case someone does not have EC2 configured
        dmode = doodad.mode.EC2AutoconfigDocker(
            image=docker_image,
            image_id=image_id,
            region=region,
            instance_type=instance_type,
            spot_price=spot_price,
            s3_log_prefix=exp_prefix,
            # Ask Vitchyr or Steven from an explanation, but basically we
            # will start just making the sub-directories within rlkit rather
            # than relying on doodad to do that.
            s3_log_name="",
            gpu=use_gpu,
            aws_s3_path=aws_s3_path,
            num_exps=num_exps_per_instance,
            **mode_kwargs
        )
    elif mode == 'gcp':
        image_name = conf.GCP_IMAGE_NAME
        if use_gpu:
            image_name = conf.GCP_GPU_IMAGE_NAME

        if gcp_kwargs is None:
            gcp_kwargs = {}
        config_kwargs = {
            **conf.GCP_DEFAULT_KWARGS,
            **dict(image_name=image_name),
            **gcp_kwargs
        }
        dmode = doodad.mode.GCPDocker(
            image=docker_image,
            gpu=use_gpu,
            gcp_bucket_name=conf.GCP_BUCKET_NAME,
            gcp_log_prefix=exp_prefix,
            gcp_log_name="",
            **config_kwargs
        )
    else:
        raise NotImplementedError("Mode not supported: {}".format(mode))

    """
    Get the mounts
    """
    mounts = create_mounts(
        base_log_dir=base_log_dir,
        mode=mode,
        sync_interval=sync_interval,
        local_input_dir_to_mount_point_dict=local_input_dir_to_mount_point_dict,
    )

    """
    Get the outputs
    """
    launch_locally = None
    target = conf.RUN_DOODAD_EXPERIMENT_SCRIPT_PATH
    if mode == 'ec2':
        # Ignored since I'm setting the snapshot dir directly
        base_log_dir_for_script = None
        run_experiment_kwargs['force_randomize_seed'] = True
        # The snapshot dir needs to be specified for S3 because S3 will
        # automatically create the experiment director and sub-directory.
        snapshot_dir_for_script = conf.OUTPUT_DIR_FOR_DOODAD_TARGET
    elif mode == 'local':
        base_log_dir_for_script = base_log_dir
        # The snapshot dir will be automatically created
        snapshot_dir_for_script = None
    elif mode == 'local_docker':
        base_log_dir_for_script = conf.OUTPUT_DIR_FOR_DOODAD_TARGET
        # The snapshot dir will be automatically created
        snapshot_dir_for_script = None
    elif mode == 'ssh':
        base_log_dir_for_script = conf.OUTPUT_DIR_FOR_DOODAD_TARGET
        # The snapshot dir will be automatically created
        snapshot_dir_for_script = None
    elif mode in ['local_singularity', 'slurm_singularity', 'sss']:
        base_log_dir_for_script = base_log_dir
        # The snapshot dir will be automatically created
        snapshot_dir_for_script = None
        launch_locally = True
        if mode == 'sss':
            dmode.set_first_time(first_sss_launch)
            first_sss_launch = False
            target = conf.SSS_RUN_DOODAD_EXPERIMENT_SCRIPT_PATH
    elif mode == 'here_no_doodad':
        base_log_dir_for_script = base_log_dir
        # The snapshot dir will be automatically created
        snapshot_dir_for_script = None
    elif mode == 'gcp':
        # Ignored since I'm setting the snapshot dir directly
        base_log_dir_for_script = None
        run_experiment_kwargs['force_randomize_seed'] = True
        snapshot_dir_for_script = conf.OUTPUT_DIR_FOR_DOODAD_TARGET
    else:
        raise NotImplementedError("Mode not supported: {}".format(mode))
    run_experiment_kwargs['base_log_dir'] = base_log_dir_for_script
    target_mount = doodad.launch_python(
        target=target,
        mode=dmode,
        mount_points=mounts,
        args={
            'method_call': method_call,
            'output_dir': snapshot_dir_for_script,
            'run_experiment_kwargs': run_experiment_kwargs,
            'mode': mode,
        },
        use_cloudpickle=True,
        target_mount=target_mount,
        verbose=verbose,
        launch_locally=launch_locally,
    )


def create_mounts(
        mode,
        base_log_dir,
        sync_interval=180,
        local_input_dir_to_mount_point_dict=None,
):
    if mode == 'sss':
        code_mounts = SSS_CODE_MOUNTS
        non_code_mounts = SSS_NON_CODE_MOUNTS
    else:
        code_mounts = CODE_MOUNTS
        non_code_mounts = NON_CODE_MOUNTS

    if local_input_dir_to_mount_point_dict is None:
        local_input_dir_to_mount_point_dict = {}
    else:
        raise NotImplementedError("TODO(vitchyr): Implement this")

    mounts = [m for m in code_mounts]
    for dir, mount_point in local_input_dir_to_mount_point_dict.items():
        mounts.append(mount.MountLocal(
            local_dir=dir,
            mount_point=mount_point,
            pythonpath=False,
        ))

    if mode != 'local':
        for m in non_code_mounts:
            mounts.append(m)

    if mode == 'ec2':
        output_mount = mount.MountS3(
            s3_path='',
            mount_point=conf.OUTPUT_DIR_FOR_DOODAD_TARGET,
            output=True,
            sync_interval=sync_interval,
            include_types=('*.txt', '*.csv', '*.json', '*.gz', '*.tar',
                           '*.log', '*.pkl', '*.mp4', '*.png', '*.jpg',
                           '*.jpeg', '*.patch'),
        )
    elif mode == 'gcp':
        output_mount = mount.MountGCP(
            gcp_path='',
            mount_point=conf.OUTPUT_DIR_FOR_DOODAD_TARGET,
            output=True,
            gcp_bucket_name=conf.GCP_BUCKET_NAME,
            sync_interval=sync_interval,
            include_types=('*.txt', '*.csv', '*.json', '*.gz', '*.tar',
                           '*.log', '*.pkl', '*.mp4', '*.png', '*.jpg',
                           '*.jpeg', '*.patch'),
        )

    elif mode in ['local', 'local_singularity', 'slurm_singularity', 'sss']:
        # To save directly to local files (singularity does this), skip mounting
        output_mount = mount.MountLocal(
            local_dir=base_log_dir,
            mount_point=None,
            output=True,
        )
    elif mode == 'local_docker':
        output_mount = mount.MountLocal(
            local_dir=base_log_dir,
            mount_point=conf.OUTPUT_DIR_FOR_DOODAD_TARGET,
            output=True,
        )
    elif mode == 'ssh':
        output_mount = mount.MountLocal(
            local_dir=base_log_dir,
            mount_point=conf.OUTPUT_DIR_FOR_DOODAD_TARGET,
            output=True,
        )
    else:
        raise NotImplementedError("Mode not supported: {}".format(mode))
    mounts.append(output_mount)
    return mounts
