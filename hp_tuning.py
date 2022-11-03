# An example of running a job by the python interface directly.

# To set up a job, the user needs to specify `method` and `hp_dict` and then
# import `submit_xt_job`. Optionally, the user can provide additional arguments
# `submit_xt_job` takes. To run the job, go to the `rl_nexus` directory and then
# execute this python file.

from rl_nexus.hp_tuning_tools import submit_xt_job


### Code setup
# The python method to run.
method = "rl_nexus.hand_dapg.hp_tuning.train"  # so we can call the method below.
def train(**config):
    """A wrapper to call the main function.
    config: a dict of desired hp values.
    """
    import os, json
    from rl_nexus.hand_dapg.dapg.examples.job_script import main

    with open(config['config'], 'r') as f:
        job_data = eval(f.read())

    # Update the config
    log_root = config['log_root']
    config['policy_size'] = tuple(config['network_width'] for _ in range(config['network_depth']))
    del config['log_root'], config['config'], config['network_width'], config['network_depth']
    job_data.update(config)
    JOB_DIR = os.path.join(log_root, job_data['env']+'_'+str(job_data['seed']))

    return main(JOB_DIR, job_data)


if __name__=='__main__':
    # (optional) A directory or a list of directories to upload
    code_paths = None # ['../hp_tuning_scripts'] # A path AAA/BBB/CCC will be uploaded as rl_nexus/CCC

    # (optional) Commandlines to setup the code.
    setup_cmds = [  # This starts from the `dilbert` directory.
    'cd rl_nexus/',
    'git clone https://github.com/chinganc/hand_dapg.git',
    'cd hand_dapg',
    'bash install.sh',
    'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin',
    'cd ..',  # back to rl_nexus
    'cd ../',  # back to dilbert
    ]

    ### Hyperparameter setting
    # A dict that specifies the values of hyperparameters to search over.
    hps_dict = dict(
                env=["kitchen_knob1_on-v3", "kitchen_light_on-v3","kitchen_sdoor_open-v3","kitchen_ldoor_open-v3","kitchen_micro_open-v3"],
                )

    # (optional) A dict that specifies the default values of hyperparameters. This
    # is meant to overwrite the default hyperparameter used in the method.
    # Therefore, this is optional, and keys here do not necessary have to appear in
    # `hps_dict` (vice versa)
    config = dict(
            log_root="../results",
            config="hand_dapg/dapg/examples/rl_scratch.txt",
            seed='randint',
            network_depth=2,
            network_width=32,
            rl_num_iter=1000,
        )

    # (optional) Number of seeds to run per hyperparameter.
    n_seeds_per_hp=20

    ### (optional) Compute resources
    compute_target='azb-cpu' # Name of the compute resource.
    docker_image='mujoco' # Name of the Docker image.
    azure_service='dilbertbatch'
    vm_size='Standard_F16s_v2'
    max_n_nodes=100 # Maximal number of nodes to launch in the job.
    max_total_runs=3000 # Maximal number of runs in the job.
    # n_sequential_runs_per_node=1  # Number of sequential runs per node.
    # n_concurrent_runs_per_node=1  # Number of concurrent runs per node.
    # low_priority=True  # Whether the job can is preemptible.
    # hold=False # whether the nodes should be released automatically (hold=False) after the job completes.
    remote_run=True  # Set False to debug locally.

    submit_xt_job(method,
                hps_dict,
                code_paths=code_paths,
                setup_cmds=setup_cmds,
                config=config,
                n_seeds_per_hp=n_seeds_per_hp,
                compute_target=compute_target,
                docker_image=docker_image,
                azure_service=azure_service,
                max_n_nodes=max_n_nodes,
                max_total_runs=max_total_runs,
                remote_run=remote_run,
                vm_size=vm_size)