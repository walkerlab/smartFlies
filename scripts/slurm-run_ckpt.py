# Usage: python3 '/gscratch/portia/jqhu/work/active_sensing/smartFlies/scripts/slurm-run_ckpt.py' 
#               --n_seeds 15 --override "action_physics=force OU_exploration=off experiment_name=force_physics_uncertainty 'r_shaping=[step,missed_time_cost,rotate_by,birthx_cl_last,cosine]' loc_algo=linear_precise num_env_steps=3600000 precise_max=[6] variant=wind_obsver_v1 'dataset=[constant_jitterx5b5]' path.curriculum_name=060226_const_jitter" 
#
# Checkpointing: training saves {save_dir}/weights/{env_name}_{outsuffix}.pt every save_interval
# updates. On requeue the script restarts with the same args; if that file exists, training
# resumes automatically from the last saved checkpoint (no load_jobid logic needed).
#
# Cancel all your jobs: squeue -u $USER -h | awk '{print $1}' | xargs scancel


import argparse
import os
import subprocess
import sys

PROJECT_DIR = '/gscratch/portia/jqhu/work/active_sensing/smartFlies/'
OUTFILES_DIR = os.path.join(PROJECT_DIR, 'scripts/slurm_outfiles')

GPU_CONFIGS = {
    'all':  'g[3040-3047,3050-3057,3060-3067,3070-3077,3080-3087,3090-3097,3091-3113,3115-3132]',
    'a100': 'g[3040-3047,3050-3057,3060-3067,3070-3077,3080-3087]',
    'l40s': 'g[3091-3113,3115-3132]',
    'h200': 'g[3125-3132]',
}


def slurm_submit(script_path):
    try:
        output = subprocess.check_output(['sbatch', script_path], universal_newlines=True)
        job_id = output.strip().split()[-1]
        return job_id
    except subprocess.CalledProcessError as e:
        print(f'Error submitting job: {e.output}', file=sys.stderr)
        sys.exit(1)


def submit(
        override,
        config_name,
        num_gpus,
        gpu_type,
        mem,
        cpus,
        time,
        partition,
        ):
    gpu_resource = GPU_CONFIGS[gpu_type]
    group_name = 'walkerlab' if partition == 'gpu-a100' else 'portia'

    os.makedirs(OUTFILES_DIR, exist_ok=True)

    script = f"""#!/bin/bash
#SBATCH --job-name={partition}
#SBATCH --partition={partition}
#SBATCH --account={group_name}
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gpus={num_gpus}
#SBATCH --mem={mem}G
#SBATCH --verbose
#SBATCH --open-mode=append
#SBATCH -o {OUTFILES_DIR}/slurm-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jqhu@uw.edu
#SBATCH --nodelist={gpu_resource}
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
cat $0
module load cuda/12.9.1
set -x
source ~/.bashrc
nvidia-smi
conda activate tamagotchi
unset LD_LIBRARY_PATH
echo $SLURMD_NODENAME
cd {PROJECT_DIR}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python3 -u -m tamagotchi.main_hydra \\
  --config-name {config_name} \\
  {override}
"""

    print('Submitting job with script:')
    print(script)
    script_path = os.path.join(OUTFILES_DIR, f'submit_{config_name}_{override.replace(" ", "_")[:60]}.sh')
    with open(script_path, 'w') as f:
        f.write(script)
    job_id = slurm_submit(script_path)
    print(f'Submitted job {job_id}')


def main():
    parser = argparse.ArgumentParser(
        description='Submit a smartFlies training job to SLURM with auto-requeue checkpointing.')
    parser.add_argument('--config', type=str, default='config',
                        help='Hydra config name to load from tamagotchi/conf/ (default: config)')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs (default: 1; parallelism via num_processes CPUs)')
    parser.add_argument('--gpu_type', type=str, default='all', choices=list(GPU_CONFIGS),
                        help='GPU hardware type to request (default: all)')
    parser.add_argument('--mem', type=int, default=32,
                        help='Memory in GB (default: 32)')
    parser.add_argument('--cpus', type=int, default=16,
                        help='CPU cores (default: 16)')
    parser.add_argument('--time', type=str, default='1-00:00:00',
                        help='Wall time limit day-HH:MM:SS (default: 1-00:00:00)')
    parser.add_argument('--partition', type=str, default='ckpt-all',
                        help='SLURM partition (default: ckpt-all)')
    parser.add_argument('--override', type=str, default='',
                        help='Additional Hydra overrides passed directly to main_hydra '
                             '(e.g. "outsuffix=run01 action_physics=force seed=1")')
    parser.add_argument('--n_seeds', type=int, default=1,
                        help='Submit N jobs with seed=1 through seed=N (default: 1)')
    parser.add_argument('--seed_from', type=int, default=1,
                        help='Start submitting jobs from this seed (default: 1)')

    args = parser.parse_args()

    for seed in range(args.seed_from, args.seed_from + args.n_seeds):
        seed_override = f'{args.override} seed={seed}'.strip()
        submit(
            override=seed_override,
            config_name=args.config,
            num_gpus=args.num_gpus,
            gpu_type=args.gpu_type,
            mem=args.mem,
            cpus=args.cpus,
            time=args.time,
            partition=args.partition,
        )


if __name__ == '__main__':
    main()