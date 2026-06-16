# Usage: python3 '/gscratch/portia/jqhu/work/active_sensing/smartFlies/scripts/slurm-run_ckpt.py' 
#               --n_seeds 15 --override "action_physics=force OU_exploration=off experiment_name=force_physics_uncertainty 'r_shaping=[step,missed_time_cost,rotate_by,birthx_cl_last,cosine]' loc_algo=linear_precise num_env_steps=3600000 precise_max=[6] variant=wind_obsver_v1 'dataset=[constant_jitterx5b5]' path.curriculum_name=060226_const_jitter" 
#
# Checkpointing: training saves {save_dir}/weights/{env_name}_{outsuffix}.pt every save_interval
# updates. On requeue the script restarts with the same args; if that file exists, training
# resumes automatically from the last saved checkpoint (no load_jobid logic needed).
#
# Cancel all your jobs: squeue -u $USER -h | awk '{print $1}' | xargs scancel


# Staged training after conti
# python3 scripts/slurm-run_ckpt.py \
#   --from_folder /gscratch/portia/jqhu/work/active_sensing/smartFlies/data/wind_sensing/apparent_wind_visual_feedback/force_physics_uncertainty/ \
#   --stage_name noisy \
#   --substr 4deaf7e9 \
#   --override "action_physics=force OU_exploration=off experiment_name=force_physics_uncertainty 'r_shaping=[step,missed_time_cost,rotate_by,birthx_cl_last,cosine]' loc_algo=linear_precise num_env_steps=7200000 precise_max=[1] precise_max=[6] variant=wind_obsver_v1 'dataset=[noisy_jitterx5b5]' path.curriculum_name=060226_noisy_jitter" 

# Drone finetune
# python3 scripts/slurm-run_ckpt.py \
#   --from_folder /gscratch/portia/jqhu/work/active_sensing/smartFlies/data/wind_sensing/apparent_wind_visual_feedback/force_physics_uncertainty/ \
#   --time 0-04:00:00 \
#   --stage_name drone_physics \
#   --override "action_physics=force OU_exploration=off experiment_name=drone_physics r_shaping=[step,missed_time_cost,rotate_by,birthx_cl_last,cosine] loc_algo=linear_precise num_env_steps=4000000 precise_min=[1] precise_max=[6] variant=wind_obsver_v1 dataset=[noisy_jitterx5b5] path.curriculum_name=061626_noisy_jitter_dron_finetune" \
#   --substr seed-4-4deaf7e9_stage
#   --dry_run \

# for seed in 4 6 7 9 10 11 21 24 29 30; do
#     python3 scripts/slurm-run_ckpt.py \
#         --from_folder /gscratch/portia/jqhu/work/active_sensing/smartFlies/data/wind_sensing/apparent_wind_visual_feedback/force_physics_uncertainty/ \
#         --time 0-04:00:00 \
#         --stage_name drone_physics_dt01 \
#         --override "env_dt=0.1 action_physics=force OU_exploration=off experiment_name=drone_physics_dt01 r_shaping=[step,missed_time_cost,rotate_by,birthx_cl_last,cosine] loc_algo=linear_precise num_env_steps=4000000 precise_min=[1] precise_max=[6] variant=wind_obsver_v1 dataset=[noisy_jitterx5b5] path.curriculum_name=061626_noisy_jitter_dron_finetune" \
#         --substr "seed-${seed}-4deaf7e9_stage" \
#         --dry_run
# done

'''
python3 /gscratch/portia/jqhu/work/active_sensing/smartFlies/scripts/slurm-run_ckpt.py --dry_run --config config_control_noCL --n_seeds 30 dry_run--override "action_physics=force OU_exploration=off experiment_name=CTL_noCL 'r_shaping=[step,missed_time_cost,rotate_by]' num_env_steps=10800000 variant=wind_obsver_v1"
'''

import argparse
import glob
import os
import re
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
        dry_run
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
    safe_override = re.sub(r'[^A-Za-z0-9_\-]', '_', override)[:60]
    script_path = os.path.join(OUTFILES_DIR, f'submit_{config_name}_{safe_override}.sh')
    if not dry_run:
        with open(script_path, 'w') as f:
            f.write(script)
        job_id = slurm_submit(script_path)
        print(f'Submitted job {job_id}')
    else:
        print(f'Dry run: Would create script at {script_path}')


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
    parser.add_argument('--from_folder', type=str, default='',
                        help='Path to experiment folder containing weights/ — discovers trained '
                             'seeds automatically (mutually exclusive with --n_seeds loop)')
    parser.add_argument('--stage_name', type=str, default='',
                        help='Short label for this continuation stage (e.g. "morewind"); '
                             'required when --from_folder is used')
    parser.add_argument('--substr', type=str, default='',
                        help='Only load checkpoints whose filename contains this substring '
                             '(e.g. a hash "4deaf7e9" to pin a specific run)')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='Print the commands that would be executed without actually submitting the jobs')

    args = parser.parse_args()

    submit_kwargs = dict(
        config_name=args.config,
        num_gpus=args.num_gpus,
        gpu_type=args.gpu_type,
        mem=args.mem,
        cpus=args.cpus,
        time=args.time,
        partition=args.partition,
        dry_run=args.dry_run
    )

    if args.from_folder:
        if not args.stage_name:
            print('Error: --stage_name is required when --from_folder is used', file=sys.stderr)
            sys.exit(1)
        # Discover seed checkpoints: plume_seed-N-HASH.pt (or .chkpt.pt for older runs)
        pattern = os.path.join(args.from_folder, 'weights', 'plume_seed-*.pt')
        pt_files = sorted(glob.glob(pattern))
        pt_files = [f for f in pt_files
                    if re.search(r'seed-\d+-[0-9a-f]{8}(_stage_[0-9a-f]{8})?(\.chkpt)?\.pt$', f)
                    and not f.endswith('_vecNormalize.pkl')
                    and (not args.substr or args.substr in os.path.basename(f))]
        if not pt_files:
            print(f'No seed checkpoints found in {args.from_folder}/weights/', file=sys.stderr)
            sys.exit(1)
        if args.dry_run:
            print('Dry run mode: found following checkpoints:')
            print('\n'.join(pt_files))
        for pt_file in pt_files:
            stem = os.path.splitext(os.path.basename(pt_file))[0]  # plume_seed-22-4deaf7e9 or .chkpt
            outsuffix = stem.replace('plume_', '', 1).replace('.chkpt', '')  # seed-22-4deaf7e9
            seed_n = int(outsuffix.split('-')[1])
            seed_override = (
                f'outsuffix={outsuffix} '
                f'resume_from={pt_file} '
                f'stage_name={args.stage_name} '
                f'seed={seed_n} '
                f'{args.override}'
            ).strip()
            submit(override=seed_override, **submit_kwargs)
            if args.dry_run:
                break  # just show the first one in dry run
    else:
        for seed in range(args.seed_from, args.seed_from + args.n_seeds):
            seed_override = f'{args.override} seed={seed}'.strip()
            submit(override=seed_override, **submit_kwargs)
            if args.dry_run:
                break  # just show the first one in dry run


if __name__ == '__main__':
    main()