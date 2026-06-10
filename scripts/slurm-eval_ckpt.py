# Usage: python3 scripts/slurm-eval_ckpt.py \
#               --from_folder /path/to/experiment/ \
#               --dataset constant_jitterx5b5 \
#               --out_dir eval \
#               --time_offsets 0.0 1.0 10.0 11.0 20.0 21.0 30.0 31.0 \
#               --viz_episodes 20
#
# Discovers all seed checkpoints in {from_folder}/weights/ and submits one
# SLURM eval job per checkpoint. Logs are saved alongside weights with the
# same stem but in {from_folder}/{out_dir}/*.evallog
#
# Cancel all your jobs: squeue -u $USER -h | awk '{print $1}' | xargs scancel

# python3 scripts/slurm-eval_ckpt.py --from_folder /gscratch/portia/jqhu/work/active_sensing/smartFlies/data/wind_sensing/apparent_wind_visual_feedback/force_physics_uncertainty/ --dataset eval_noisy_jitterx5b5 --substr stage --dry_run

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


def submit_eval(
        model_fname,
        dry_run,
        dataset,
        out_dir,
        time_offsets,
        viz_episodes,
        num_gpus,
        gpu_type,
        mem,
        cpus,
        time,
        partition,
        extra_args,
        ):
    gpu_resource = GPU_CONFIGS[gpu_type]
    group_name = 'walkerlab' if partition == 'gpu-a100' else 'portia'

    # Derive log path: replace weights/ dir with out_dir, .pt -> .evallog
    logfile = model_fname.replace('.pt', '.evallog')
    # Replace the weights component of the path with out_dir
    logfile = re.sub(r'/weights/', f'/{out_dir}/', logfile)

    logdir = os.path.dirname(logfile)
    logbase = os.path.basename(logfile)

    offsets_str = ' '.join(str(t) for t in time_offsets)
    modifier = f'--out_dir {out_dir} --time_offsets {offsets_str}'

    os.makedirs(OUTFILES_DIR, exist_ok=True)

    script = f"""#!/bin/bash
#SBATCH --job-name=eval-{partition}
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
echo "Log directory = {logdir}"
mkdir -p {logdir}
echo "Saving eval logs in {logbase}"
python3 -u tamagotchi/evalCli_hydra.py \\
    --dataset {dataset} \\
    --fixed_eval {modifier} \\
    --viz_episodes {viz_episodes} \\
    --model_fname {model_fname} \\
    {extra_args} >> {logfile} 2>&1
"""

    print('Submitting eval job with script:')
    print(script)
    safe_stem = re.sub(r'[^A-Za-z0-9_\-]', '_', os.path.basename(model_fname))[:60]
    script_path = os.path.join(OUTFILES_DIR, f'eval_{safe_stem}.sh')
    if dry_run:
        exit(0)
    with open(script_path, 'w') as f:
        f.write(script)
    job_id = slurm_submit(script_path)
    print(f'Submitted eval job {job_id} for {os.path.basename(model_fname)}')


def main():
    parser = argparse.ArgumentParser(
        description='Submit smartFlies eval jobs to SLURM — one job per checkpoint.')
    parser.add_argument('--from_folder', type=str, required=True,
                        help='Path to experiment folder containing weights/ subdirectory')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name to evaluate on (e.g. constant_jitterx5b5)')
    parser.add_argument('--out_dir', type=str, default='eval',
                        help='Output subdirectory name inside the experiment folder (default: eval)')
    parser.add_argument('--time_offsets', type=float, nargs='+',
                        default=[0.0, 10.0, 20.0, 30.0],
                        # default=[0.0, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
                        help='Time offsets in seconds passed to evalCli (default: 0.0 1.0 10.0 11.0 20.0 21.0 30.0 31.0)')
    parser.add_argument('--viz_episodes', type=int, default=20,
                        help='Number of episodes to visualize (default: 20)')
    parser.add_argument('--substr', type=str, default='',
                        help='Only eval checkpoints whose filename contains this substring')
    parser.add_argument('--extra_args', type=str, default='',
                        help='Additional arguments passed verbatim to evalCli.py')
    parser.add_argument('--num_gpus', type=int, default=0,
                        help='Number of GPUs (default: 0)')
    parser.add_argument('--gpu_type', type=str, default='all', choices=list(GPU_CONFIGS),
                        help='GPU hardware type to request (default: all)')
    parser.add_argument('--mem', type=int, default=32,
                        help='Memory in GB (default: 32)')
    parser.add_argument('--cpus', type=int, default=8,
                        help='CPU cores (default: 8)')
    parser.add_argument('--time', type=str, default='4:00:00',
                        help='Wall time limit HH:MM:SS (default: 4:00:00)')
    parser.add_argument('--partition', type=str, default='ckpt-all',
                        help='SLURM partition (default: ckpt-all)')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='Print the commands that would be executed without actually submitting the jobs')

    args = parser.parse_args()

    pattern = os.path.join(args.from_folder, 'weights', 'plume_seed-*.pt')
    pt_files = sorted(glob.glob(pattern))
    if args.substr == 'stage':
        print('Filtering for stage checkpoints...')
        # Matches plume_seed-3-4deaf7e9_stage_3323316f.chkpt.pt (not plain .pt seeds)
        pt_files = [f for f in pt_files
                    if re.search(r'seed-\d+-[0-9a-f]{8}_stage_[0-9a-f]+\.chkpt\.pt$', f)
                    and not f.endswith('_vecNormalize.pkl')]
        print('\n'.join(pt_files))
    elif args.substr:
        print(f'Filtering for checkpoints containing "{args.substr}"...')
        pt_files = [f for f in pt_files
                    if re.search(r'seed-\d+-[0-9a-f]{8}(\.chkpt)?\.pt$', f)
                    and not f.endswith('_vecNormalize.pkl')
                    and (args.substr in os.path.basename(f))]

    if not pt_files:
        print(f'No seed checkpoints found in {args.from_folder}/weights/', file=sys.stderr)
        sys.exit(1)

    print(f'Found {len(pt_files)} checkpoint(s) to evaluate.')

    if args.dry_run:
        print('Dry run mode: found following pts')
        print('\n'.join(pt_files))

    for pt_file in pt_files:
        submit_eval(
            model_fname=pt_file,
            dataset=args.dataset,
            out_dir=args.out_dir,
            time_offsets=args.time_offsets,
            viz_episodes=args.viz_episodes,
            num_gpus=args.num_gpus,
            gpu_type=args.gpu_type,
            mem=args.mem,
            cpus=args.cpus,
            time=args.time,
            partition=args.partition,
            extra_args=args.extra_args,
            dry_run=args.dry_run,
        )


if __name__ == '__main__':
    main()
