"""Aggregate evalCli success rates across an experiment folder.

Finds every {dataset}_summary.csv below --base_dir (one per evaluated checkpoint and
dataset, written by evalCli under <save_dir>/<out_dir>/<stem>/), counts HOME/OOB/OOT
outcomes, and writes one TSV row per file with the success percentage, plus (when the csvs have a
rotate_by column from --eval_rotations) a *_by_rotation.tsv with one row per file and rotation angle.

Usage:
    python3 -m tamagotchi.eval.tabulate_from_summary --base_dir <save_dir> [--subdir eval]
"""
import tqdm
import os
import pandas as pd
import argparse
import glob
import datetime

def parse_summary_files(fnames, BASE_DIR, by_rotation=False):
    """One row per summary csv; with by_rotation, one row per (csv, rotate_by) instead.
    rotate_by is the plume rotation angle written by `evalCli --eval_rotations` (NaN if the csv has none)."""
    counts_df = []
    for fname in tqdm.tqdm(fnames):
        s_all = pd.read_csv(fname)
        if by_rotation and 'rotate_by' in s_all.columns:
            parts = [(angle, g) for angle, g in s_all.groupby('rotate_by')]
        else:
            parts = [(float('nan'), s_all)]
        for angle, s in parts:
            counts_df.append(_summary_row(fname, s, BASE_DIR, angle, by_rotation))
    return _finalize(counts_df)

def _summary_row(fname, s, BASE_DIR, angle, by_rotation):
    dataset = str(fname).split('/')[-1].replace('_summary.csv','')
    row = {
        'dataset': dataset,
        'HOME': sum(s['reason'] == 'HOME'),
        'OOB': sum(s['reason'] == 'OOB'),
        'OOT': sum(s['reason'] == 'OOT'),
        'total': len(s['reason']),
        # the eval subdir is the checkpoint stem, e.g. plume_seed-3-4deaf7e9.chkpt -> 'seed-3-4deaf7e9'
        'seed': os.path.basename(os.path.dirname(str(fname))).split('_')[1],
        'model_dir': str(fname).replace(f'{dataset}_summary.csv','').replace(os.path.expanduser(BASE_DIR), ''),
        'code': str(fname).split('code')[-1].split('_')[0],
        'fname': str(fname)
    }
    if by_rotation:
        row['rotate_by'] = angle
    if not row['seed']:
        row['seed'] = row["model_dir"].split("_")[1]
    return row

def _finalize(counts_df):
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    counts_df = pd.DataFrame(counts_df)
    # Eg poisson_mag_noisy3x5b5_0.7 - first check if the last part is a number, if not, set it to 1
    counts_df['relative_plume_density'] = [ds.split("_")[-1] if is_number(ds.split("_")[-1]) else 1 for ds in counts_df['dataset']]
    # Eg poisson_mag_noisy3x5b5_0.7 - then check if the last part is a number, if not, the whole string is the condition
    counts_df['condition'] = ["_".join(ds.split("_")[:-1]) if is_number(ds.split("_")[-1]) else ds for ds in counts_df['dataset']]
    counts_df['Success_pct'] = counts_df['HOME'] / counts_df['total'] * 100
    return counts_df

def dataset_matches(fname, prefixes):
    """True if the summary file's dataset name starts with one of `prefixes`, so that
    'eval_noisy_jitter' keeps eval_noisy_jitterx5b5[_0.3]_summary.csv but not
    eval_noisy_jitter_coveragex5b5[_0.3]_summary.csv. After the prefix, the rest of the name
    must be empty, a plume-density suffix (_0.3), or continue the word (x5b5...), not start a new '_' part."""
    name = os.path.basename(str(fname)).replace('_summary.csv', '')
    for p in prefixes:
        if not name.startswith(p):
            continue
        rest = name[len(p):]
        if rest == '' or not rest.startswith('_') or rest[1:].replace('.', '', 1).isdigit():
            return True
    return False

def main(args):
    if args.substr:
        files = set(glob.glob(f"{args.base_dir}/*{args.substr}*/**/*_summary.csv", recursive=True))
        files |= set(glob.glob(f"{args.base_dir}/**/*{args.substr}*_summary.csv", recursive=True))
        files = sorted(files)
        print(f"Reading {args.base_dir}/*{args.substr}*/**, {len(files)} files found")
    elif args.subdir_prefix:
        files = glob.glob(f"{args.base_dir}/{args.subdir_prefix}*/**/*_summary.csv", recursive=True)
        print(f"Reading directory {args.base_dir}/{args.subdir_prefix}*/**, {len(files)} files found")
    elif args.subdir:
        files = glob.glob(f"{args.base_dir}/{args.subdir}/**/*_summary.csv", recursive=True)
        print(f"Reading directory {args.base_dir}/{args.subdir}/**/, {len(files)} files found")
    else:
        files = glob.glob(f"{args.base_dir}/**/*_summary.csv", recursive=True)
        print(f"Reading directory {args.base_dir}, {len(files)} files found")
    
    if args.dataset:
        files = sorted(f for f in files if dataset_matches(f, args.dataset))
        print(f"--dataset {args.dataset}: kept {len(files)} files")

    assert len(files) > 0, "No files found, check the directory"
    summary_dfs = parse_summary_files(files, args.base_dir)
    summary_rot = parse_summary_files(files, args.base_dir, by_rotation=True)

    current_date = datetime.date.today()

    if args.out_prefix:
        full_out_path = f'{args.base_dir}/{args.out_prefix}.tsv'
    else:
        full_out_path = f'{args.base_dir}/performance_all_{current_date}.tsv'
    summary_dfs.to_csv(full_out_path, sep='\t', index=False)
    print(f"Saved to {full_out_path}")
    if summary_rot['rotate_by'].notna().any():
        rot_out_path = full_out_path.replace('.tsv', '_by_rotation.tsv')
        summary_rot.to_csv(rot_out_path, sep='\t', index=False)
        print(f"Saved per-rotation rows to {rot_out_path}")
    return summary_dfs
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tabulate evalCli success rates from *_summary.csv files')
    parser.add_argument('--base_dir', default=os.getcwd(), type=str, help='Base directory of the experiment')
    parser.add_argument('--subdir_prefix', default='', type=str, help='Find subdirs that startwith the suffix')
    parser.add_argument('--substr', default='', type=str, help='Find subdirs (or summary files) whose name contains the substring')
    parser.add_argument('--dataset', nargs='+', default=[], help='Only keep *_summary.csv whose dataset name starts with one of these, '
                        'e.g. --dataset eval_noisy_jitter eval_constant_jitter (skips eval_noisy_jitter_coveragex5b5)')
    parser.add_argument('--subdir', default='', type=str, help='Find the subdir')
    parser.add_argument('--out_prefix', default='', type=str, help='Output file prefix')
    
    args = parser.parse_args()
    main(args)
