# python3 /src/tamagotchi/eval/tabulate_from_summary.py --base_dir /src/data/TrainingCurriculum/sw_dist_logstep_wind_cond_0.01/
# python3 tamagotchi/eval/tabulate_from_summary.py --base_dir data/wind_sensing/apparent_wind_only/sw_dist_logstep_wind_0.001_best_seeds/

# aggregate performance eval results of all models in a directory 

import tqdm
import os
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
import argparse
import glob
import datetime
import mlflow

def parse_summary_files(fnames, BASE_DIR):
    counts_df = []
    for fname in tqdm.tqdm(fnames):
        s = pd.read_csv(fname) 
        dataset = str(fname).split('/')[-1].replace('_summary.csv','')
        row = {
            'dataset': dataset,
            'HOME': sum(s['reason'] == 'HOME'),
            'OOB': sum(s['reason'] == 'OOB'),
            'OOT': sum(s['reason'] == 'OOT'),
            'total': len(s['reason']),
            # 'seed': str(fname).split('seed')[-1].split('/')[0], # old fname scheme
            'seed': os.path.basename(os.path.dirname(str(fname))).split('_')[1],
            'model_dir': str(fname).replace(f'{dataset}_summary.csv','').replace(os.path.expanduser(BASE_DIR), ''),
            'code': str(fname).split('code')[-1].split('_')[0],
            'fname': str(fname)
        }
        if not row['seed']:
            # dj fname naming scheme changed... get seed info from model_dir
            row['seed'] = row["model_dir"].split("_")[1]
            
        counts_df.append(row)
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
    # pivot_df = counts_df.pivot(index=['model_dir', 'seed'], columns='dataset', values='HOME').reset_index()
    return counts_df

def main(args):
    if args.subdir_prefix:
        files = glob.glob(f"{args.base_dir}/{args.subdir_prefix}*/**/*_summary.csv", recursive=True)
        print(f"Reading directory {args.base_dir}/{args.subdir_prefix}*/**, {len(files)} files found")
    elif args.subdir:
        files = glob.glob(f"{args.base_dir}/{args.subdir}/**/*_summary.csv", recursive=True)
        print(f"Reading directory {args.base_dir}/{args.subdir}/**/, {len(files)} files found")
    else:
        files = glob.glob(f"{args.base_dir}/**/*_summary.csv", recursive=True)
        print(f"Reading directory {args.base_dir}, {len(files)} files found")
    
    assert len(files) > 0, "No files found, check the directory"
    summary_dfs = parse_summary_files(files, args.base_dir)

    current_date = datetime.date.today()

    if args.out_prefix:
        full_out_path = f'{args.base_dir}/{args.out_prefix}.tsv'
    else:
        full_out_path = f'{args.base_dir}/performance_all_{current_date}.tsv'
    summary_dfs.to_csv(full_out_path, sep='\t', index=False)
    print(f"Saved to {full_out_path}")
    return summary_dfs
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--base_dir', default=os.getcwd(), type=str, help='Base directory of the experiment')
    parser.add_argument('--subdir_prefix', default='', type=str, help='Find subdirs that startwith the suffix')
    parser.add_argument('--subdir', default='', type=str, help='Find the subdir')
    parser.add_argument('--out_prefix', default='', type=str, help='Output file prefix')
    
    args = parser.parse_args()
    main(args)
    # mlflow.set_tracking_uri(uri="http://dev0.uwcnc.net:5000/")
    # mlflow.set_system_metrics_sampling_interval(60)
    # # Create a new MLflow Experiment
    # experiment_name = os.path.basename(args.base_dir) 
    # experiment = mlflow.get_experiment_by_name(experiment_name)
    # experiment_id = experiment.experiment_id
    # # Create an MLflow client
    # client = mlflow.tracking.MlflowClient()
    # # Get the list of runs for the experiment, sorted by start time
    # runs = client.search_runs(experiment_ids=[experiment_id], order_by=["start_time DESC"])

    # run_name = 'eval'
    # mlflow.set_experiment(experiment_name)
    # # Start an MLflow run
    # with mlflow.start_run(run_name=run_name):
    #     summary_dfs = main(args)
    #     summary_dfs['condition'] = df['condition'].astype('category')
    #     summary_dfs['seed'] = df['seed'].astype('category')

    #     summary_dfs['experiment'] = [ps.split('/')[1] for ps in summary_dfs['model_dir']]

    # print(set(df.experiment))
    # print(len(set(df.seed)))
    # # log summary df as a table artifact in mlflow
    # mlflow.log_artifact(summary_dfs, artifact_path='summary_table')
    