"""evalCli_hydra.py — evaluation entry point that loads training args from the
saved _args.json (written by main.py during training) and compares them against
the evalCli argparse defaults before running the eval loop.

Usage:
    python evalCli_hydra.py --model_fname /path/to/weights/plume_<suffix>.pt [--proceed]

The script will:
1. Derive the matching _args.json path from the weight file location.
2. Load that JSON as the "training config".
3. Build the evalCli argparse namespace (defaults only, no user overrides).
4. Print a side-by-side diff of the two configs.
5. Stop unless --proceed is passed.
"""
from __future__ import division
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from evalCli import eval_loop
import torch
import json
import argparse
import sys


# ---------------------------------------------------------------------------
# Reconstruct the evalCli argparse defaults (no user input — just defaults)
# ---------------------------------------------------------------------------
def build_evalcli_defaults() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=137)
    parser.add_argument('--algo', default='ppo')
    parser.add_argument('--diffusionx', type=float, default=1.0)
    parser.add_argument('--obs_noise', type=float, default=0.0)
    parser.add_argument('--apparent_wind', type=bool, default=False)
    parser.add_argument('--visual_feedback', type=bool, default=False)
    parser.add_argument('--saccade', type=bool, default=False)
    parser.add_argument('--haltere', type=bool, default=False)


    # Hardcoded values that evalCli.__main__ sets after parse_args
    hardcoded = {
        'det': True,
        'env_name': 'plume',
        'env_dt': 0.04,
        'turnx': 1.0,
        'movex': 1.0,
        'birthx': 1.0,
        'loc_algo': 'quantile',
        'time_algo': 'uniform',
        'diff_max': 0.8,
        'diff_min': 0.8,
        'auto_movex': False,
        'auto_reward': False,
        'wind_rel': True,
        'action_feedback': False,
        'walking': False,
        'radiusx': 1.0,
        'r_shaping': ['step', 'oob'],
        'rewardx': 1.0,
        'squash_action': True,
        'diffusion_min': 1.0,   # set to diffusionx=1.0
        'diffusion_max': 1.0,
        'flipping': False,
        'odor_scaling': False,
        'qvar': 0.0,
        'stray_max': 2.0,
        'birthx_max': 1.0,
        'masking': None,
        'stride': 1,
        'act_noise': 0.0,
        'dynamic': False,
        'stacking': 0,
        'recurrent_policy': True,
    }

    defaults = vars(parser.parse_args([]))
    defaults.update(hardcoded)
    return defaults


# ---------------------------------------------------------------------------
# Locate and load the _args.json saved alongside the weights during training
# ---------------------------------------------------------------------------
def find_training_json(model_fname: str) -> str:
    f_prefix = os.path.basename(model_fname).replace('.pt', '')
    weights_dir = os.path.dirname(model_fname)
    exp_dir = os.path.dirname(weights_dir)   # one level up from weights/
    if '.chkpt' in f_prefix:
        f_prefix = f_prefix.split('.chkpt')[0]  # remove .chkpt.{epoch} if present
    json_path = os.path.join(exp_dir, 'json', f_prefix + '_args.json')
    return json_path


def load_training_config(json_path: str) -> dict:
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Training args JSON not found: {json_path}")
    with open(json_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Pretty comparison
# ---------------------------------------------------------------------------
def compare_configs(train_cfg: dict, eval_defaults: dict) -> None:
    train_keys = set(train_cfg.keys())
    eval_keys  = set(eval_defaults.keys())

    only_in_train  = sorted(train_keys - eval_keys)
    only_in_eval   = sorted(eval_keys  - train_keys)
    in_both        = sorted(train_keys & eval_keys)

    differing = [(k, eval_defaults[k], train_cfg[k])
                 for k in in_both if eval_defaults[k] != train_cfg[k]]
    matching  = [k for k in in_both if eval_defaults[k] == train_cfg[k]]

    COL_W = 36  # key column width
    VAL_W = 32  # value column width

    def row(k, a, b):
        return f"  {str(k):<{COL_W}}  {str(a):<{VAL_W}}  {str(b)}"

    sep = "-" * 100

    print("\n" + "=" * 100)
    print("  CONFIG COMPARISON:  evalCli defaults  vs  training _args.json")
    print("=" * 100)

    print(f"\n[1] Keys ONLY in training JSON ({len(only_in_train)}) — evalCli is missing these:")
    print(sep)
    print(f"  {'KEY':<{COL_W}}  {'TRAINING VALUE'}")
    print(sep)
    for k in only_in_train:
        print(f"  {str(k):<{COL_W}}  {train_cfg[k]}")

    print(f"\n[2] Keys ONLY in evalCli defaults ({len(only_in_eval)}) — not in training JSON:")
    print(sep)
    print(f"  {'KEY':<{COL_W}}  {'EVALCLI DEFAULT'}")
    print(sep)
    for k in only_in_eval:
        print(f"  {str(k):<{COL_W}}  {eval_defaults[k]}")

    print(f"\n[3] Keys present in BOTH but with DIFFERENT values ({len(differing)}):")
    print(sep)
    print(row('KEY', 'EVALCLI DEFAULT', 'TRAINING VALUE'))
    print(sep)
    for k, ev, tr in differing:
        print(row(k, ev, tr))

    print(f"\n[4] Keys present in BOTH with MATCHING values ({len(matching)}) — omitted for brevity.")
    print("    (Pass --show_matching to print them.)\n")


# ---------------------------------------------------------------------------
# Apply a subset of training config keys into eval args
# ---------------------------------------------------------------------------
def apply_configs(args: argparse.Namespace, cfg: dict, keys = []) -> argparse.Namespace:
    applied = []
    if len(keys) == 0:
        keys = cfg.keys()  # apply all by default
    for k in keys:
        if k in cfg:
            setattr(args, k, cfg[k])
            applied.append(f"  {k} = {cfg[k]}")
    if applied:
        print("\n[Inherited from config]")
        print("\n".join(applied))
    return args


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='evalCli_hydra — compare training config then eval')
    parser.add_argument('--model_fname', required=True,
                        help='Absolute path to .pt weight file')
    parser.add_argument('--show_matching', action='store_true', default=False,
                        help='Also print keys that match between evalCli and training config')
    # Eval arguments 
    parser.add_argument('--dataset', default='constantx5b5')
    parser.add_argument('--out_dir', type=str, default='eval')
    parser.add_argument('--test_episodes', type=int, default=100)
    parser.add_argument('--viz_episodes', type=int, default=2)
    parser.add_argument('--no_viz', type=bool, default=True)
    parser.add_argument('--fixed_eval', action='store_true', default=False)
    parser.add_argument('--test_sparsity', action='store_true', default=False)
    parser.add_argument('--only_test_sparsity', action='store_true', default=False)
    parser.add_argument('--no_vec_norm_stats', action='store_true', default=False)
    parser.add_argument('--dry_run', action='store_true', default=False)
    parser.add_argument('--mlflow', type=bool, default=False)
    parser.add_argument('--ou_eval', type=bool, default=True)
    parser.add_argument('--time_offsets', type=float, nargs='+', default=[0.0, 1.0])
    parser.add_argument('--device', default='cpu')
    # Eval experiments
    parser.add_argument('--flip_ventral_optic_flow', type=bool, default=False)
    parser.add_argument('--perturb_RNN_by_ortho_set', type=str, default=False)
    parser.add_argument('--perturb_RNN_by', type=str, default=False)
    args = parser.parse_args()
    args.f_prefix = os.path.basename(args.model_fname).replace(".pt", "") # eg: plume_seed_hash
    args.f_dir = os.path.dirname(args.model_fname) # f_dir should follow {/path/to/experiment}/weights
    args.exp_dir = os.path.dirname(args.f_dir) # {/path/to/experiment}
    args.abs_out_dir = '/'.join([args.exp_dir, args.out_dir, args.f_prefix]) # {/path/to/experiment}/{args.out_dir=eval}/plume_seed_hash/
    
    print(f"Output directory: {args.abs_out_dir}")
    if not args.dry_run:
        # make sure the directory exists
        os.makedirs('/'.join([args.exp_dir, args.out_dir]), exist_ok=True)
        os.makedirs(args.abs_out_dir, exist_ok=True)
    

    # Step 1: locate and load training config
    json_path = find_training_json(args.model_fname)
    print(f"Training config: {json_path}")
    train_cfg = load_training_config(json_path)

    # Step 2: build evalCli argparse defaults
    eval_defaults = build_evalcli_defaults()
    args = apply_configs(args, eval_defaults)

    # Step 3: compare
    compare_configs(train_cfg, eval_defaults)

    # Step. 4: inherit subset of args when relevant
    agent_setting = ['if_vec_norm', 'if_train_actor_std', 'rnn_type', 'variant']
    env_setting = ['apparent_wind', 'action_physics', 'apparent_wind_allo', 'wind_rel', 'squash_action', 'r_shaping', 'visual_feedback']
    # env_setting = ['apparent_wind', 'action_physics', 'apparent_wind_allo', 'wind_rel', 'squash_action', 'r_shaping', 'rotate_by', 'visual_feedback'] # just keep rotate_by off for now
    #'ou_eval' set to true
    args = apply_configs(args, train_cfg, keys=agent_setting + env_setting)
    print("\n" + "=" * 100)
    print("Proceeding to evaluation with args:")
    print(args)
    print("=" * 100 + "\n")
    if args.show_matching:
        in_both  = sorted(set(train_cfg) & set(eval_defaults))
        matching = [k for k in in_both if eval_defaults[k] == train_cfg[k]]
        print(f"\n[4] Matching keys ({len(matching)}):")
        print("-" * 60)
        for k in matching:
            print(f"  {str(k):<36}  {eval_defaults[k]}")
        print()

    if args.dry_run:
        print("Stopping after config comparison.  Pass --proceed to run the eval loop.")
        sys.exit(0)

    # check if args.device is available
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            args.device = 'cpu'
            print("CUDA is not available, switching to CPU.")
    
    # actor_critic, obs_rms, optimizer_state_dict = torch.load(args.model_fname, map_location=torch.device('cpu'))
    try:
        actor_critic, obs_rms, optimizer_state_dict = torch.load(args.model_fname, map_location=torch.device(args.device), weights_only=False)
    except ValueError:
        actor_critic, obs_rms = torch.load(args.model_fname, map_location=torch.device(args.device), weights_only=False)
    
    eval_loop(args, actor_critic, test_sparsity=args.test_sparsity)




if __name__ == '__main__':
    main()
