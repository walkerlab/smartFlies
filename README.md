# tamagotchi

Deep-RL training for plume-tracking agents. A recurrent PPO agent learns to
track an odor plume to its source in a simulated wind tunnel, from wind,
odor, orientation, and course-direction observations. Optional Gaussian
wind-observer heads let the network additionally predict the ambient wind
direction as an auxiliary task.

## Layout

```
tamagotchi/
  main_hydra.py    Hydra entry point (conf/config.yaml)
  main.py          builds args, envs, policy, PPO; resolves checkpoints and the wandb run
  training.py      PPO training loop: rollouts, curriculum, checkpoints, logging
  env.py           PlumeEnvironment_v3 (gym.Env) + vectorized env plumbing
  data_util.py     plume-data loading, curriculum schedules, wandb metric helpers
  sim_cli.py       generates the plume simulation datasets training reads
  sim_utils.py     plume simulation internals
  wb.py            thin wandb wrapper used by all logging call sites
  config.py        per-host data directory resolution
  a2c_ppo_acktr/   Policy/MLPBase model, PPO update, rollout storage
  conf/            Hydra configs: OU_exploration/, path/, physics/, curriculum/
  evalCli.py       evaluation engine: assay rollouts -> eps logs (.pkl) + summary (.csv)
  evalCli_hydra.py eval entry point: inherits the training run's _args.json, calls eval_loop
  eval/            post-hoc analysis: traj/activity DataFrames, plots, videos, eigen analysis
scripts/
  gen_cl_sweep_diff.py  generates curriculum-config families sweeping the diff_min/diff_max ladder
  slurm-run_ckpt.py     submits training jobs (auto-requeue checkpointing, seed sweeps)
  slurm-eval_ckpt.py    submits one eval job per checkpoint in an experiment folder
```

## Training

```bash
# single run
python3 -m tamagotchi.main_hydra experiment_name=my_exp outsuffix=run01

# multirun sweep over seeds
python3 -m tamagotchi.main_hydra -m seed=1,2,3 action_physics=force \
    OU_exploration=medium_anneal experiment_name=force_physics

# quick smoke test (tiny run, wandb off by default in this config)
WANDB_MODE=offline python3 -m tamagotchi.main_hydra --config-name config_debug
```

Key config groups (see `conf/config.yaml` for the full surface):

- `auxiliary_arch` — wind-observer architecture. `none` (default) is a plain
  actor-critic; `default`, `separate_wind_head`, `wind_cond_policy`, and
  `wind_cond_policy_detached` add wind-prediction heads trained with an NLL
  auxiliary loss weighted by `wind_loss_coef`.
- `action_physics` — `kinematics` (forward air speed + angular velocity, wind
  drifts the agent) or `force` (body-frame thrust + yaw torque). Both read the
  `physics=` group from `conf/physics/`: `force` requires its coefficients, and
  `kinematics` uses its speed/yaw-rate limits and optional acceleration caps.
- `OU_exploration` — Ornstein-Uhlenbeck action-noise presets.
- `path` — per-cluster data/save directories; `path.curriculum_name` selects a
  JSON curriculum schedule from `conf/curriculum/`.
- `wandb` — set `wandb=0` to disable experiment logging.

Runs write checkpoints (`weights/*.chkpt.pt`, rolling), training CSVs
(`train_logs/`), and the resolved args (`json/*_args.json`) under
`path.save_dir`. Re-running the same command resumes from the rolling
checkpoint. Staged training continues from a parent run via
`stage_name=... resume_from=... outsuffix=...`.

## Evaluation

```bash
# evaluate one checkpoint; config is inherited from the _args.json saved next to the weights
python3 -m tamagotchi.evalCli_hydra --model_fname <save_dir>/weights/<ckpt>.pt --dataset constantx5b5

# fan out over every checkpoint in an experiment folder on SLURM
python3 scripts/slurm-eval_ckpt.py --from_folder <save_dir> --dataset constantx5b5
```

`evalCli_hydra` diffs the training config against the eval defaults and refuses to
run without `--proceed` when they disagree. Results land in `<out_dir>/` as
per-episode logs (`{dataset}.pkl`) and `{dataset}_summary.csv`; post-hoc tools in
`tamagotchi/eval/` (trajectory videos via `postEvalCli.py`, success-rate tables via
`tabulate_from_summary.py`, mp4 stitching via `vid_stitch_cli.py`) consume those.

## Data

Training reads pickled plume/wind simulations (`puff_data_<dataset>.pickle`,
`wind_data_<dataset>.pickle`) from the data directory resolved by
`tamagotchi/config.py` for the current host. Generate datasets with
`python3 -m tamagotchi.sim_cli --help`.

## Provenance

Extracted and refactored from
[walkerlab/smartFlies](https://github.com/walkerlab/smartFlies), which also
contains the evaluation/analysis code and the supplement of
[Singh et al., Nature Machine Intelligence 2023](https://www.nature.com/articles/s42256-022-00599-w)
that this codebase descends from.
