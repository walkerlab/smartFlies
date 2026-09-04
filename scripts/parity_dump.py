"""Dump a deterministic 100-step rollout + one PPO update for pipeline parity checks.

The smartFlies (variant=wind_obsver_v1) and tamagotchi pipelines both ship a package
named ``tamagotchi``, so they cannot be imported into one process. Run this script
once from each repo root and compare the two dumps offline:

    # old pipeline
    cd /path/to/smartFlies  && python scripts/parity_dump.py --out /tmp/parity_old.pt
    # new pipeline
    cd /path/to/tamagotchi  && python scripts/parity_dump.py --out /tmp/parity_new.pt
    # compare
    python scripts/parity_compare.py /tmp/parity_old.pt /tmp/parity_new.pt

The script auto-detects which pipeline it is running inside and records three
independent stages, each seeded identically on both sides:

  A. env only      - one raw PlumeEnvironment_v3 (built through the pipeline's own
                     make_env), driven by a fixed pre-generated action sequence.
                     Records obs / reward / done / location / wind / odor per step.
  B. rollout       - the training stack (make_vec_envs -> VecNormalize -> Policy)
                     with num_processes=1, num_steps=N; the policy chooses actions.
                     Records normalized obs, actions, values, log-probs, the wind
                     observer's wind_mu / wind_logvar and the wind targets.
  C. one PPO update - compute_returns + agent.update on that rollout. Records the
                     losses, the wind-loss extras, and per-parameter deltas.

Identical inputs + identical code => bitwise-identical dumps (CPU, 1 thread). The
first stage that diverges points at the module that differs.

Randomness alignment across the two pipelines (see conf/deprecated.yaml):
  * the old env draws one extra np.random.uniform in __init__ -> we reseed numpy
    right before the first reset() on both sides;
  * the old env draws odorx in reset() when odor_scaling=True -> forced off here
    (it has no effect under odor_01=True anyway).
"""
import argparse
import copy
import hashlib
import importlib.util
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # import the package of THIS checkout, not a pip-installed one

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

IS_OLD = importlib.util.find_spec("tamagotchi.main_wind_obsver_v1") is not None
PIPELINE = "smartFlies/wind_obsver_v1" if IS_OLD else "tamagotchi"

if IS_OLD:
    from tamagotchi.a2c_ppo_acktr.model_wind_obsver_v1 import Policy
    from tamagotchi.a2c_ppo_acktr.ppo_wind_obsver_v1 import PPO
    from tamagotchi.a2c_ppo_acktr.storage_wind_obsver_v1 import RolloutStorage
else:
    from tamagotchi.a2c_ppo_acktr.model import Policy
    from tamagotchi.a2c_ppo_acktr.ppo import PPO
    from tamagotchi.a2c_ppo_acktr.storage import RolloutStorage
from tamagotchi.env import make_env, make_vec_envs

# Matches the CL_fly_coeff / refactor_test launch (env_dt=0.1, stray_max=2, ...);
# num_processes/num_mini_batch are shrunk so a single in-process env can run PPO.
BASE_OVERRIDES = [
    "action_physics=force",
    "r_shaping=[step,missed_time_cost,rotate_by,birthx_cl_last,cosine]",
    "loc_algo=slice_linear",
    "dataset=[constant_jitterx5b5]",
    "qvar=1.5",
    "birthx=0.8",
    "birthx_upper=0.1",
    "odor_01=true",
    "env_dt=0.1",
    "stray_max=2",
    "experiment_name=parity_check",
    "num_processes=1",
    "num_mini_batch=1",
]
OLD_ONLY = ["variant=wind_obsver_v1", "odor_scaling=false"]
NEW_ONLY = ["auxiliary_arch=separate_wind_head"]


def build_args(seed, num_steps, extra_overrides):
    """Compose conf/config.yaml the way main_hydra does and return an argparse.Namespace."""
    overrides = BASE_OVERRIDES + (OLD_ONLY if IS_OLD else NEW_ONLY) + [
        f"seed={seed}",
        f"num_steps={num_steps}",
        f"path.base_dir={REPO_ROOT}",
    ] + list(extra_overrides)
    with initialize_config_dir(config_dir=os.path.join(REPO_ROOT, "tamagotchi", "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    d = OmegaConf.to_container(cfg, resolve=True)
    # --- same transforms as main_hydra.run() ---
    if isinstance(d.get("physics"), dict):
        d["force_physics"] = d.pop("physics")
    if isinstance(d.get("path"), dict):
        d.update(d.pop("path"))
    d["outsuffix"] = "parity"
    args = argparse.Namespace(**d)
    # --- same preprocessing as main() before the envs are built ---
    args.cuda = False
    args.device = torch.device("cpu")
    args.rotate_by = None if not args.rotate_by else args.rotate_by
    args.model_fname = f"{args.env_name}_{args.outsuffix}.pt"
    return args, overrides


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def state_digest(module):
    h = hashlib.sha1()
    for k, v in module.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def flat_params(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


# ----------------------------------------------------------------------------- stage A
def stage_env_only(args, seed, num_steps):
    a = copy.deepcopy(args)
    # make_env expects the per-env (scalar) curriculum values that make_vec_envs would set
    a.dataset = args.dataset[0]
    a.qvar = args.qvar if not isinstance(args.qvar, (list, tuple)) else args.qvar[0]
    a.diff_max = args.diff_max[0]
    a.diff_min = args.diff_min[0]
    a.reset_offset_tmax = 30
    a.t_val_min = 60
    env = make_env(a.env_name, seed, 0, None, True, a)()  # log_dir=None: no Monitor wrapper

    rs = np.random.RandomState(20240904)
    actions = rs.randn(num_steps, env.action_space.shape[0]).astype(np.float32)  # pre-squash, like policy output

    np.random.seed(seed)  # align the env's numpy stream right before the first reset
    obs0 = env.reset()
    rec = {k: [] for k in ["obs", "reward", "done", "location", "angle", "ambient_wind",
                           "air_velocity", "stray_distance", "odor_obs", "r_radial_step", "done_reason"]}
    rec["obs0"] = np.asarray(obs0, dtype=np.float64)
    rec["init_location"] = np.asarray(env.agent_location, dtype=np.float64)
    rec["init_angle"] = np.asarray(env.agent_angle, dtype=np.float64)
    rec["rotate_by"] = env.rotate_by
    rec["step_offset"] = env.step_offset
    rec["puff_density"] = env.puff_density
    n_resets = 0
    for t in range(num_steps):
        obs, reward, done, info = env.step(actions[t])
        rec["obs"].append(np.asarray(obs, dtype=np.float64))
        rec["reward"].append(float(reward))
        rec["done"].append(bool(done))
        rec["location"].append(np.asarray(info["location"], dtype=np.float64))
        rec["angle"].append(np.asarray(info["angle"], dtype=np.float64))
        rec["ambient_wind"].append(np.asarray(info["ambient_wind"], dtype=np.float64))
        rec["air_velocity"].append(np.asarray(info["air_velcity"], dtype=np.float64))
        rec["stray_distance"].append(float(info["stray_distance"]))
        rec["odor_obs"].append(float(info["odor_obs"]))
        rec["r_radial_step"].append(float(info["r_radial_step"]))
        rec["done_reason"].append(str(info["done"]))
        if done:
            n_resets += 1
            env.reset()
    out = {k: (np.asarray(v) if isinstance(v, list) and k != "done_reason" else v) for k, v in rec.items()}
    out["actions_pre_squash"] = actions
    out["n_resets"] = n_resets
    out["action_space_shape"] = tuple(env.action_space.shape)
    out["obs_space_shape"] = tuple(env.observation_space.shape)
    out["rewards_table"] = dict(env.rewards)
    out["env_kwargs"] = {k: (v if isinstance(v, (int, float, str, bool, list, tuple, type(None))) else str(v))
                         for k, v in env.arguments.items() if k != "self"}
    env.close()
    return out


# ----------------------------------------------------------------------------- stage B + C
def stage_rollout_and_update(args, seed, num_steps):
    curriculum_vars = {
        "dataset": args.dataset,
        "qvar": args.qvar,
        "diff_max": args.diff_max,
        "diff_min": args.diff_min,
        "reset_offset_tmax": [30, 30, 30, 30],
        "t_val_min": [60, 60, 60, 60],
    }
    seed_everything(seed)
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes, args.gamma, None,
                         args.device, True, args=args, **curriculum_vars)
    if not args.if_vec_norm:
        envs.venv.norm_obs = False

    torch.manual_seed(seed)  # identical initial weights on both sides
    actor_critic = Policy(
        envs.observation_space.shape, envs.action_space,
        base_kwargs={"recurrent": args.recurrent_policy, "rnn_type": args.rnn_type,
                     "hidden_size": args.hidden_size,
                     "auxiliary_arch": getattr(args, "auxiliary_arch", "separate_wind_head")},
        args=args)
    if hasattr(actor_critic, "configure_ou"):
        actor_critic.configure_ou(args)  # old pipeline; no-op RNG-wise, mirrors main()
    actor_critic.to(args.device)
    init_digest = state_digest(actor_critic)

    wind_loss_coef = getattr(args, "wind_loss_coef", 1e-2)
    agent = PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                args.value_loss_coef, args.entropy_coef, lr=args.lr, eps=args.eps,
                max_grad_norm=args.max_grad_norm, weight_decay=args.weight_decay,
                track_ppo_fraction=True, wind_loss_coef=wind_loss_coef)
    rollouts = RolloutStorage(args.num_steps, args.num_processes, envs.observation_space.shape,
                              envs.action_space, actor_critic.recurrent_hidden_state_size)

    np.random.seed(seed)  # align the env's numpy stream right before the first reset
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(args.device)

    act_dim = envs.action_space.shape[0]
    rec = {k: [] for k in ["obs_norm", "action", "value", "action_log_prob", "reward_norm", "done",
                           "wind_mu", "wind_logvar", "wind_target", "ambient_wind", "location",
                           "wind_obs_raw", "odor_obs_raw", "reward_raw"]}
    for step in range(args.num_steps):
        with torch.no_grad():
            out = actor_critic.act(rollouts.obs[step], rollouts.recurrent_hidden_states[step], rollouts.masks[step])
        value, action, action_log_prob, recurrent_hidden_states, activities = out[:5]
        obs, reward, done, infos = envs.step(action)
        wind_vels = np.asarray([info["ambient_wind"] for info in infos], dtype=np.float64)
        wind_dirs = wind_vels / (np.linalg.norm(wind_vels, axis=1, keepdims=True) + 1e-8)
        wind_dirs = torch.tensor(wind_dirs, dtype=torch.float32, device=args.device)
        masks = torch.FloatTensor([[0.0] if d else [1.0] for d in done])
        bad_masks = torch.FloatTensor([[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos])
        if step == args.num_steps - 1:
            obs = envs.reset()  # training_loop does this on the last step
        kw = {"wind_targets": wind_dirs}
        if IS_OLD:
            kw["ou_state"] = torch.zeros(args.num_processes, act_dim)
        rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks, **kw)

        rec["obs_norm"].append(rollouts.obs[step].numpy().copy())
        rec["action"].append(action.numpy().copy())
        rec["value"].append(value.numpy().copy())
        rec["action_log_prob"].append(action_log_prob.numpy().copy())
        rec["reward_norm"].append(reward.numpy().copy())
        rec["done"].append(np.asarray(done, dtype=bool))
        rec["wind_mu"].append(activities["wind_mu"].numpy().copy() if activities.get("wind_mu") is not None else np.full((args.num_processes, 2), np.nan))
        rec["wind_logvar"].append(activities["wind_logvar"].numpy().copy() if activities.get("wind_logvar") is not None else np.full((args.num_processes, 2), np.nan))
        rec["wind_target"].append(wind_dirs.numpy().copy())
        rec["ambient_wind"].append(wind_vels)
        rec["location"].append(np.asarray([info["location"] for info in infos], dtype=np.float64))
        rec["wind_obs_raw"].append(np.asarray([info["wind_obs"] for info in infos], dtype=np.float64))
        rec["odor_obs_raw"].append(np.asarray([info["odor_obs"] for info in infos], dtype=np.float64))
        rec["reward_raw"].append(np.asarray([info["reward"] for info in infos], dtype=np.float64))

    # --- one update, exactly as training_loop ---
    envs.step(action)  # training_loop steps once more after the rollout
    with torch.no_grad():
        next_value = actor_critic.get_value(rollouts.obs[-1], rollouts.recurrent_hidden_states[-1], rollouts.masks[-1]).detach()
    rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)
    returns = rollouts.returns.clone()
    params_before = flat_params(actor_critic)
    value_loss, action_loss, dist_entropy, clip_fraction, advantages, extras = agent.update(rollouts)
    params_after = flat_params(actor_critic)

    vn = envs.venv  # VecNormalize
    out = {k: np.asarray(v) for k, v in rec.items()}
    out.update({
        "init_param_digest": init_digest,
        "n_params": int(sum(p.numel() for p in actor_critic.parameters())),
        "param_names": list(params_before.keys()),
        "obs_rms_mean": np.asarray(vn.obs_rms.mean), "obs_rms_var": np.asarray(vn.obs_rms.var),
        "ret_rms_mean": np.asarray(vn.ret_rms.mean), "ret_rms_var": np.asarray(vn.ret_rms.var),
        "next_value": next_value.numpy(), "returns": returns.numpy(),
        "update": {"value_loss": float(value_loss), "action_loss": float(action_loss),
                   "dist_entropy": float(dist_entropy), "clip_fraction": float(clip_fraction),
                   "advantages": advantages.numpy(),
                   "wind_loss_epoch": float(extras["wind_loss_epoch"]) if extras else None,
                   "wind_nll_all": extras["wind_nll_all"].numpy() if extras else None,
                   "wind_sqerr_all": extras["wind_sqerr_all"].numpy() if extras else None,
                   "wind_logvar_all": extras["wind_logvar_all"].numpy() if extras else None},
        "param_delta_norm": {k: float((params_after[k] - params_before[k]).norm()) for k in params_before},
        "params_after": {k: v.numpy() for k, v in params_after.items()},
        "post_update_digest": state_digest(actor_critic),
        "wind_loss_coef": wind_loss_coef,
    })
    envs.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output .pt file (torch.save of a dict)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--override", action="append", default=[], help="extra hydra override(s), e.g. --override stray_max=4")
    ap.add_argument("--skip-update", action="store_true", help="only run stage A (env only)")
    cli = ap.parse_args()

    args, overrides = build_args(cli.seed, cli.steps, cli.override)
    print(f"[parity] pipeline={PIPELINE} repo={REPO_ROOT}")
    print(f"[parity] overrides={overrides}")
    print(f"[parity] torch={torch.__version__} numpy={np.__version__} python={sys.version.split()[0]}")

    dump = {"pipeline": PIPELINE, "repo": REPO_ROOT, "seed": cli.seed, "steps": cli.steps,
            "overrides": overrides, "torch": torch.__version__, "numpy": np.__version__}
    seed_everything(cli.seed)
    dump["A_env"] = stage_env_only(args, cli.seed, cli.steps)
    print(f"[parity] stage A done: {dump['A_env']['n_resets']} resets, "
          f"reward sum {dump['A_env']['reward'].sum():.4f}, done reasons {sorted(set(dump['A_env']['done_reason']))}")
    if not cli.skip_update:
        dump["BC_rollout_update"] = stage_rollout_and_update(args, cli.seed, cli.steps)
        u = dump["BC_rollout_update"]["update"]
        print(f"[parity] stage B/C done: value_loss={u['value_loss']:.6f} action_loss={u['action_loss']:.6f} "
              f"entropy={u['dist_entropy']:.6f} clip={u['clip_fraction']:.4f} wind_loss={u['wind_loss_epoch']}")
        print(f"[parity] init digest {dump['BC_rollout_update']['init_param_digest'][:12]} -> "
              f"post-update digest {dump['BC_rollout_update']['post_update_digest'][:12]}")
    torch.save(dump, cli.out)
    print(f"[parity] wrote {cli.out}")


if __name__ == "__main__":
    main()
