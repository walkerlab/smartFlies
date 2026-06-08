"""Hydra entry point for dronmagotchi training.

Replaces the datajoint-based experiment manager. Loads conf/config.yaml,
converts it to the argparse.Namespace that main.main() expects, and runs it.

Usage:
    python3 -m tamagotchi.main_hydra env_name=plume outsuffix=run01
    python3 -m tamagotchi.main_hydra -m seed=1,2,3 action_physics=air_vel_angvel,ground_vel_angvel
    python3 -m tamagotchi.main_hydra -m seed=1,2,3 action_physics=force OU_exploration=medium_anneal experiment_name=force_physics r_shaping='r_shaping=[OU_locked_to_schedule,missed_time_cost,rotate_by,birthx_cl_last,cosine]'
"""
import argparse
import os
import hashlib
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
def adam_smoke(label):
    import torch
    import torch.optim as optim
    p = torch.nn.Parameter(torch.randn(64, 7))
    optim.Adam([p], lr=1e-3, eps=1e-5, weight_decay=0.0)


adam_smoke("") # something about the import order causes a weird interaction - if init here, bug disappears

from tamagotchi import main as base_main
from tamagotchi import main_wind_obsver_v1
from tamagotchi import main_wind_obsver_v2

def _auto_outsuffix(cfg_dict: dict) -> str:
    """Build a unique per-job outsuffix: seed-{seed}-{hash}, hash over the full override string."""
    override_dirname = HydraConfig.get().job.override_dirname  # e.g. "seed=1,action_physics=air_vel_angvel"
    # take out seed if present
    if "seed" in override_dirname:
        override_dirname = ",".join([part for part in override_dirname.split(",") if not part.startswith("seed=")])
    seed = cfg_dict.get("seed", "")
    h = hashlib.sha1(override_dirname.encode("utf-8")).hexdigest()[:8]
    return f"seed-{seed}-{h}"


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    # Hydra changes cwd; resolve save/log dirs relative to original cwd so reruns land in the same place.
    orig_cwd = hydra.utils.get_original_cwd()
    for k in ("save_dir", "log_dir"):
        v = cfg_dict.get(k)
        if v and not os.path.isabs(v):
            cfg_dict[k] = os.path.join(orig_cwd, v)

    # flatten path subconfig into top level
    if isinstance(cfg_dict.get("path"), dict):
        cfg_dict.update(cfg_dict.pop("path"))

    # Build an outsuffix
    if cfg_dict.get("stage_name") and not cfg_dict.get("outsuffix"):
        raise ValueError(
            "staged training (stage_name is set) requires an explicit outsuffix= override. "
            "Auto-generation would bake stage_name/resume_from into the hash, creating a "
            "new outsuffix that won't match the original wandb run."
        )
    if not cfg_dict.get("outsuffix"):
        cfg_dict["outsuffix"] = _auto_outsuffix(cfg_dict)

    print("Running with config:")
    print(OmegaConf.to_yaml(cfg))

    args = argparse.Namespace(**cfg_dict)
    import torch
    args.cuda = torch.cuda.is_available()

    # Dispatch to the appropriate main function based on variant
    variant = cfg_dict.get("variant", "base")
    if variant == "wind_obsver_v1":
        main_wind_obsver_v1.main(args=vars(args))
    elif variant == "wind_obsver_v2":
        main_wind_obsver_v2.main(args=vars(args))
    elif variant == "base":
        base_main.main(args=vars(args))
    else:
        raise ValueError(f"Unknown variant: {variant}. Must be 'base', 'wind_obsver_v1', or 'wind_obsver_v2'")


if __name__ == "__main__":
    run()