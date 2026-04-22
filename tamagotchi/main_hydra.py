"""Hydra entry point for dronmagotchi training.

Replaces the datajoint-based experiment manager. Loads conf/config.yaml,
converts it to the argparse.Namespace that main.main() expects, and runs it.

Usage:
    python3 -m tamagotchi.main_hydra env_name=plume outsuffix=run01
    python3 -m tamagotchi.main_hydra -m seed=1,2,3 action_physics=air_vel_angvel,ground_vel_angvel
"""
import argparse
import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from tamagotchi import main as base_main


def _auto_outsuffix(cfg_dict: dict) -> str:
    """Build a unique per-job outsuffix from Hydra override_dirname + seed."""
    override_dirname = HydraConfig.get().job.override_dirname  # e.g. "seed=1,action_physics=air_vel_angvel"
    sanitized = override_dirname.replace(",", "_").replace("=", "-").replace("/", "-")
    seed = cfg_dict.get("seed", "")
    if sanitized and f"seed-{seed}" not in sanitized and f"seed={seed}" not in sanitized:
        sanitized = f"{sanitized}_seed-{seed}"
    elif not sanitized:
        sanitized = f"seed-{seed}"
    return sanitized


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    # Hydra changes cwd; resolve save/log dirs relative to original cwd so reruns land in the same place.
    orig_cwd = hydra.utils.get_original_cwd()
    for k in ("save_dir", "log_dir"):
        v = cfg_dict.get(k)
        if v and not os.path.isabs(v):
            cfg_dict[k] = os.path.join(orig_cwd, v)

    if not cfg_dict.get("outsuffix"):
        cfg_dict["outsuffix"] = _auto_outsuffix(cfg_dict)
    
    print("Running with config:")
    print(OmegaConf.to_yaml(cfg))

    args = argparse.Namespace(**cfg_dict)
    import torch
    args.cuda = torch.cuda.is_available()
    base_main.main(args=vars(args))


if __name__ == "__main__":
    run()