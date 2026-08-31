"""Thin wandb wrapper used for all experiment logging.

Import as `from tamagotchi import wb` and call `wb.set_experiment`,
`wb.start_run`, `wb.log_metric` / `log_params` / `log_artifact` /
`log_summary`, and `wb.flush` to commit a step.
"""
from contextlib import contextmanager
import os
import pandas as pd
import wandb
from PIL import Image

_project = None
_experiment = None
_run = None


def set_experiment(experiment_name):
    global _experiment
    _experiment = experiment_name


def search_runs(filter_string=""):
    name = None
    if "run_name" in filter_string:
        name = filter_string.split("'")[1]
    try:
        api = wandb.Api()
        project = _experiment or os.environ.get("WANDB_PROJECT", "dronmagotchi")
        entity = os.environ.get("WANDB_ENTITY")
        path = f"{entity}/{project}" if entity else project
        runs = api.runs(path, filters={"displayName": {"$eq": name}} if name else None)
        rows = [{"run_id": r.id, "run_name": r.name} for r in runs]
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


@contextmanager
def start_run(run_name=None, run_id=None, dir=None, **_):
    global _run
    _run = wandb.init(
        project=_experiment or os.environ.get("WANDB_PROJECT", "dronmagotchi"),
        name=run_name,
        id=run_id,
        resume="allow" if run_id else None,
        reinit=True,
        dir=dir,  # co-locate wandb/ folder with experiment outputs
    )
    try:
        yield _run
    finally:
        wandb.finish()
        _run = None


def log_params(params):
    if wandb.run is not None:
        wandb.config.update({k: v for k, v in params.items() if _jsonable(v)}, allow_val_change=True)


def log_metric(key, value, step=None):
    if wandb.run is not None:
        wandb.log({key: value}, step=step, commit=False)


def flush(step=None):
    if wandb.run is not None:
        wandb.log({}, step=step, commit=True)


def log_artifact(local_path, artifact_path=None, step=None):
    if wandb.run is None or not os.path.exists(local_path):
        return
    base = os.path.basename(local_path)
    is_image = local_path.endswith(('.png', '.jpg', '.jpeg'))
    if is_image:
        panel_key = artifact_path if artifact_path else base
        # Log images through a single explicit stepped path to avoid W&B step drift.
        with Image.open(local_path) as image:
            wandb.log({panel_key: wandb.Image(image.copy(), caption=base)}, step=step, commit=False)
        return

    wandb.save(local_path, base_path=os.path.dirname(local_path) or ".", policy="live")


def log_summary(key, value):
    if wandb.run is not None:
        wandb.run.summary[key] = value


def _jsonable(v):
    try:
        import json
        json.dumps(v)
        return True
    except Exception:
        return False
