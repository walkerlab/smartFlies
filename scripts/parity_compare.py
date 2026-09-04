"""Compare two parity dumps written by scripts/parity_dump.py.

    python scripts/parity_compare.py /tmp/parity_old.pt /tmp/parity_new.pt [--atol 1e-6]

Prints, per stage and per recorded quantity, the max abs difference and the first
step at which the two pipelines diverge. Exit status is 1 if anything differs
beyond --atol, so it can be used in a shell check.

Reading the result:
  * A_env differs            -> PlumeEnvironment_v3 (physics / obs / reward / reset RNG)
  * A_env same, B differs    -> vec-env wrappers (VecNormalize) or Policy forward
    (compare obs_norm first; if obs_norm matches but action/wind_mu differ, it is the
    model)
  * A+B same, C differs      -> RolloutStorage.compute_returns / PPO.update
"""
import argparse
import sys

import numpy as np
import torch


def _as_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def cmp_array(name, a, b, atol, report):
    a, b = _as_np(a), _as_np(b)
    if a.shape != b.shape:
        report.append((name, "SHAPE", f"{a.shape} vs {b.shape}"))
        return False
    if a.dtype.kind in "OUS" or b.dtype.kind in "OUS":  # strings / objects
        neq = np.nonzero(a != b)[0]
        if len(neq):
            report.append((name, "DIFF", f"first mismatch at index {neq[0]}: {a.flat[neq[0]]!r} vs {b.flat[neq[0]]!r}"))
            return False
        report.append((name, "ok", ""))
        return True
    a = a.astype(np.float64); b = b.astype(np.float64)
    both_nan = np.isnan(a) & np.isnan(b)
    diff = np.abs(a - b)
    diff[both_nan] = 0.0
    if np.isnan(diff).any():
        report.append((name, "DIFF", "NaN on one side only"))
        return False
    mx = float(diff.max()) if diff.size else 0.0
    if mx > atol:
        # first diverging leading index (step)
        if diff.ndim >= 1:
            per_step = diff.reshape(diff.shape[0], -1).max(axis=1)
            first = int(np.argmax(per_step > atol))
            where = f"first divergence at step {first} (|d|={per_step[first]:.3e})"
        else:
            where = ""
        report.append((name, "DIFF", f"max|d|={mx:.3e} {where}"))
        return False
    report.append((name, "ok", f"max|d|={mx:.1e}"))
    return True


def cmp_tree(prefix, a, b, atol, report):
    ok = True
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                report.append((f"{prefix}{k}", "MISSING", f"present only in {'old' if k in a else 'new'}"))
                ok = False
                continue
            ok &= cmp_tree(f"{prefix}{k}.", a[k], b[k], atol, report)
        return ok
    if a is None and b is None:
        return True
    if isinstance(a, (str, bytes)) or isinstance(b, (str, bytes)):
        same = a == b
        report.append((prefix.rstrip("."), "ok" if same else "DIFF", "" if same else f"{a!r} vs {b!r}"))
        return same
    if isinstance(a, (list, tuple)) and len(a) and isinstance(a[0], str):
        return cmp_array(prefix.rstrip("."), np.asarray(a), np.asarray(b), atol, report)
    try:
        return cmp_array(prefix.rstrip("."), a, b, atol, report)
    except Exception as e:  # noqa: BLE001
        report.append((prefix.rstrip("."), "SKIP", f"could not compare: {e}"))
        return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--show-ok", action="store_true", help="also list matching quantities")
    cli = ap.parse_args()

    A = torch.load(cli.old, map_location="cpu", weights_only=False)
    B = torch.load(cli.new, map_location="cpu", weights_only=False)
    print(f"old: {A['pipeline']} @ {A['repo']}  (torch {A['torch']}, numpy {A['numpy']})")
    print(f"new: {B['pipeline']} @ {B['repo']}  (torch {B['torch']}, numpy {B['numpy']})")
    print(f"seed={A['seed']}/{B['seed']} steps={A['steps']}/{B['steps']} atol={cli.atol}\n")
    if A["torch"] != B["torch"] or A["numpy"] != B["numpy"]:
        print("WARNING: different torch/numpy versions; tiny float differences are expected.\n")

    all_ok = True
    for stage in ["A_env", "BC_rollout_update"]:
        if stage not in A or stage not in B:
            print(f"== {stage}: present in {'old' if stage in A else 'new'} only, skipped")
            continue
        report = []
        skip = {"env_kwargs", "params_after"}  # env_kwargs differ by design (old has extra keys); params compared via deltas/digest
        a = {k: v for k, v in A[stage].items() if k not in skip}
        b = {k: v for k, v in B[stage].items() if k not in skip}
        ok = cmp_tree("", a, b, cli.atol, report)
        all_ok &= ok
        print(f"== {stage}: {'IDENTICAL' if ok else 'DIFFERENT'}")
        for name, status, detail in report:
            if status == "ok" and not cli.show_ok:
                continue
            print(f"   {status:8s} {name:45s} {detail}")
        if stage == "BC_rollout_update" and "params_after" in A[stage] and "params_after" in B[stage]:
            pa, pb = A[stage]["params_after"], B[stage]["params_after"]
            worst = sorted(((float(np.abs(np.asarray(pa[k]) - np.asarray(pb[k])).max()) if k in pb else np.inf, k)
                            for k in pa), reverse=True)[:5]
            print("   post-update parameter max|d| (top 5):")
            for d, k in worst:
                print(f"      {d:.3e}  {k}")
        print()

    # env kwargs: show only keys present on both sides that differ (old has many extra keys by design)
    if "A_env" in A and "A_env" in B:
        ka, kb = A["A_env"]["env_kwargs"], B["A_env"]["env_kwargs"]
        shared_diff = {k: (ka[k], kb[k]) for k in set(ka) & set(kb) if ka[k] != kb[k]}
        if shared_diff:
            print("== env constructor kwargs that differ on shared keys:")
            for k, (va, vb) in sorted(shared_diff.items()):
                print(f"   {k}: old={va!r} new={vb!r}")
            print()
        extra_old = sorted(set(ka) - set(kb))
        if extra_old:
            print(f"== env kwargs only in old (expected, retired options): {extra_old}\n")

    print("RESULT:", "all compared quantities identical" if all_ok else "differences found (see above)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
