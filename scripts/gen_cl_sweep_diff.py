#!/usr/bin/env python3
"""Generate a family of curriculum configs that sweep the diff_min/diff_max ladder.

Replaces the old ``tamagotchi/conf/curriculum/CL_sweep.ipynb`` flow, which swept
``now_init`` levels. The ``linear`` location algorithm is now driven by
``diff_min``/``diff_max``: the pair brackets the plume-density quantile that agents
are initialized at, so the two walk together (a fixed-width window sliding from easy
to hard) rather than a single scalar growing.

What is swept here is the *number of lessons* the window takes to walk across its
range: ``diff_min`` from ``--diff_min_range`` (default 0.1 -> 0.5) and ``diff_max``
from ``--diff_max_range`` (default 0.2 -> 0.6), endpoint-inclusive, evenly spaced.
Both endpoints stay inside the allowed 0.1~0.6 band. One config is written per step
count, all sharing a filename prefix so ``slurm-run_ckpt.py --sweep_config <prefix>``
picks the whole family up and submits one job per config.

Everything that is not the diff ladder (``birthx``, ``wind_cond``, ``rotate_by``,
``meta``) is copied verbatim from a base config, so a sweep differs from the base in
exactly one dimension. The ladder is restarted from scratch inside each ``rotate_by``
stage, matching how ``build_tc_schedule_dict`` lays out ``linear`` lessons.

Examples
--------
# 6 configs off the const-jitter base, 2..8 lessons per rotate_by stage
python3 scripts/gen_cl_sweep_diff.py --base 081626_const_jitter --steps 2 3 4 5 6 8

# preview only, nothing written
python3 scripts/gen_cl_sweep_diff.py --base 081626_const_jitter --steps 2 4 8 --dry_run

# then, on hyak:
python3 scripts/slurm-run_ckpt.py --n_seeds 5 --sweep_config sweep_diffsteps \
    --sweep_outsuffix --override "... loc_algo=linear 'dataset=[constant_jitterx5b5]'"
"""

import argparse
import glob
import json
import os
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURRICULUM_DIR = os.path.join(REPO_DIR, 'tamagotchi', 'conf', 'curriculum')

# Hard bounds on the diff quantile thresholds, per the location algorithm.
DIFF_FLOOR, DIFF_CEIL = 0.1, 0.6

GLOBAL_KEYS = ('birthx', 'wind_cond')
# Lesson suffixes that belong to the loc_algo and are therefore *replaced* by the
# sweep rather than carried over from the base config.
LOC_SUFFIXES = ('_diff_min', '_diff_max', '_now_init_long')
# Everything data_util.validate_tc_schedule_datasets accepts as a per-dataset lesson.
PER_DS_SUFFIXES = ('_diff_max', '_diff_min', '_now_init_long', '_rotate_by')

ROUND_TO = 6  # match the precision of the hand-authored configs


def resolve_base_path(base):
    """Accept a curriculum name, a bare filename, or a path."""
    candidates = [base, f'{base}.json',
                  os.path.join(CURRICULUM_DIR, base),
                  os.path.join(CURRICULUM_DIR, f'{base}.json')]
    for c in candidates:
        if os.path.isfile(c):
            return c
    sys.exit(f'Error: base curriculum "{base}" not found. Tried:\n  ' + '\n  '.join(candidates))


def infer_dataset(base_schedule, override=None):
    """Pull the dataset name out of the base config's per-dataset lesson keys."""
    if override:
        return override
    names = set()
    for k in base_schedule:
        if k in GLOBAL_KEYS or k == 'meta':
            continue
        suffix = next((s for s in PER_DS_SUFFIXES if k.endswith(s)), None)
        if suffix:
            names.add(k[:-len(suffix)])
    if not names:
        sys.exit('Error: base config has no per-dataset lesson keys; pass --dataset explicitly.')
    if len(names) > 1:
        sys.exit(f'Error: base config spans multiple datasets {sorted(names)}; '
                 'pass --dataset to pick one.')
    return names.pop()


def stage_layout(base_schedule, dataset, total_num_updates):
    """Return [(stage_start, stage_duration), ...] for the diff ladder to repeat in.

    Stages are the ``rotate_by`` lessons, matching build_tc_schedule_dict's
    rotate_stage_times. Each stage runs until the next one starts; the last runs until
    the first ``birthx`` lesson (the birthx tail owns the rest of the run) or, absent
    that, to the end of the run.
    """
    rotate_key = f'{dataset}_rotate_by'
    starts = sorted(int(t) for t in base_schedule.get(rotate_key, {}))
    if not starts:
        # No rotate_by staging: one ladder across the whole run.
        return [(0, total_num_updates)]

    birthx_times = sorted(int(t) for t in base_schedule.get('birthx', {}))
    tail_end = next((t for t in birthx_times if t > starts[-1]), total_num_updates)

    layout = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else tail_end
        if end <= start:
            sys.exit(f'Error: stage starting at {start} has non-positive duration '
                     f'(next boundary {end}); check the base config.')
        layout.append((start, end - start))
    return layout


def linspace(start, end, n):
    """Endpoint-inclusive even spacing (np.linspace without the numpy import)."""
    if n == 1:
        return [end]
    step = (end - start) / (n - 1)
    return [start + i * step for i in range(n)]


def build_ladder(n_steps, layout, diff_min_range, diff_max_range):
    """diff_min/diff_max lesson dicts: an n_steps-long walk restarted in each stage."""
    min_levels = linspace(diff_min_range[0], diff_min_range[1], n_steps)
    max_levels = linspace(diff_max_range[0], diff_max_range[1], n_steps)

    diff_min, diff_max = {}, {}
    for start, duration in layout:
        # floor, not round: keeps the last lesson of a stage inside that stage
        lesson_time = duration // n_steps
        if lesson_time < 1:
            sys.exit(f'Error: {n_steps} lessons do not fit in a {duration}-update stage '
                     f'(starting at {start}); drop that step count.')
        for j in range(n_steps):
            t = start + j * lesson_time
            diff_min[t] = round(float(min_levels[j]), ROUND_TO)
            diff_max[t] = round(float(max_levels[j]), ROUND_TO)
    return diff_min, diff_max


def validate(schedule, dataset, n_steps):
    """Fail fast on anything training.py would reject or silently ignore."""
    for k in schedule:
        if k in GLOBAL_KEYS or k == 'meta':
            continue
        suffix = next((s for s in PER_DS_SUFFIXES if k.endswith(s)), None)
        if suffix is None:
            sys.exit(f'Error: n={n_steps}: unknown lesson key "{k}"')
        if k[:-len(suffix)] != dataset:
            sys.exit(f'Error: n={n_steps}: lesson "{k}" does not match dataset "{dataset}" '
                     '- it would silently never apply')

    lo = schedule[f'{dataset}_diff_min']
    hi = schedule[f'{dataset}_diff_max']
    if sorted(lo) != sorted(hi):
        sys.exit(f'Error: n={n_steps}: diff_min and diff_max lesson times disagree')
    for t in sorted(lo):
        for name, v in (('diff_min', lo[t]), ('diff_max', hi[t])):
            if not (DIFF_FLOOR - 1e-9 <= v <= DIFF_CEIL + 1e-9):
                sys.exit(f'Error: n={n_steps}: {name}={v} at update {t} is outside '
                         f'the allowed [{DIFF_FLOOR}, {DIFF_CEIL}] band')
        if lo[t] >= hi[t]:
            sys.exit(f'Error: n={n_steps}: diff_min ({lo[t]}) >= diff_max ({hi[t]}) '
                     f'at update {t} - the quantile window would be empty')


def make_config(base_raw, dataset, layout, n_steps, diff_min_range, diff_max_range, base_name):
    schedule = {k: v for k, v in base_raw.items()
                if k == 'meta' or not any(k.endswith(s) for s in LOC_SUFFIXES)}

    diff_min, diff_max = build_ladder(n_steps, layout, diff_min_range, diff_max_range)
    schedule[f'{dataset}_diff_min'] = diff_min
    schedule[f'{dataset}_diff_max'] = diff_max

    meta = dict(schedule.get('meta', {}))
    meta['note'] = (
        f'diff-ladder sweep off {base_name}: {n_steps} lesson(s) per rotate_by stage, '
        f'diff_min {diff_min_range[0]}->{diff_min_range[1]}, '
        f'diff_max {diff_max_range[0]}->{diff_max_range[1]} (endpoint-inclusive, evenly '
        f'spaced, restarted each stage). Everything else is copied from the base. '
        f'Use with loc_algo=linear.'
    )
    meta['sweep'] = {'base': base_name, 'variable': 'diff_ladder_num_lessons',
                     'num_lessons': n_steps,
                     'diff_min_range': list(diff_min_range),
                     'diff_max_range': list(diff_max_range)}
    schedule['meta'] = meta  # keep meta last for readability

    validate(schedule, dataset, n_steps)
    # JSON object keys must be strings; load_tc_schedule casts them back to ints.
    # Sort numerically (not lexicographically) so the written file reads in update order
    # whether the keys came from the base config (str) or from build_ladder (int).
    return {k: ({str(t): val for t, val in sorted(v.items(), key=lambda kv: int(kv[0]))}
                if k != 'meta' and isinstance(v, dict) else v)
            for k, v in schedule.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base', default='081626_const_jitter',
                   help='Base curriculum to copy everything-but-the-diff-ladder from '
                        '(name, filename, or path). Default: %(default)s')
    p.add_argument('--steps', type=int, nargs='+', default=[2, 3, 4, 6, 8, 12],
                   help='Step counts to sweep: the number of diff lessons per rotate_by '
                        'stage, endpoint-inclusive (4 reproduces the base ladder). '
                        'Default: %(default)s')
    p.add_argument('--diff_min_range', type=float, nargs=2, default=[0.1, 0.5],
                   metavar=('START', 'END'), help='diff_min walk. Default: %(default)s')
    p.add_argument('--diff_max_range', type=float, nargs=2, default=[0.2, 0.6],
                   metavar=('START', 'END'), help='diff_max walk. Default: %(default)s')
    p.add_argument('--dataset', default=None,
                   help='Dataset prefix for the lesson keys. Default: inferred from --base')
    p.add_argument('--prefix', default='sweep_diffsteps',
                   help='Filename prefix; pass the same value to slurm-run_ckpt.py '
                        '--sweep_config. Default: %(default)s')
    p.add_argument('--out_dir', default=CURRICULUM_DIR,
                   help='Where to write the configs. Default: the curriculum conf dir')
    p.add_argument('--clean', action='store_true',
                   help='Delete existing {prefix}*.json first, so a stale step count '
                        'from an earlier run does not get swept along')
    p.add_argument('--dry_run', action='store_true',
                   help='Print the ladders without writing anything')
    args = p.parse_args()

    for name, rng in (('--diff_min_range', args.diff_min_range),
                      ('--diff_max_range', args.diff_max_range)):
        for v in rng:
            if not (DIFF_FLOOR - 1e-9 <= v <= DIFF_CEIL + 1e-9):
                sys.exit(f'Error: {name} value {v} is outside the allowed '
                         f'[{DIFF_FLOOR}, {DIFF_CEIL}] band')
        if rng[0] > rng[1]:
            sys.exit(f'Error: {name} must be non-decreasing, got {rng}')
    if args.diff_min_range[0] >= args.diff_max_range[0] or \
            args.diff_min_range[1] >= args.diff_max_range[1]:
        sys.exit('Error: diff_min must stay strictly below diff_max at both endpoints')

    steps = sorted(set(args.steps))
    if any(n < 2 for n in steps):
        sys.exit('Error: --steps values must be >= 2 (a 1-lesson ladder does not walk)')

    base_path = resolve_base_path(args.base)
    base_name = os.path.splitext(os.path.basename(base_path))[0]
    with open(base_path) as f:
        base_raw = json.load(f)

    dataset = infer_dataset(base_raw, args.dataset)
    total_num_updates = base_raw.get('meta', {}).get('total_num_updates')
    if total_num_updates is None:
        sys.exit(f'Error: {base_name} has no meta.total_num_updates; '
                 'load_tc_schedule needs it to realign lesson times.')
    layout = stage_layout(base_raw, dataset, int(total_num_updates))

    dropped = [k for k in base_raw if any(k.endswith(s) for s in LOC_SUFFIXES)]
    print(f'base:     {base_path}')
    print(f'dataset:  {dataset}')
    print(f'stages:   {[f"{s}(+{d})" for s, d in layout]}  '
          f'total_num_updates={total_num_updates}')
    print(f'replaced: {dropped or "(none)"}')
    print(f'sweeping: diff_min {args.diff_min_range[0]}->{args.diff_min_range[1]}, '
          f'diff_max {args.diff_max_range[0]}->{args.diff_max_range[1]} '
          f'over {steps} lessons/stage\n')

    if args.clean and not args.dry_run:
        for stale in sorted(glob.glob(os.path.join(args.out_dir, f'{args.prefix}*.json'))):
            os.remove(stale)
            print(f'removed  {os.path.basename(stale)}')

    os.makedirs(args.out_dir, exist_ok=True)
    written = []
    for n in steps:
        cfg = make_config(base_raw, dataset, layout, n,
                          args.diff_min_range, args.diff_max_range, base_name)
        fname = f'{args.prefix}_n{n:02d}.json'
        path = os.path.join(args.out_dir, fname)

        lo = cfg[f'{dataset}_diff_min']
        hi = cfg[f'{dataset}_diff_max']
        first_stage = [t for t in sorted(lo, key=int) if int(t) < layout[0][1]]
        every = layout[0][1] // n
        print(f'{fname}  {n:>2} lessons/stage, every {every} updates, '
              f'{len(lo)} lessons total')
        print('    ' + '  '.join(f'@{t}:[{lo[t]:.3f},{hi[t]:.3f}]' for t in first_stage)
              + ('  ... x%d stages' % len(layout) if len(layout) > 1 else ''))

        if not args.dry_run:
            with open(path, 'w') as f:
                json.dump(cfg, f, indent=2)
                f.write('\n')
            written.append(path)

    if args.dry_run:
        print('\ndry run: nothing written')
    else:
        print(f'\nwrote {len(written)} configs to {args.out_dir}')
        print(f'sweep them with: python3 scripts/slurm-run_ckpt.py --n_seeds <N> '
              f'--sweep_config {args.prefix} --sweep_outsuffix '
              f'--override "... loc_algo=linear \'dataset=[{dataset}]\'"')


if __name__ == '__main__':
    main()
