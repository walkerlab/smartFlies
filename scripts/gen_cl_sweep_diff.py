#!/usr/bin/env python3
"""Generate a family of curriculum configs that sweep the diff_min/diff_max ladder.

The ``slice`` location algorithm (``loc_algo=slice_linear``) initializes agents at a
plume-density quantile drawn as ``q ~ U(diff_min, diff_max)``, then takes the
``[q - 0.1, q]`` quantile band of the longitudinal puff distribution. So the two
thresholds couple: they bracket *which* quantile an episode starts at, and the pair
walks together as a window sliding from easy (near the source) to hard (plume edge).
``diff_min`` must stay >= 0.1 so ``q - 0.1`` never goes negative, and both stay <= 0.6.

This replaces the ``now_init_long`` ladder that ``CL_sweep.ipynb`` used to sweep (that
one drives the ``precise`` branch instead). What is swept here is the *number of
lessons* the window takes to cross its range: ``diff_min`` from ``--diff_min_range``
(default 0.1 -> 0.5) and ``diff_max`` from ``--diff_max_range`` (default 0.2 -> 0.6),
endpoint-inclusive and evenly spaced. One config is written per step count, all
sharing a filename prefix so ``slurm-run_ckpt.py --sweep_config <prefix>`` picks the
whole family up and submits one job per config.

Everything that is not the diff ladder (``birthx``, ``wind_cond``, ``rotate_by``,
``meta``) is copied verbatim from a base config, so a sweep differs from the base in
exactly one dimension. The ladder is restarted from scratch inside each ``rotate_by``
stage, matching how ``build_tc_schedule_dict`` lays out the ``slice`` lessons.

``CL_sweep.ipynb`` imports :func:`build_ladder`, :func:`make_config` and
:func:`validate` from here so the notebook's plots and the configs written on disk
can never drift apart; the notebook supplies its own from-scratch skeleton instead of
a base config on disk.

Generated families are *not* checked in - run this (or the notebook) wherever you
submit from. Keep the prefixes disjoint so one ``--sweep_config`` never mixes two
families: this script writes ``sweep_diff_base_*`` by default and the notebook writes
``sweep_diff_const_*`` / ``sweep_diff_noisy_*``, while the shared ``sweep_diff`` root
is there when you do want to sweep every diff family at once.

Examples
--------
# 6 configs off the const-jitter base, 2..12 lessons per rotate_by stage
python3 scripts/gen_cl_sweep_diff.py --base 081626_const_jitter --steps 2 3 4 6 8 12

# preview only, nothing written
python3 scripts/gen_cl_sweep_diff.py --base 081626_const_jitter --steps 2 4 8 --dry_run

# then, on hyak:
python3 scripts/slurm-run_ckpt.py --n_seeds 5 --sweep_config sweep_diff \
    --sweep_outsuffix --override "... loc_algo=slice_linear 'dataset=[constant_jitterx5b5]'"
"""

import argparse
import glob
import json
import os
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURRICULUM_DIR = os.path.join(REPO_DIR, 'tamagotchi', 'conf', 'curriculum')

# Hard bounds on the diff thresholds. The floor is structural: get_initial_location
# takes the [q - 0.1, q] quantile band, so q < 0.1 would ask for a negative quantile.
DIFF_FLOOR, DIFF_CEIL = 0.1, 0.6
# The loc_algo substring that reads diff_min/diff_max (see validate_tc_schedule_datasets).
DIFF_LOC_ALGO = 'slice'

GLOBAL_KEYS = ('birthx', 'wind_cond')
# Lesson suffixes owned by the loc_algo, and therefore *replaced* by the sweep rather
# than carried over from the base config.
LOC_SUFFIXES = ('_diff_min', '_diff_max', '_now_init_long')
# Everything data_util.validate_tc_schedule_datasets accepts as a per-dataset lesson.
PER_DS_SUFFIXES = ('_diff_max', '_diff_min', '_now_init_long', '_rotate_by')

ROUND_TO = 6  # match the precision of the hand-authored configs


def linspace(start, end, n):
    """Endpoint-inclusive even spacing (np.linspace without the numpy import)."""
    if n == 1:
        return [end]
    step = (end - start) / (n - 1)
    return [start + i * step for i in range(n)]


def check_ranges(diff_min_range, diff_max_range):
    """Reject threshold ranges the location algorithm cannot honour."""
    for name, rng in (('diff_min_range', diff_min_range), ('diff_max_range', diff_max_range)):
        for v in rng:
            if not (DIFF_FLOOR - 1e-9 <= v <= DIFF_CEIL + 1e-9):
                raise ValueError(f'{name} value {v} is outside the allowed '
                                 f'[{DIFF_FLOOR}, {DIFF_CEIL}] band')
        if rng[0] > rng[1]:
            raise ValueError(f'{name} must be non-decreasing, got {list(rng)}')
    if diff_min_range[0] >= diff_max_range[0] or diff_min_range[1] >= diff_max_range[1]:
        raise ValueError('diff_min must stay strictly below diff_max at both endpoints, '
                         f'got min={list(diff_min_range)} max={list(diff_max_range)}')


def resolve_base_path(base):
    """Accept a curriculum name, a bare filename, or a path."""
    candidates = [base, f'{base}.json',
                  os.path.join(CURRICULUM_DIR, base),
                  os.path.join(CURRICULUM_DIR, f'{base}.json')]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise ValueError(f'base curriculum "{base}" not found. Tried:\n  ' + '\n  '.join(candidates))


def infer_dataset(base_schedule, override=None):
    """Pull the dataset name out of a config's per-dataset lesson keys."""
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
        raise ValueError('base config has no per-dataset lesson keys; pass --dataset explicitly.')
    if len(names) > 1:
        raise ValueError(f'base config spans multiple datasets {sorted(names)}; '
                         'pass --dataset to pick one.')
    return names.pop()


def stage_layout(base_schedule, dataset, total_num_updates):
    """Return [(stage_start, stage_duration), ...] for the diff ladder to repeat in.

    Stages are the ``rotate_by`` lessons, matching build_tc_schedule_dict's
    rotate_stage_times. Each stage runs until the next one starts; the last runs until
    the first ``birthx`` lesson beyond it (the birthx tail owns the rest of the run)
    or, absent that, to the end of the run.
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
            raise ValueError(f'stage starting at {start} has non-positive duration '
                             f'(next boundary {end}); check the base config.')
        layout.append((start, end - start))
    return layout


def build_ladder(n_steps, layout, diff_min_range=(0.1, 0.5), diff_max_range=(0.2, 0.6)):
    """diff_min/diff_max lesson dicts: an n_steps-long walk restarted in each stage.

    Args:
        n_steps: number of lessons per stage, endpoint-inclusive (so n_steps=2 is just
            the two endpoints, and n_steps=4 reproduces the 081626_* configs).
        layout: [(stage_start, stage_duration), ...] from :func:`stage_layout`.
        diff_min_range, diff_max_range: (start, end) of each threshold's walk.

    Returns:
        (diff_min, diff_max): two dict[int update -> float] lesson tracks.
    """
    if n_steps < 2:
        raise ValueError(f'n_steps must be >= 2 (a 1-lesson ladder does not walk), got {n_steps}')
    check_ranges(diff_min_range, diff_max_range)
    min_levels = linspace(diff_min_range[0], diff_min_range[1], n_steps)
    max_levels = linspace(diff_max_range[0], diff_max_range[1], n_steps)

    diff_min, diff_max = {}, {}
    for start, duration in layout:
        # floor, not round: keeps the last lesson of a stage inside that stage
        lesson_time = duration // n_steps
        if lesson_time < 1:
            raise ValueError(f'{n_steps} lessons do not fit in a {duration}-update stage '
                             f'(starting at {start}); drop that step count.')
        for j in range(n_steps):
            t = start + j * lesson_time
            diff_min[t] = round(float(min_levels[j]), ROUND_TO)
            diff_max[t] = round(float(max_levels[j]), ROUND_TO)
    return diff_min, diff_max


def validate(schedule, dataset):
    """Fail fast on anything training.py would reject or silently ignore.

    Mirrors data_util.validate_tc_schedule_datasets (with loc_algo=slice_linear) so a
    bad config is caught at generation time instead of minutes into a GPU job, and
    adds the diff-specific band/ordering checks that validator does not make.
    """
    for k in schedule:
        if k in GLOBAL_KEYS or k == 'meta':
            continue
        suffix = next((s for s in PER_DS_SUFFIXES if k.endswith(s)), None)
        if suffix is None:
            raise ValueError(f'unknown lesson key "{k}"')
        if k[:-len(suffix)] != dataset:
            raise ValueError(f'lesson "{k}" does not match dataset "{dataset}" '
                             '- it would silently never apply')
        if suffix == '_now_init_long':
            raise ValueError(f'lesson "{k}" is read by the "precise" branch of '
                             f'get_initial_location, not "{DIFF_LOC_ALGO}" - it cannot '
                             'coexist with a diff ladder in one config')

    for suffix in ('_diff_min', '_diff_max'):
        if f'{dataset}{suffix}' not in schedule:
            raise ValueError(f'config is missing the "{dataset}{suffix}" track')
    lo = schedule[f'{dataset}_diff_min']
    hi = schedule[f'{dataset}_diff_max']
    if sorted(int(t) for t in lo) != sorted(int(t) for t in hi):
        raise ValueError('diff_min and diff_max lesson times disagree')
    for t in sorted(lo, key=int):
        for name, v in (('diff_min', lo[t]), ('diff_max', hi[t])):
            if not (DIFF_FLOOR - 1e-9 <= v <= DIFF_CEIL + 1e-9):
                raise ValueError(f'{name}={v} at update {t} is outside the allowed '
                                 f'[{DIFF_FLOOR}, {DIFF_CEIL}] band')
        if lo[t] >= hi[t]:
            raise ValueError(f'diff_min ({lo[t]}) >= diff_max ({hi[t]}) at update {t} '
                             '- the quantile window would be empty')


def make_config(base_raw, dataset, layout, n_steps,
                diff_min_range=(0.1, 0.5), diff_max_range=(0.2, 0.6), base_name='(inline)'):
    """Return a JSON-ready config: ``base_raw`` with its loc_algo ladder swapped out.

    ``base_raw`` may be a config loaded off disk or a skeleton built in a notebook; it
    only needs the non-loc_algo tracks (``birthx``, ``wind_cond``, ``*_rotate_by``) and
    a ``meta`` with ``total_num_updates``.
    """
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
        f'Use with loc_algo=slice_linear.'
    )
    meta['sweep'] = {'base': base_name, 'variable': 'diff_ladder_num_lessons',
                     'num_lessons': n_steps,
                     'diff_min_range': list(diff_min_range),
                     'diff_max_range': list(diff_max_range)}
    schedule['meta'] = meta  # keep meta last for readability

    validate(schedule, dataset)
    # JSON object keys must be strings; load_tc_schedule casts them back to ints.
    # Sort numerically (not lexicographically) so the written file reads in update order
    # whether the keys came from the base config (str) or from build_ladder (int).
    return {k: ({str(t): val for t, val in sorted(v.items(), key=lambda kv: int(kv[0]))}
                if k != 'meta' and isinstance(v, dict) else v)
            for k, v in schedule.items()}


def dump_config(cfg, path):
    """Write one config to disk in the same style as the hand-authored ones."""
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)
        f.write('\n')
    return path


def describe(cfg, dataset, layout, n_steps):
    """One-line-plus-ladder summary of a generated config, for logs and notebooks."""
    lo = cfg[f'{dataset}_diff_min']
    hi = cfg[f'{dataset}_diff_max']
    first_stage = [t for t in sorted(lo, key=int) if int(t) < layout[0][1]]
    every = layout[0][1] // n_steps
    head = (f'{n_steps:>2} lessons/stage, every {every} updates, {len(lo)} lessons total')
    ladder = '  '.join(f'@{t}:[{lo[t]:.3f},{hi[t]:.3f}]' for t in first_stage)
    if len(layout) > 1:
        ladder += f'  ... x{len(layout)} stages'
    return head, ladder


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base', default='081626_const_jitter',
                   help='Base curriculum to copy everything-but-the-diff-ladder from '
                        '(name, filename, or path). Default: %(default)s')
    p.add_argument('--steps', type=int, nargs='+', default=[2, 3, 4, 6, 8, 12],
                   help='Step counts to sweep: the number of diff lessons per rotate_by '
                        'stage, endpoint-inclusive (4 reproduces the 081626_* ladder). '
                        'Default: %(default)s')
    p.add_argument('--diff_min_range', type=float, nargs=2, default=[0.1, 0.5],
                   metavar=('START', 'END'), help='diff_min walk. Default: %(default)s')
    p.add_argument('--diff_max_range', type=float, nargs=2, default=[0.2, 0.6],
                   metavar=('START', 'END'), help='diff_max walk. Default: %(default)s')
    p.add_argument('--dataset', default=None,
                   help='Dataset prefix for the lesson keys. Default: inferred from --base')
    p.add_argument('--prefix', default='sweep_diff_base',
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

    try:
        check_ranges(args.diff_min_range, args.diff_max_range)
        steps = sorted(set(args.steps))
        if any(n < 2 for n in steps):
            raise ValueError('--steps values must be >= 2 (a 1-lesson ladder does not walk)')

        base_path = resolve_base_path(args.base)
        base_name = os.path.splitext(os.path.basename(base_path))[0]
        with open(base_path) as f:
            base_raw = json.load(f)

        dataset = infer_dataset(base_raw, args.dataset)
        total_num_updates = base_raw.get('meta', {}).get('total_num_updates')
        if total_num_updates is None:
            raise ValueError(f'{base_name} has no meta.total_num_updates; '
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

        configs = [(n, make_config(base_raw, dataset, layout, n, args.diff_min_range,
                                   args.diff_max_range, base_name)) for n in steps]
    except ValueError as e:
        sys.exit(f'Error: {e}')

    if args.clean and not args.dry_run:
        for stale in sorted(glob.glob(os.path.join(args.out_dir, f'{args.prefix}*.json'))):
            os.remove(stale)
            print(f'removed  {os.path.basename(stale)}')

    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)
    written = []
    for n, cfg in configs:
        fname = f'{args.prefix}_n{n:02d}.json'
        head, ladder = describe(cfg, dataset, layout, n)
        print(f'{fname}  {head}')
        print(f'    {ladder}')
        if not args.dry_run:
            written.append(dump_config(cfg, os.path.join(args.out_dir, fname)))

    if args.dry_run:
        print('\ndry run: nothing written')
    else:
        print(f'\nwrote {len(written)} configs to {args.out_dir}')
        print(f'sweep them with: python3 scripts/slurm-run_ckpt.py --n_seeds <N> '
              f'--sweep_config {args.prefix} --sweep_outsuffix '
              f'--override "... loc_algo=slice_linear \'dataset=[{dataset}]\'"')


if __name__ == '__main__':
    main()
