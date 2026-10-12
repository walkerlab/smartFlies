# python3 scripts/gen_cl_sweep_diff.py --base 081626_noisy_jitter_coverage     --steps 3 5 7 --birthx_steps 3 5 7 --prefix sweep_357_noisy --plot
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
stage, matching how the retired ``build_tc_schedule_dict`` laid out the ``slice`` lessons.

Optionally the ``birthx`` tail is swept as a second dimension (``--birthx_steps``).
The tail is the odor-sparsity ramp that opens after the last ``rotate_by`` stage and
runs to the end of training; in the base configs it steps ``0.7 -> 0.1`` in 4 lessons
spaced geometrically (0.7, 0.366, 0.191, 0.1). Its *window* (first tail lesson ->
``total_num_updates``) is kept from the base, so the diff stages are never touched;
what is swept is again the number of lessons the ramp takes. The ramp fills the first
75% of the window in equal slots and *ends* there (475 for a 400->500 window); the
last 25% is a pure hold at the final sparsity (``--birthx_hold_frac``), the same
length for every step count. ``--birthx_range`` and ``--birthx_spacing`` are fixed
knobs (defaults: the base tail's own first/last values, geometric). When both
``--steps`` and ``--birthx_steps`` are given the full grid is written; hold either at
a single value to sweep the other alone. Note the hand-authored 081626_* tail fills
its whole window, so reproducing it exactly needs ``--birthx_steps 4
--birthx_hold_frac 0``.

``CL_sweep.ipynb`` imports :func:`build_ladder`, :func:`make_config` and
:func:`validate` from here so the notebook's plots and the configs written on disk
can never drift apart; the notebook supplies its own from-scratch skeleton instead of
a base config on disk.

Generated families are *not* checked in - run this (or the notebook) wherever you
submit from. Keep the prefixes disjoint so one ``--sweep_config`` never mixes two
families: this script writes ``sweep_diff_base_*`` by default and the notebook writes
its own per-dataset ``swp_diff_*`` families.

Examples
--------
# 6 configs off the const-jitter base, 2..12 lessons per rotate_by stage
python3 scripts/gen_cl_sweep_diff.py --base 081626_const_jitter --steps 2 3 4 6 8 12

# preview only, nothing written
python3 scripts/gen_cl_sweep_diff.py --base 081626_const_jitter --steps 2 4 8 --dry_run

# sweep the birthx tail alone (diff ladder held at the base's 4 lessons): 1..8 lessons
# over the base's 400->500 window, 0.7 -> 0.1 geometric; 1 lesson = jump straight to 0.1
python3 scripts/gen_cl_sweep_diff.py --base 081626_const_jitter --steps 4 \
    --birthx_steps 1 2 4 8 --prefix sweep_bx_base

# full 3x3 grid, linear birthx values 0.7 -> 0.3
python3 scripts/gen_cl_sweep_diff.py --base 081626_const_jitter --steps 2 4 8 \
    --birthx_steps 2 4 8 --birthx_range 0.7 0.3 --birthx_spacing linear --prefix sweep_grid

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

# birthx is the lower bound of the per-episode puff-density draw
# (PlumeEnvironment.reset: U(birthx, 1.0)), so it must be a probability and non-zero
# (the env clips birthx_max at 0.01 for the same reason).
BIRTHX_FLOOR, BIRTHX_CEIL = 0.01, 1.0
BIRTHX_SPACINGS = ('geometric', 'linear')
# Fraction of the birthx window kept after the ramp ends so the agent trains at the
# final sparsity for the same stretch whatever the step count (CL_sweep.ipynb's
# HOLD_FRAC). The hand-authored 081626_* tails fill the whole window (hold 0).
BIRTHX_HOLD_FRAC = 0.25

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


def geomspace(start, end, n):
    """Endpoint-inclusive geometric spacing (np.geomspace without the numpy import)."""
    if n == 1:
        return [end]
    ratio = (end / start) ** (1.0 / (n - 1))
    return [start * ratio ** i for i in range(n)]


def check_birthx_range(birthx_range):
    """Reject birthx values the env cannot use as a puff-density lower bound."""
    for v in birthx_range:
        if not (BIRTHX_FLOOR - 1e-9 <= v <= BIRTHX_CEIL + 1e-9):
            raise ValueError(f'birthx_range value {v} is outside the allowed '
                             f'[{BIRTHX_FLOOR}, {BIRTHX_CEIL}] band')
    if birthx_range[0] <= birthx_range[1]:
        raise ValueError('birthx_range must go from dense to sparse (start > end), '
                         f'got {list(birthx_range)}')


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

    Stages are the ``rotate_by`` lessons, matching the retired build_tc_schedule_dict's
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
    # The birthx tail owns the run after the last rotate_by stage - every CL config in
    # conf/curriculum follows that, and the retired build_tc_schedule_dict's 'birthx_cl_last' put
    # the lessons there explicitly. A birthx lesson landing *between* the first and last
    # stage means the tail is interleaved with the ladder, which is a skeleton mistake:
    # catch it here rather than let it silently stretch the final stage. (A lesson at or
    # before the first stage start is fine - that is the no-CL control's flat birthx.)
    interior = [t for t in birthx_times if starts[0] < t <= starts[-1]]
    if interior:
        raise ValueError(
            f'birthx lessons {interior} fall inside the rotate_by stages '
            f'(stages start at {starts}), so the birthx tail is interleaved with the '
            f'diff ladder. The tail must open after the last stage starts - move it to '
            f'{starts[-1] + (starts[1] - starts[0] if len(starts) > 1 else 0)} or later.')

    tail_end = next((t for t in birthx_times if t > starts[-1]), total_num_updates)

    layout = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else tail_end
        if end <= start:
            raise ValueError(
                f'rotate_by stage starting at update {start} ends at {end}, so it has '
                f'a duration of {end - start}. The boundary comes from '
                + ('the first birthx lesson after that stage' if end in birthx_times
                   else 'meta.total_num_updates')
                + f' - the birthx tail must open *after* the last rotate_by stage '
                  f'(stages start at {starts}, birthx lessons at {birthx_times}).')
        layout.append((start, end - start))
    return layout


def birthx_tail_window(base_schedule, dataset, total_num_updates):
    """Return (tail_start, tail_end, base_range) of the base config's birthx tail.

    The tail is every ``birthx`` lesson after the last ``rotate_by`` stage opens (the
    same boundary :func:`stage_layout` uses to close that stage), and it runs to the
    end of the schedule. ``base_range`` is (first, last) value of that tail, so a
    sweep can keep the base's own dense -> sparse endpoints by default.
    """
    rotate_key = f'{dataset}_rotate_by'
    starts = sorted(int(t) for t in base_schedule.get(rotate_key, {}))
    last_stage_start = starts[-1] if starts else -1
    tail = sorted((int(t), v) for t, v in base_schedule.get('birthx', {}).items()
                  if int(t) > last_stage_start)
    if not tail:
        raise ValueError(
            'base config has no birthx tail (no birthx lesson after the last rotate_by '
            f'stage starts at {last_stage_start}); there is no window to sweep '
            '--birthx_steps in. Pick a base with a birthx ramp, e.g. 081626_const_jitter.')
    tail_start = tail[0][0]
    if tail_start >= total_num_updates:
        raise ValueError(f'birthx tail opens at {tail_start} but meta.total_num_updates '
                         f'is {total_num_updates}; no room for a ramp.')
    return tail_start, total_num_updates, (tail[0][1], tail[-1][1])


def build_birthx_tail(n_steps, window, birthx_range=(0.7, 0.1), spacing='geometric',
                      hold_frac=BIRTHX_HOLD_FRAC):
    """birthx lesson dict: an n_steps-long dense -> sparse ramp across the tail window.

    The ramp occupies the first ``1 - hold_frac`` of the window and is cut into
    ``n_steps`` equal slots, each lesson opening one slot; the curriculum therefore
    *ends* at ``tail_start + (1 - hold_frac) * window`` (475 for the 081626_* 400->500
    window at the default 25% hold) and the remaining ``hold_frac`` is a pure hold at
    the sparse endpoint, the same length for every step count. So n_steps=3 gives
    400, 425, 450 and holds from 475; the hand-authored 081626_* tail (400, 425, 450,
    475) is ``n_steps=4`` with ``hold_frac=0``. Values are spaced geometrically by
    default - the base configs step 0.7, 0.366, 0.191, 0.1 - or linearly. n_steps=1 is
    the no-ramp control: one lesson at ``tail_start`` straight to the sparse endpoint.

    Args:
        n_steps: number of lessons, endpoint-inclusive (>= 1).
        window: (tail_start, tail_end) from :func:`birthx_tail_window`.
        birthx_range: (dense, sparse) endpoints of the ramp.
        spacing: 'geometric' or 'linear' spacing of the values.
        hold_frac: fraction of the window reserved after the last lesson, in [0, 1).

    Returns:
        dict[int update -> float] birthx lesson track.
    """
    if n_steps < 1:
        raise ValueError(f'birthx n_steps must be >= 1, got {n_steps}')
    if spacing not in BIRTHX_SPACINGS:
        raise ValueError(f'birthx spacing must be one of {BIRTHX_SPACINGS}, got {spacing!r}')
    if not (0.0 <= hold_frac < 1.0):
        raise ValueError(f'birthx hold_frac must be in [0, 1), got {hold_frac}')
    check_birthx_range(birthx_range)
    tail_start, tail_end = window
    ramp_span = (tail_end - tail_start) * (1.0 - hold_frac)
    slot = ramp_span / n_steps
    times = [tail_start + int(round(j * slot)) for j in range(n_steps)]
    if len(set(times)) < n_steps:
        raise ValueError(f'{n_steps} birthx lessons do not fit in the '
                         f'{ramp_span:g}-update ramp (tail {tail_start}->{tail_end} minus '
                         f'a {hold_frac:.0%} hold); drop that step count or lower '
                         '--birthx_hold_frac.')
    space = geomspace if spacing == 'geometric' else linspace
    levels = space(birthx_range[0], birthx_range[1], n_steps)
    return {t: round(float(v), ROUND_TO) for t, v in zip(times, levels)}


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

    for t, v in sorted(schedule.get('birthx', {}).items(), key=lambda kv: int(kv[0])):
        if not (BIRTHX_FLOOR - 1e-9 <= v <= BIRTHX_CEIL + 1e-9):
            raise ValueError(f'birthx={v} at update {t} is outside the allowed '
                             f'[{BIRTHX_FLOOR}, {BIRTHX_CEIL}] band')
    # A birthx lesson inside a diff stage would stretch/interleave the ladder; the
    # diff lessons are the stage starts and everything in between, so any birthx lesson
    # strictly between the first and last diff lesson is misplaced.
    diff_times = sorted(int(t) for t in lo)
    stray = [int(t) for t in schedule.get('birthx', {})
             if diff_times[0] < int(t) <= diff_times[-1]]
    if stray:
        raise ValueError(f'birthx lessons {stray} fall inside the diff ladder '
                         f'({diff_times[0]}..{diff_times[-1]}); the birthx tail must '
                         'open after the last diff lesson')


def make_config(base_raw, dataset, layout, n_steps,
                diff_min_range=(0.1, 0.5), diff_max_range=(0.2, 0.6), base_name='(inline)',
                birthx_steps=None, birthx_range=None, birthx_spacing='geometric',
                birthx_hold_frac=BIRTHX_HOLD_FRAC):
    """Return a JSON-ready config: ``base_raw`` with its loc_algo ladder swapped out.

    ``base_raw`` may be a config loaded off disk or a skeleton built in a notebook; it
    only needs the non-loc_algo tracks (``birthx``, ``wind_cond``, ``*_rotate_by``) and
    a ``meta`` with ``total_num_updates``.

    With ``birthx_steps`` the base's birthx tail (see :func:`birthx_tail_window`) is
    replaced by a ``birthx_steps``-lesson ramp over the same window; birthx lessons at or
    before the first stage (a flat no-CL control value) are kept. ``birthx_range``
    defaults to the base tail's own (first, last) values.
    """
    schedule = {k: v for k, v in base_raw.items()
                if k == 'meta' or not any(k.endswith(s) for s in LOC_SUFFIXES)}

    diff_min, diff_max = build_ladder(n_steps, layout, diff_min_range, diff_max_range)
    schedule[f'{dataset}_diff_min'] = diff_min
    schedule[f'{dataset}_diff_max'] = diff_max

    birthx_note = 'birthx copied from the base.'
    if birthx_steps is not None:
        total = int(schedule.get('meta', {}).get('total_num_updates', layout[-1][0] + layout[-1][1]))
        tail_start, tail_end, base_range = birthx_tail_window(base_raw, dataset, total)
        birthx_range = tuple(birthx_range) if birthx_range else base_range
        kept = {int(t): v for t, v in base_raw.get('birthx', {}).items()
                if int(t) <= layout[0][0]}
        kept.update(build_birthx_tail(birthx_steps, (tail_start, tail_end),
                                      birthx_range, birthx_spacing, birthx_hold_frac))
        schedule['birthx'] = kept
        birthx_note = (f'birthx tail rebuilt: {birthx_steps} lesson(s) over '
                       f'{tail_start}->{tail_end} with a {birthx_hold_frac:.0%} trailing '
                       f'hold, {birthx_range[0]}->{birthx_range[1]} {birthx_spacing}.')

    meta = dict(schedule.get('meta', {}))
    meta['note'] = (
        f'diff-ladder sweep off {base_name}: {n_steps} lesson(s) per rotate_by stage, '
        f'diff_min {diff_min_range[0]}->{diff_min_range[1]}, '
        f'diff_max {diff_max_range[0]}->{diff_max_range[1]} (endpoint-inclusive, evenly '
        f'spaced, restarted each stage). {birthx_note} Everything else is copied from '
        f'the base. Use with loc_algo=slice_linear.'
    )
    variables = ['diff_ladder_num_lessons']
    if birthx_steps is not None:
        variables.append('birthx_tail_num_lessons')
    meta['sweep'] = {'base': base_name, 'variable': variables,
                     'num_lessons': n_steps,
                     'diff_min_range': list(diff_min_range),
                     'diff_max_range': list(diff_max_range)}
    if birthx_steps is not None:
        meta['sweep'].update({'birthx_num_lessons': birthx_steps,
                              'birthx_range': list(birthx_range),
                              'birthx_spacing': birthx_spacing,
                              'birthx_hold_frac': birthx_hold_frac,
                              'birthx_window': [tail_start, tail_end]})
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


def _step_xy(lessons, t_end):
    """(times, values) for a post-step plot, with the last value held to ``t_end``."""
    ts = sorted(lessons, key=int)
    xs = [int(t) for t in ts] + [t_end]
    ys = [lessons[t] for t in ts] + [lessons[ts[-1]]]
    return xs, ys


def plot_family(configs, dataset, layout, total_num_updates, path, title=''):
    """Preview the whole family in one figure: the diff_min ladders (solid, left axis) and
    the birthx tails (dashed, twin right axis).

    Within a family the diff ladder depends only on the diff step count and the birthx
    tail only on the birthx step count, so each distinct ladder and each distinct tail
    is drawn once: colors follow ``n`` on the left axis and ``b`` on the right, with one
    legend each. When birthx is not swept the shared base tail is drawn once in gray.

    Args:
        configs: [((n_steps, birthx_steps or None), cfg), ...] as built in :func:`main`.
        path: PNG to write. matplotlib is imported lazily so generating configs never
            needs it.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ladders = {}   # n -> diff_min lessons (identical for every b)
    tails = {}     # b -> birthx lessons (identical for every n)
    for (n, b), cfg in configs:
        ladders.setdefault(n, cfg[f'{dataset}_diff_min'])
        if b is not None and cfg.get('birthx'):
            tails.setdefault(b, cfg['birthx'])

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax_bx = ax.twinx()
    diff_cmap = plt.get_cmap('tab10' if len(ladders) <= 10 else 'tab20')
    bx_cmap = plt.get_cmap('Dark2' if len(tails) <= 8 else 'tab20b')

    for i, (n, lessons) in enumerate(sorted(ladders.items())):
        xs, ys = _step_xy(lessons, total_num_updates)
        ax.step(xs, ys, where='post', color=diff_cmap(i % diff_cmap.N), lw=1.6,
                label=f'diff n{n:02d}')
    if tails:
        for i, (b, lessons) in enumerate(sorted(tails.items())):
            xs, ys = _step_xy(lessons, total_num_updates)
            ax_bx.step(xs, ys, where='post', color=bx_cmap(i % bx_cmap.N), lw=1.6, ls='--',
                       label=f'birthx b{b:02d}')
    elif configs[0][1].get('birthx'):
        xs, ys = _step_xy(configs[0][1]['birthx'], total_num_updates)
        ax_bx.step(xs, ys, where='post', color='0.45', lw=1.6, ls='--', label='birthx (base)')
    sweep = configs[0][1].get('meta', {}).get('sweep', {})
    if 'birthx_window' in sweep and sweep.get('birthx_hold_frac', 0) > 0:
        w0, w1 = sweep['birthx_window']
        hold_start = w0 + (w1 - w0) * (1.0 - sweep['birthx_hold_frac'])
        ax.axvspan(hold_start, w1, color='0.9', zorder=0,
                   label=f'birthx hold ({sweep["birthx_hold_frac"]:.0%})')

    for start, _ in layout[1:]:
        ax.axvline(start, color='0.8', lw=0.8, ls=':')
    ax.set_xlim(0, total_num_updates)
    ax.set_xlabel('update')
    ax.set_ylabel('diff_min (solid)')
    ax.set_ylim(0, DIFF_CEIL + 0.05)
    ax_bx.set_ylabel('birthx (dashed)')
    ax_bx.set_ylim(0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc='upper left', fontsize=8, frameon=False)
    ax_bx.legend(loc='upper right', fontsize=8, frameon=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
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
    sweep = cfg.get('meta', {}).get('sweep', {})
    if 'birthx_num_lessons' in sweep:
        bx = cfg['birthx']
        tail = [t for t in sorted(bx, key=int) if int(t) > layout[0][0]]
        head += f'; birthx {len(tail)} lessons'
        ladder += '\n    birthx ' + '  '.join(f'@{t}:{bx[t]:.3f}' for t in tail)
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
    p.add_argument('--birthx_steps', type=int, nargs='+', default=None,
                   help='Also sweep the birthx tail: number of lessons the dense->sparse '
                        'ramp takes over the base tail window (1 = jump straight to the '
                        'sparse value; 4 with --birthx_hold_frac 0 reproduces the 081626_* '
                        'tail). Combined with --steps as a full grid. Default: birthx '
                        'copied verbatim from the base')
    p.add_argument('--birthx_range', type=float, nargs=2, default=None,
                   metavar=('DENSE', 'SPARSE'),
                   help='birthx ramp endpoints, dense first. Default: the base tail\'s own '
                        'first/last values (0.7 -> 0.1 in the 081626_* configs)')
    p.add_argument('--birthx_spacing', choices=BIRTHX_SPACINGS, default='geometric',
                   help='Spacing of the birthx values between the endpoints. Default: '
                        '%(default)s, matching the hand-authored tails')
    p.add_argument('--birthx_hold_frac', type=float, default=BIRTHX_HOLD_FRAC,
                   help='Fraction of the birthx window kept as a pure hold after the ramp '
                        'ends, so the agent trains at the final sparsity for the same '
                        'stretch regardless of step count (CL_sweep.ipynb HOLD_FRAC). '
                        'Default: %(default)s')
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
                   help='Print the ladders without writing any config (a --plot preview '
                        'is still written)')
    p.add_argument('--plot', action='store_true',
                   help='Also save {out_dir}/{prefix}_preview.png: diff_min of every config '
                        'on the left axis, the birthx tail on a twin right axis. Needs '
                        'matplotlib')
    args = p.parse_args()

    try:
        check_ranges(args.diff_min_range, args.diff_max_range)
        steps = sorted(set(args.steps))
        if any(n < 2 for n in steps):
            raise ValueError('--steps values must be >= 2 (a 1-lesson ladder does not walk)')
        birthx_steps = sorted(set(args.birthx_steps)) if args.birthx_steps else [None]
        if any(b is not None and b < 1 for b in birthx_steps):
            raise ValueError('--birthx_steps values must be >= 1')
        if args.birthx_range:
            check_birthx_range(args.birthx_range)
        if not (0.0 <= args.birthx_hold_frac < 1.0):
            raise ValueError(f'--birthx_hold_frac must be in [0, 1), got {args.birthx_hold_frac}')
        if args.birthx_steps is None and (args.birthx_range or args.birthx_spacing != 'geometric'
                                          or args.birthx_hold_frac != BIRTHX_HOLD_FRAC):
            raise ValueError('--birthx_range / --birthx_spacing / --birthx_hold_frac only apply '
                             'together with --birthx_steps (without it birthx is copied from '
                             'the base)')

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
              f'over {steps} lessons/stage')
        if args.birthx_steps:
            tail_start, tail_end, base_range = birthx_tail_window(base_raw, dataset,
                                                                  int(total_num_updates))
            bx_range = tuple(args.birthx_range) if args.birthx_range else base_range
            print(f'          birthx tail {tail_start}->{tail_end} '
                  f'({args.birthx_hold_frac:.0%} trailing hold), '
                  f'{bx_range[0]}->{bx_range[1]} {args.birthx_spacing}, '
                  f'over {birthx_steps} lessons '
                  f'({len(steps) * len(birthx_steps)} configs)')
        print()

        configs = [((n, b), make_config(base_raw, dataset, layout, n, args.diff_min_range,
                                        args.diff_max_range, base_name,
                                        birthx_steps=b, birthx_range=args.birthx_range,
                                        birthx_spacing=args.birthx_spacing,
                                        birthx_hold_frac=args.birthx_hold_frac))
                   for n in steps for b in birthx_steps]
    except ValueError as e:
        sys.exit(f'Error: {e}')

    if args.clean and not args.dry_run:
        for stale in sorted(glob.glob(os.path.join(args.out_dir, f'{args.prefix}*.json'))):
            os.remove(stale)
            print(f'removed  {os.path.basename(stale)}')

    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)
    written = []
    for (n, b), cfg in configs:
        fname = f'{args.prefix}_n{n:02d}' + (f'_b{b:02d}' if b is not None else '') + '.json'
        head, ladder = describe(cfg, dataset, layout, n)
        print(f'{fname}  {head}')
        print(f'    {ladder}')
        if not args.dry_run:
            written.append(dump_config(cfg, os.path.join(args.out_dir, fname)))

    if args.plot:
        os.makedirs(args.out_dir, exist_ok=True)
        plot_path = plot_family(
            [(key, cfg) for key, cfg in configs], dataset, layout, int(total_num_updates),
            os.path.join(args.out_dir, f'{args.prefix}_preview.png'),
            title=f'{args.prefix}: diff ladder off {base_name}'
                  + (f', birthx {args.birthx_spacing}' if args.birthx_steps else ''))
        print(f'\npreview  {plot_path}')

    if args.dry_run:
        print('\ndry run: no configs written')
    else:
        print(f'\nwrote {len(written)} configs to {args.out_dir}')
        print(f'sweep them with: python3 scripts/slurm-run_ckpt.py --n_seeds <N> '
              f'--sweep_config {args.prefix} --sweep_outsuffix '
              f'--override "... loc_algo=slice_linear \'dataset=[{dataset}]\'"')


if __name__ == '__main__':
    main()
