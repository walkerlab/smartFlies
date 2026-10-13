"""Trajectories by episode outcome (HOME / OOB / OOT), grouped by plume start time, in the unrotated plume frame.

Two sources feed the same per-step frame (see `FRAME_COLUMNS`):

* eval: `episode_frame(logs, summaries)` from evalCli's in-memory episode logs. evalCli calls
  `plot_outcome_starts` after each dataset finishes, so the `{dataset}.pkl` files do not have to be
  loaded afterwards. Every episode is rotated back by -rotate_by (`--eval_rotations` rotate the
  plume, wind and start locations counter-clockwise by rotate_by), so all rotations of an outcome
  share one figure, `<out_prefix>_<outcome>_unrotated.png`, with one panel per start-time window.
* training: `StepFrameBuffer` collects the same columns from the per-step `infos` of the vectorised
  env over the last few updates; `draw_outcome_grid` then draws the outcome x window grid under the
  per-update trajectory figure (tamagotchi/train_plots.py).

Panel contents: odor encounters (red) and misses (black), start (green circle), end (purple
triangle), large wind-direction changes (blue x) and the goal circle at the origin.

`draw_radial_success` / `plot_radial_success` add the HOME success rate (radius, %) against the
plume rotation (angle), to see the agent's bias.
"""
import collections

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from tamagotchi import config

OUTCOMES = ('OOB', 'OOT', 'HOME')
FRAME_COLUMNS = ['ep_idx', 'outcome', 'loc_x', 'loc_y', 't_val', 'odor_01', 'wind_angle_ground_theta_dt1']
# Start-time windows. Eval: the fixed-eval time offsets, 10 s apart. Training: the start time is
# uniform over [t_val_min, t_val_min + reset_offset_tmax) = [60, 90) s, so 2 s windows every 5 s.
EVAL_WINDOWS = dict(start_t_val_min=59, start_t_val_max=62, n_windows=4, window_step=10)
TRAIN_WINDOWS = dict(start_t_val_min=60, start_t_val_max=62, n_windows=6, window_step=5)
PANEL_STYLE = dict(point_size=1, wind_marker_size=10, wind_change_thresh=0.1, endpoint_size=5, goal_radius=0.2)


def _rotate(xy, angle_deg):
    """Rotate rows of `xy` counter-clockwise by angle_deg (the env's rotation convention).
    `angle_deg` is a scalar or one angle per row."""
    c, s = np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))
    return np.stack([c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]], axis=1)


def _wind_theta01(wind):
    """Ambient wind angle in the 0-1 units of log_analysis (angle / pi shifted into [0, 1])."""
    return (np.angle(wind[:, 0] + 1j * wind[:, 1]) / np.pi + 1) / 2


def _theta01_dt1(theta01, episode_start=None):
    """Step-to-step change of the 0-1 wind angle, wrapped into [-0.5, 0.5) so a wind blowing along
    -x (angle +-180 deg, theta01 jumping between 0 and 1 with the noise) does not read as a change on
    every step. 0 on the first step of each episode (`episode_start`: boolean mask; default row 0)."""
    dt1 = np.diff(theta01, prepend=theta01[:1])
    dt1 = (dt1 + 0.5) % 1.0 - 0.5
    if episode_start is None:
        dt1[:1] = 0.0
    else:
        dt1[episode_start] = 0.0
    return dt1


def episode_frame(logs, summaries):
    """One row per step of every eval episode, in the unrotated plume frame: loc, t_val, odor_01, wind change.

    Locations and wind vectors are rotated by -rotate_by (0 when the summaries have no rotate_by).
    `wind_angle_ground_theta_dt1` is the wrapped step-to-step change of the ambient wind angle (first step 0).
    """
    odor_threshold = config.env['odor_threshold']
    frames = []
    for log, summ in zip(logs, summaries):
        xy = np.asarray(log['trajectory'], dtype=float).reshape(-1, 2)
        infos = [rec[0] for rec in log['infos']]
        wind = np.asarray([info['ambient_wind'] for info in infos], dtype=float)
        angle = summ.get('rotate_by', 0.0) or 0.0
        if angle:
            xy = _rotate(xy, -angle)
            wind = _rotate(wind, -angle)
        theta01 = _wind_theta01(wind)
        frames.append(pd.DataFrame({
            'ep_idx': summ['idx'],
            'outcome': summ['reason'],
            'loc_x': xy[:, 0],
            'loc_y': xy[:, 1],
            't_val': [info['t_val'] for info in infos],
            'odor_01': (np.asarray([info['odor_obs'] for info in infos], dtype=float) > odor_threshold).astype(int),
            'wind_angle_ground_theta_dt1': _theta01_dt1(theta01),
        }))
    return pd.concat(frames, ignore_index=True)


class StepFrameBuffer:
    """Training-side source of the per-step frame, over the last `window_updates` updates.

    Per update: `start_update()`, then after every `envs.step` `add_step(infos)` and, for each env
    whose episode finished at that step, `mark_done(env_i, episode_id, info)`; finally
    `end_update(update_idx)` folds the update into a frame. `frame()` concatenates the window, with
    `dataset`, `rotate_by` and `update_idx` added to `FRAME_COLUMNS`. Episodes still running when the
    rollout ends have no outcome and are dropped. One update is ~16k rows / ~2 MB.
    """
    _KEY_STRIDE = 1_000_000  # step key = env index * stride + per-env episode serial

    def __init__(self, num_envs, window_updates=5):
        self.num_envs = num_envs
        self.frames = collections.deque(maxlen=window_updates)
        self._serial = np.zeros(num_envs, dtype=np.int64)
        self.start_update()

    def start_update(self):
        self._loc, self._t, self._odor, self._wind, self._key = [], [], [], [], []
        self._done = {}  # key -> (episode_id, outcome, rotate_by, dataset)
        self._serial[:] = 0

    def _keys(self):
        return np.arange(self.num_envs, dtype=np.int64) * self._KEY_STRIDE + self._serial

    def add_step(self, infos):
        self._loc.append(np.asarray([info['location'] for info in infos], dtype=np.float32).reshape(self.num_envs, 2))
        self._t.append(np.asarray([info['t_val'] for info in infos], dtype=np.float32))
        self._odor.append(np.asarray([info['odor_obs'] for info in infos], dtype=np.float32))
        self._wind.append(np.asarray([info['ambient_wind'] for info in infos], dtype=np.float32).reshape(self.num_envs, 2))
        self._key.append(self._keys())

    def mark_done(self, env_i, episode_id, info):
        """Call after `add_step` for the step on which env `env_i` finished its episode."""
        key = int(env_i * self._KEY_STRIDE + self._serial[env_i])
        self._done[key] = (int(episode_id), info['done'], float(info.get('rotate_by') or 0.0), info.get('dataset', ''))
        self._serial[env_i] += 1

    def end_update(self, update_idx):
        cols = FRAME_COLUMNS + ['dataset', 'rotate_by', 'update_idx']
        if not self._loc or not self._done:
            self.frames.append(pd.DataFrame(columns=cols))
            return self.frames[-1]
        # (T, N, ...) -> env-major, time-ordered rows, so every episode is one contiguous block
        loc = np.stack(self._loc).transpose(1, 0, 2).reshape(-1, 2)
        wind = np.stack(self._wind).transpose(1, 0, 2).reshape(-1, 2)
        t_val = np.stack(self._t).T.reshape(-1)
        odor = np.stack(self._odor).T.reshape(-1)
        key = np.stack(self._key).T.reshape(-1)
        done = pd.DataFrame.from_dict(self._done, orient='index', columns=['episode_id', 'outcome', 'rotate_by', 'dataset'])
        keep = np.isin(key, done.index.values)
        loc, wind, t_val, odor, key = loc[keep], wind[keep], t_val[keep], odor[keep], key[keep]
        ep = done.loc[key]
        angle = ep['rotate_by'].values
        loc = _rotate(loc, -angle)
        wind = _rotate(wind, -angle)
        dt1 = _theta01_dt1(_wind_theta01(wind), episode_start=np.r_[True, key[1:] != key[:-1]])
        df = pd.DataFrame({
            'ep_idx': int(update_idx) * 100_000 + ep['episode_id'].values,  # unique across the window
            'outcome': ep['outcome'].values,
            'loc_x': loc[:, 0], 'loc_y': loc[:, 1],
            't_val': t_val,
            'odor_01': (odor > config.env['odor_threshold']).astype(int),
            'wind_angle_ground_theta_dt1': dt1,
            'dataset': ep['dataset'].values,
            'rotate_by': angle,
            'update_idx': int(update_idx),
        })
        self.frames.append(df)
        return df

    def frame(self):
        frames = [f for f in self.frames if len(f)]
        if not frames:
            return pd.DataFrame(columns=FRAME_COLUMNS + ['dataset', 'rotate_by', 'update_idx'])
        return pd.concat(frames, ignore_index=True)

    def update_range(self):
        """(first, last) update index held in the window, or None when empty."""
        idx = [int(f['update_idx'].min()) for f in self.frames if len(f)]
        return (min(idx), max(idx)) if idx else None


def window_bounds(start_t_val_min, start_t_val_max, n_windows, window_step):
    return [(start_t_val_min + window_step * k, start_t_val_max + window_step * k) for k in range(n_windows)]


def assign_windows(df, bounds):
    """Return a copy of `df` with a `win` column: the window index of the episode's first t_val, -1 if none."""
    first_t = df.groupby('ep_idx')['t_val'].transform('first').values
    win = np.full(len(df), -1)
    for k, (lo, hi) in enumerate(bounds):
        win[(first_t > lo) & (first_t < hi)] = k
    df = df.copy()
    df['win'] = win
    return df


def _limits(d, goal_radius):
    """Axis limits covering the points of `d` and the goal circle, with a 3% margin."""
    if d.empty:
        x0, x1, y0, y1 = -1.0, 1.0, -1.0, 1.0
    else:
        x0, x1 = d['loc_x'].min(), d['loc_x'].max()
        y0, y1 = d['loc_y'].min(), d['loc_y'].max()
    if goal_radius:
        x0, x1 = min(x0, -goal_radius), max(x1, goal_radius)
        y0, y1 = min(y0, -goal_radius), max(y1, goal_radius)
    px, py = 0.03 * (x1 - x0 or 1), 0.03 * (y1 - y0 or 1)
    return x0 - px, x1 + px, y0 - py, y1 + py


def draw_window_panel(ax, g, legend=False, point_size=1, wind_marker_size=10, wind_change_thresh=0.1,
                      endpoint_size=5, goal_radius=0.2):
    """Draw the steps of one start-time window (`g`, rows of the frame) into `ax`."""
    if len(g):
        miss, hit = g['odor_01'] == 0, g['odor_01'] == 1
        ax.plot(g.loc[miss, 'loc_x'], g.loc[miss, 'loc_y'], ls='', marker='s', ms=point_size,
                color='black', alpha=0.5, rasterized=True)
        ax.plot(g.loc[hit, 'loc_x'], g.loc[hit, 'loc_y'], ls='', marker='*', ms=point_size,
                color='red', rasterized=True)
        wind = g['wind_angle_ground_theta_dt1'].abs() > wind_change_thresh
        ax.plot(g.loc[wind, 'loc_x'], g.loc[wind, 'loc_y'], ls='', marker='x',
                ms=wind_marker_size, color='blue')
        start = g.drop_duplicates('ep_idx', keep='first')
        end = g.drop_duplicates('ep_idx', keep='last')
        ax.plot(start['loc_x'], start['loc_y'], ls='', marker='o', ms=endpoint_size,
                mfc='none', mec='tab:green', mew=0.8, label='start' if legend else None)
        ax.plot(end['loc_x'], end['loc_y'], ls='', marker='^', ms=endpoint_size,
                mfc='none', mec='tab:purple', mew=0.8, label='end' if legend else None)
    if goal_radius:
        ax.add_patch(Circle((0, 0), goal_radius, fill=False, ec='black', lw=1, zorder=5))
    ax.set_aspect('equal', adjustable='box')


def plot_outcome_starts(logs, summaries, out_prefix, dataset='', seed='', outcomes=OUTCOMES,
                        start_t_val_min=59, start_t_val_max=62, n_windows=4, window_step=10,
                        point_size=1, wind_marker_size=10, wind_change_thresh=0.1, endpoint_size=5,
                        goal_radius=0.2, dpi=150):
    """Eval: write one figure per outcome (see the module docstring). Returns the file names."""
    style = dict(point_size=point_size, wind_marker_size=wind_marker_size, wind_change_thresh=wind_change_thresh,
                 endpoint_size=endpoint_size, goal_radius=goal_radius)
    bounds = window_bounds(start_t_val_min, start_t_val_max, n_windows, window_step)
    df = assign_windows(episode_frame(logs, summaries), bounds)
    n_win_all = df[df['win'] >= 0].drop_duplicates('ep_idx').groupby('win').size()   # episodes per window, all outcomes
    written = []
    for outcome in outcomes:
        d_out = df[df['outcome'] == outcome]
        n_total = d_out['ep_idx'].nunique()
        d_sel = d_out[d_out['win'] >= 0]
        if d_sel.empty:
            continue
        x0, x1, y0, y1 = _limits(d_sel, goal_radius)
        ncols = 2
        nrows = int(np.ceil(n_windows / ncols))
        col_w = 6
        row_h = col_w * (y1 - y0) / (x1 - x0) + 0.7
        fig, axs = plt.subplots(nrows, ncols, figsize=(col_w * ncols, row_h * nrows),
                                sharex=True, sharey=True, squeeze=False)
        for k, ax in enumerate(axs.ravel()):
            if k >= n_windows:
                ax.axis('off')
                continue
            lo, hi = bounds[k]
            g = d_out[d_out['win'] == k]
            draw_window_panel(ax, g, legend=(k == 0), **style)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            n_in, n_win = g['ep_idx'].nunique(), int(n_win_all.get(k, 0))
            pct = 100 * n_in / n_win if n_win else float('nan')
            ax.set_title(f"t_val ({lo}, {hi}): {n_in} / {n_total} {outcome} eps\n"
                         f"{pct:.1f}% {outcome} in window ({n_in}/{n_win})", fontsize=9)
        if axs[0, 0].get_legend_handles_labels()[0]:
            axs[0, 0].legend(loc='upper right', fontsize=7, markerscale=1.2)
        fig.suptitle(f"{dataset} | {seed}\n{outcome}, all rotations unrotated to 0°", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fname = f"{out_prefix}_{outcome}_unrotated.png"
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        written.append(fname)
        print("Saving", fname)
    return written


def draw_outcome_grid(axs, df, bounds, outcomes=OUTCOMES, label='', style=PANEL_STYLE, limits='panel'):
    """Training: fill the (len(outcomes) x len(bounds)) axes array `axs` from one dataset's frame.

    Rows are outcomes, columns start-time windows. Every panel keeps a 1:1 metre scale; with
    `limits='panel'` each one zooms to its own episodes (tick labels on every panel), with
    `limits='block'` all panels share the range of the windowed data so positions line up across
    the grid. `label` names the dataset on the first column.
    """
    df = assign_windows(df, bounds)
    n_win_all = df[df['win'] >= 0].drop_duplicates('ep_idx').groupby('win').size()
    block = _limits(df[df['win'] >= 0], style['goal_radius'])
    for r, outcome in enumerate(outcomes):
        d_out = df[df['outcome'] == outcome]
        n_total = d_out['ep_idx'].nunique()
        for k, (lo, hi) in enumerate(bounds):
            ax = axs[r, k]
            g = d_out[d_out['win'] == k]
            draw_window_panel(ax, g, legend=(r == 0 and k == 0), **style)
            x0, x1, y0, y1 = _limits(g, style['goal_radius']) if (limits == 'panel' and len(g)) else block
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            n_in, n_win = g['ep_idx'].nunique(), int(n_win_all.get(k, 0))
            pct = 100 * n_in / n_win if n_win else float('nan')
            ax.set_title(f"{outcome} | t0 ({lo:g}, {hi:g}) s\n{n_in}/{n_total} {outcome}; "
                         f"{pct:.0f}% of window (n={n_win})", fontsize=7)
            ax.tick_params(labelsize=5 if limits == 'panel' else 6)
            if k == 0:
                ax.set_ylabel(f"{label}\n{outcome}" if label else outcome, fontsize=7)
            elif limits != 'panel':
                ax.tick_params(labelleft=False)
            if r < len(outcomes) - 1 and limits != 'panel':
                ax.tick_params(labelbottom=False)
    if axs[0, 0].get_legend_handles_labels()[0]:
        axs[0, 0].legend(loc='upper right', fontsize=6, markerscale=1.2)


def radial_success_table(df, outcome_col='reason'):
    """HOME count, total and % per rotate_by (degrees), sorted by angle in [0, 360)."""
    by_angle = df.groupby('rotate_by')[outcome_col].agg(n='size', home=lambda r: (r == 'HOME').sum())
    by_angle['pct'] = 100 * by_angle['home'] / by_angle['n']
    by_angle['theta'] = np.deg2rad(by_angle.index.values.astype(float) % 360)
    return by_angle.sort_values('theta')


def _angle_label(theta):
    deg = np.rad2deg(theta)
    return f"{deg if deg <= 180 else deg - 360:g}°"


def draw_radial_success(ax, df, title='', outcome_col='reason', by=None, colors=None):
    """Polar `ax`: HOME success rate (%) against rotate_by (degrees, counter-clockwise like the env).

    `by` (e.g. 'dataset') draws one polygon per group, coloured from `colors`. Returns False (and
    writes a note into the axes) when the rows carry no `rotate_by`.
    """
    if 'rotate_by' not in df.columns or df['rotate_by'].isna().all():
        ax.text(0.5, 0.5, 'no rotate_by', transform=ax.transAxes, ha='center', va='center', fontsize=8)
        ax.set_title(title, fontsize=9, pad=14)
        return False
    df = df[df['rotate_by'].notna()]
    groups = list(df.groupby(by)) if by else [(None, df)]
    all_theta = np.unique(np.deg2rad(df['rotate_by'].astype(float).values % 360))
    for name, d in groups:
        tbl = radial_success_table(d, outcome_col)
        theta = np.append(tbl['theta'].values, tbl['theta'].values[0])   # close the loop
        r = np.append(tbl['pct'].values, tbl['pct'].values[0])
        color = (colors or {}).get(name) if name is not None else 'tab:blue'
        overall = 100 * tbl['home'].sum() / tbl['n'].sum()
        lbl = f"{name} ({overall:.0f}%, n={int(tbl['n'].sum())})" if name is not None else None
        ax.plot(theta, r, color=color, lw=2, marker='o', ms=4, label=lbl)
        ax.fill(theta, r, color=color, alpha=0.15)
        for th, val in zip(tbl['theta'].values, tbl['pct'].values):
            ax.annotate(f"{val:.0f}", (th, val), textcoords='offset points', xytext=(0, 6), ha='center',
                        fontsize=6, color=color)
    ax.set_ylim(0, 100)
    ax.set_xticks(all_theta)
    ax.set_xticklabels([_angle_label(a) for a in all_theta], fontsize=7)
    ax.tick_params(axis='y', labelsize=6)
    overall = 100 * (df[outcome_col] == 'HOME').mean()
    ax.set_title(f"{title}\nHOME % by rotate_by (overall {overall:.1f}%, n={len(df)})", fontsize=8, pad=12)
    if by:
        ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(-0.25, 1.12), frameon=False)
    return True


def plot_radial_success(summaries, out_fname, title='', dpi=150):
    """Eval: one polar figure. Needs the `rotate_by` column that `--eval_rotations` adds to the
    summaries; without it there is nothing to plot around the circle and the function returns None."""
    df = pd.DataFrame(summaries)
    if 'rotate_by' not in df.columns or df['rotate_by'].isna().all():
        return None
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5.5, 5.5))
    draw_radial_success(ax, df, title=title)
    fig.tight_layout()
    fig.savefig(out_fname, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print("Saving", out_fname)
    return out_fname
