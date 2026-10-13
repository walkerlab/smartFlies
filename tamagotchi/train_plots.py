"""Per-update training figures.

`plot_trajectories` draws the sampled trajectories of `TrajectoryStorage` (training.py), one panel
per trajectory over its plume frame, rows = datasets. When the per-step frame of the last few
updates (`outcome_start_plots.StepFrameBuffer`) is passed as `unrotated_df`, the figure continues
below with, per dataset, the outcome x start-time-window grid of ALL episodes of those updates in
the unrotated plume frame (the training twin of evalCli's `*_unrotated.png`).

`plot_density_and_radial` puts the HOME-by-plume-density bars (left) and the HOME-by-rotation
polar plot (right) of the same episode rows into one file.
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tamagotchi import config
from tamagotchi import data_util as utils
from tamagotchi.eval import outcome_start_plots as osp


def get_index_by_dataset(envs, dataset):
    """Get indices of the remotes that loaded the provided dataset."""
    idx = []
    for k, v in envs.remote_directory.items():
        if v['dataset'] == dataset:
            idx.append(k)
    return idx


def draw_traj_panel(ax, fig, episode, dataset, traj_num, envs, odor_threshold=None):
    """One sampled trajectory over its plume frame (puffs at the episode's end t_val, wind arrow)."""
    trajectory = episode['trajectory']
    outcome = episode['outcome']
    color = 'green' if outcome == 'HOME' else 'red'
    ax.set_title(f"{dataset}\nTraj {traj_num} ({outcome.upper()})", fontsize=9, pad=10)

    if len(trajectory) == 0:
        ax.text(0.5, 0.5, 'Empty\ntrajectory', ha='center', va='center', transform=ax.transAxes,
                fontsize=10, alpha=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return

    # Load and prepare dataset for environmental context
    info = episode['episode_info']
    get_ds_at = get_index_by_dataset(envs, info['dataset'])[0]
    data_puffs = envs.get_attr_at(get_ds_at, 'data_puffs_all')[0]
    data_wind = envs.get_attr_at(get_ds_at, 'data_wind_all')[0]

    # Apply episode-specific transformations
    drop_idxs = data_puffs['puff_number'].unique()
    drop_idxs = pd.Series(drop_idxs).sample(frac=(1 - info['plume_density']))
    data_puffs = data_puffs.query("puff_number not in @drop_idxs")
    data_puffs = data_puffs[data_puffs['time'] == info['t_val']]
    data_puffs = utils.rotate_puffs_optimized(data_puffs, info['rotate_by'], False)
    data_wind = utils.rotate_wind_optimized(data_wind, info['rotate_by'], False)

    # Plot puffs and wind vectors
    utils.plot_puffs_and_wind_vectors(data_puffs, data_wind, info['t_val'], ax=ax, fig=fig, fname='',
                                      show=False, threshold=odor_threshold)

    # Home region: 0.2m radius around the odor source at the origin
    ax.add_patch(plt.Circle((0, 0), 0.2, facecolor='none', edgecolor='black', linewidth=1.2, zorder=4))

    # Plot the trajectory, one point per step coloured by odor
    x, y, odor_obs = zip(*trajectory)
    colors = np.where(np.array(odor_obs) > 0, 'yellow', 'magenta')
    ax.scatter(x, y, c=colors, s=5, alpha=0.8)

    # Mark start and end points
    ax.plot(x[0], y[0], 'o', color=color, markersize=8, alpha=0.9, markeredgecolor='black', markeredgewidth=1)
    ax.plot(x[-1], y[-1], 's', color=color, markersize=8, alpha=0.9, markeredgecolor='black', markeredgewidth=1)

    # Cap axis magnitude at 6 for "wide" datasets, but always keep +-2 around the origin in view
    if 'wide' in dataset.lower():
        xmin, xmax = max(min(x), -6), min(max(x), 6)
        ymin, ymax = max(min(y), -6), min(max(y), 6)
        xmin, xmax = min(xmin, -2), max(xmax, 2)
        ymin, ymax = min(ymin, -2), max(ymax, 2)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    ax.text(0.02, 0.98, f'Length: {len(trajectory)}', transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))


def plot_trajectories(traj_storage, envs, plt_path=None, title=None, figsize=None, args=None,
                      unrotated_df=None, windows=osp.TRAIN_WINDOWS, update_label='', grid_limits='panel',
                      save_path=None):
    """Sampled trajectories (rows = datasets, columns = trajectories), plus, when `unrotated_df` is
    given, the outcome x start-window grid of every episode in it, one block of rows per dataset.
    `update_label` (e.g. "updates 95-99") is shown on the grid's row labels; `grid_limits` is
    'panel' (each grid cell zooms to its own episodes) or 'block' (one range per dataset).
    `save_path` is the older name of `plt_path` (kept for the training_wind_obsver_v2 call site)."""
    if plt_path is None:
        plt_path = save_path
    if plt_path is None:
        raise ValueError("plot_trajectories needs plt_path (or save_path)")
    trajectories = traj_storage.get_trajectories()
    dataset_trajs = {}
    for dataset, outcomes in trajectories.items():
        all_trajs = []
        for outcome in traj_storage.possible_outcomes:
            for episode in outcomes.get(outcome, []):
                episode_data = episode.copy()
                episode_data['outcome'] = outcome
                all_trajs.append(episode_data)
        dataset_trajs[dataset] = all_trajs
    has_grid = unrotated_df is not None and len(unrotated_df) > 0
    datasets = list(dataset_trajs.keys())
    if has_grid:  # datasets seen only in the frame still get a grid block
        datasets += [ds for ds in unrotated_df['dataset'].unique() if ds not in dataset_trajs]
    max_trajs = max([len(t) for t in dataset_trajs.values()] + [0])
    if not datasets or (max_trajs == 0 and not has_grid):
        print("No trajectories to plot")
        return

    n_rows, n_cols = len(datasets), max(max_trajs, 1)
    bounds = osp.window_bounds(**windows)
    n_win, n_out = len(bounds), len(osp.OUTCOMES)
    odor_threshold = getattr(args, 'odor_threshold', None) if args is not None else None

    top_h = 5 * n_rows
    grid_h = 2.9 * n_out * n_rows if has_grid else 0
    if figsize is None:
        figsize = (max(4 * n_cols, 3.3 * n_win), top_h + grid_h)
    fig = plt.figure(figsize=figsize)
    if has_grid:
        gs = fig.add_gridspec(2, 1, height_ratios=[top_h, grid_h], hspace=0.1)
        gs_top = gs[0].subgridspec(n_rows, n_cols)
        gs_grid = gs[1].subgridspec(n_out * n_rows, n_win, hspace=0.4, wspace=0.08)
    else:
        gs_top = fig.add_gridspec(n_rows, n_cols)

    # Top: one panel per sampled trajectory
    for row, dataset in enumerate(datasets):
        trajs = dataset_trajs.get(dataset, [])
        for col in range(n_cols):
            ax = fig.add_subplot(gs_top[row, col])
            if col < len(trajs):
                draw_traj_panel(ax, fig, trajs[col], dataset, col + 1, envs, odor_threshold=odor_threshold)
            else:
                ax.set_title(f"{dataset}\n(No trajectory)", fontsize=9, pad=10, alpha=0.6)
                ax.text(0.5, 0.5, 'No trajectory\ncollected', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, alpha=0.4)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            if row == n_rows - 1:
                ax.set_xlabel('X position', fontsize=8)
            if col == 0:
                ax.set_ylabel('Y position', fontsize=8)
            ax.set_aspect('equal', adjustable='box')
            ax.tick_params(labelsize=7)

    # Bottom: per dataset, all episodes of the window by outcome (rows) and start time (columns)
    if has_grid:
        for row, dataset in enumerate(datasets):
            axs = np.array([[fig.add_subplot(gs_grid[row * n_out + r, k]) for k in range(n_win)]
                            for r in range(n_out)])
            d = unrotated_df[unrotated_df['dataset'] == dataset]
            label = f"{dataset}\n{update_label}\n{d['ep_idx'].nunique()} eps, unrotated" if update_label else dataset
            osp.draw_outcome_grid(axs, d, bounds, label=label, limits=grid_limits)

    if title:
        fig.suptitle(title, fontsize=16, y=0.995)
    try:
        with warnings.catch_warnings():  # best effort on the nested gridspecs; the layout is set above
            warnings.simplefilter('ignore', UserWarning)
            fig.tight_layout(rect=[0, 0, 1, 0.99] if title else None)
    except Exception as e:
        print(f"tight_layout skipped: {e}")
    fig.savefig(plt_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Trajectories plot saved to {plt_path}")


def plot_update_trajectories(traj_storage, envs, update_num):
    """Convenience function to plot trajectories for a specific update"""
    title = f"Individual Trajectories - Update {update_num}"
    plt_path = f"/src/tamagotchi/trajectories_update_{update_num}.png"
    plot_trajectories(traj_storage, envs, plt_path, title=title)


def draw_home_density(ax, eps_df):
    """HOME fraction by plume-density bin, one bar group per dataset, with the HOME count on each bar."""
    bins = np.linspace(eps_df['plume_density'].min(), eps_df['plume_density'].max(), 10)
    bin_width = bins[1] - bins[0]
    datasets = eps_df['dataset'].unique()
    offset_factor = 0.8 / len(datasets)
    for i, dataset in enumerate(datasets):
        subset = eps_df[eps_df['dataset'] == dataset].copy()
        subset['plume_bin'] = pd.cut(subset['plume_density'], bins=bins)
        grouped = subset.groupby('plume_bin', observed=False)
        success_rate = grouped['outcome'].apply(lambda x: (x == 'HOME').mean())
        n_success = grouped['outcome'].apply(lambda x: (x == 'HOME').sum())
        bin_centers = grouped['plume_density'].apply(lambda x: x.mean()).values
        bin_centers_shifted = bin_centers + (i - len(datasets) / 2) * offset_factor * bin_width
        color = config.dataset_colors.get(dataset)
        ax.bar(bin_centers_shifted, success_rate.values, width=offset_factor * bin_width,
               label=dataset, alpha=0.8, color=color)
        for x, y, n in zip(bin_centers_shifted, success_rate.values, n_success.values):
            if np.isfinite(x) and np.isfinite(y):
                ax.text(x, y + 0.02, str(n), ha='center', va='bottom', fontsize=8, color=color)
    ax.set_xlabel('Plume Density')
    ax.set_ylabel(f"HOME fraction; n = {len(eps_df)}")
    ax.set_ylim(0, 1.05)
    ax.legend(title='Dataset', fontsize=7, title_fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_density_and_radial(eps_df, plt_path, dpi=100):
    """One file: HOME by plume density (left) and HOME by rotate_by (right) from the episode rows
    `eps_df` (train_logs.EPISODE_COLUMNS; usually the last few updates concatenated)."""
    lo, hi = int(eps_df['update_idx'].min()), int(eps_df['update_idx'].max())
    rng = f"update {lo}" if lo == hi else f"updates {lo}-{hi}"
    fig = plt.figure(figsize=(10, 4.6))
    ax_density = fig.add_subplot(1, 2, 1)
    ax_radial = fig.add_subplot(1, 2, 2, projection='polar')
    if (eps_df['outcome'] == 'HOME').any():
        draw_home_density(ax_density, eps_df)
    else:
        ax_density.text(0.5, 0.5, 'no HOME episodes', transform=ax_density.transAxes, ha='center', va='center')
    ax_density.set_title(f"Success rate by plume density ({rng})", fontsize=9)
    osp.draw_radial_success(ax_radial, eps_df, title=rng, outcome_col='outcome', by='dataset',
                            colors=config.dataset_colors)
    fig.tight_layout()
    fig.savefig(plt_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return plt_path
