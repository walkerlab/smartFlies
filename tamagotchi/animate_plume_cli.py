"""Animate a plume dataset (puffs + wind vector) straight from its dataset name.

Reuses the per-frame renderers in `data_util` (`plot_puffs`, `plot_wind_vectors`)
and precomputes the per-frame puff slices / scatter size factor the same way
`data_util.make_animation_update` does, so the hot loop stays cheap.

Examples
--------
# 5 simulated seconds starting at t=60s, realtime, mp4
python -m tamagotchi.animate_plume_cli constantx5b5 --t_start 60 --duration 5

# whole dataset, 4x speedup, wider arena, gif (no ffmpeg needed)
python -m tamagotchi.animate_plume_cli switch45x5b5 --duration -1 --speed 4 \
    --xlim -1 10 --ylim -5 5 --out switch45.gif

In a notebook
-------------
from tamagotchi.animate_plume_cli import animate_plume
from IPython.display import HTML
anim, fig = animate_plume('constantx5b5', t_start=60, duration=5, out_fname=None)
HTML(anim.to_jshtml())
"""

import argparse
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

from tamagotchi import config
from tamagotchi import data_util


def animate_plume(dataset,
                  out_fname=None,
                  t_start=None,
                  duration=5.0,
                  t_end=None,
                  env_dt=0.04,
                  fps=25,
                  speed=1.0,
                  xlim=(-2, 12),
                  ylim=(-5, 5),
                  figsize=(10, 5),
                  dpi=100,
                  data_dir=config.datadir,
                  puff_sparsity=1.0,
                  radius_multiplier=1.0,
                  wind_vector=True,
                  invert_colors=False,
                  verbose=True):
    """Build (and optionally save) an animation of a plume dataset.

    Args:
        dataset: dataset name, i.e. `{data_dir}/puff_data_{dataset}.pickle`.
        out_fname: output path. `.mp4` needs ffmpeg, `.gif` uses Pillow.
                   None → don't save, just return the animation object.
        t_start: first simulated second to render (None → start of dataset).
        duration: how many simulated seconds to render; <0 → until the end.
        t_end: explicit end time; overrides `duration` when given.
        env_dt: dataset downsampling used by `load_plume` (0.02/0.04/0.05/0.10).
        fps: playback frame rate of the written video.
        speed: simulated seconds per playback second (1.0 = realtime).
               Frames are strided to hit this, so it never re-times the video.
        xlim/ylim: arena limits in meters.
        data_dir: where the pickles live (defaults to `config.datadir`).
        wind_vector: draw the wind direction indicator.
        invert_colors: dark background (white text/arrow).

    Returns:
        (anim, fig) — keep a reference to `anim` or it gets garbage collected.
    """
    if t_end is None and duration is not None and duration >= 0 and t_start is not None:
        t_end = t_start + duration

    data_puffs, data_wind = data_util.load_plume(
        dataset=dataset,
        t_val_min=t_start,
        t_val_max=t_end,
        env_dt=env_dt,
        puff_sparsity=puff_sparsity,
        radius_multiplier=radius_multiplier,
        data_dir=data_dir,
    )
    if data_puffs.empty:
        raise ValueError(f"No puffs in [{t_start}, {t_end}] for dataset '{dataset}'")

    # --- frame timeline: puff times, so puff/wind lookups can never mismatch --
    # (both dataframes sit on the same tidx grid after load_plume)
    puffs_by_time = {t: grp for t, grp in data_puffs.groupby('time')}
    frame_times = np.array(sorted(puffs_by_time.keys()))

    # `duration` without an explicit `t_start`: trim from the front of the data
    if t_end is None and duration is not None and duration >= 0:
        frame_times = frame_times[frame_times <= frame_times[0] + duration]

    # --- stride frames to get the requested playback speed -------------------
    stride = max(1, int(round(speed / (fps * env_dt))))
    frame_times = frame_times[::stride]
    if len(frame_times) == 0:
        raise ValueError("No frames left to render -- check t_start/duration/speed")
    if verbose:
        eff_speed = stride * env_dt * fps
        print(f"[animate_plume] {len(frame_times)} frames, "
              f"t=[{frame_times[0]:.2f}, {frame_times[-1]:.2f}]s, "
              f"stride={stride}, {fps} fps -> {eff_speed:.2g}x speed, "
              f"{len(frame_times)/fps:.1f}s of video")

    # --- figure --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fgcolor = 'white' if invert_colors else 'black'
    if invert_colors:
        fig.patch.set_facecolor('black')

    # scatter size factor: computed once instead of per frame (needs a draw)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fig.canvas.draw()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    scatter_size_factor = 6250 * ((bbox.width / 8.0) ** 2)

    t_zero = frame_times[0]

    def update(i):
        t_val = frame_times[i]
        ax.clear()
        if invert_colors:
            ax.set_facecolor('black')

        data_util.plot_wind_vectors(data_puffs, data_wind, t_val, ax,
                                    invert_colors=invert_colors,
                                    wind_vector=wind_vector)
        data_util.plot_puffs(puffs_by_time[t_val], t_val, ax=ax, fig=fig,
                             show=False,
                             scatter_size_factor=scatter_size_factor)

        # crosshair at the odor source
        ax.plot([0, 0], [-0.3, +0.3], linestyle=':', lw=2, c=fgcolor)
        ax.plot([-0.3, +0.3], [0, 0], linestyle=':', lw=2, c=fgcolor)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_title(f"{dataset}  t={t_val - t_zero:.2f}s", color=fgcolor)
        ax.set_xlabel('Arena length [m]', color=fgcolor)
        ax.set_ylabel('Arena width [m]', color=fgcolor)
        ax.tick_params(colors=fgcolor)
        for spine in ax.spines.values():
            spine.set_color(fgcolor)
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(frame_times),
                                   interval=1000.0 / fps, blit=False)

    if out_fname is not None:
        ext = os.path.splitext(out_fname)[1].lower()
        if ext == '.gif':
            writer = animation.PillowWriter(fps=fps)
        else:
            if not animation.FFMpegWriter.isAvailable():
                raise RuntimeError(
                    f"ffmpeg not found, cannot write '{out_fname}'. "
                    "Install ffmpeg or use a .gif output instead.")
            writer = animation.FFMpegWriter(fps=fps, bitrate=-1)
        outdir = os.path.dirname(os.path.abspath(out_fname))
        os.makedirs(outdir, exist_ok=True)
        anim.save(out_fname, writer=writer,
                  savefig_kwargs={'facecolor': fig.get_facecolor()})
        if verbose:
            print("Saved", out_fname)

    return anim, fig


def main():
    parser = argparse.ArgumentParser(
        description='Animate a plume dataset by name',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', type=str,
                        help="dataset name, e.g. 'constantx5b5' "
                             "(reads {data_dir}/puff_data_{dataset}.pickle)")
    parser.add_argument('--out', type=str, default=None,
                        help='output file; default {dataset}_plume.mp4 in --outdir')
    parser.add_argument('--outdir', type=str, default='.')
    parser.add_argument('--t_start', type=float, default=None,
                        help='first simulated second to render (default: dataset start)')
    parser.add_argument('--duration', type=float, default=5.0,
                        help='simulated seconds to render; negative = to the end')
    parser.add_argument('--t_end', type=float, default=None,
                        help='explicit end time; overrides --duration')
    parser.add_argument('--env_dt', type=float, default=0.04,
                        help='dataset timestep after downsampling (0.02/0.04/0.05/0.1)')
    parser.add_argument('--fps', type=int, default=25, help='playback fps')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='simulated seconds per playback second (1 = realtime)')
    parser.add_argument('--xlim', type=float, nargs=2, default=[-2, 12])
    parser.add_argument('--ylim', type=float, nargs=2, default=[-5, 5])
    parser.add_argument('--figsize', type=float, nargs=2, default=[10, 5])
    parser.add_argument('--dpi', type=int, default=100)
    parser.add_argument('--data_dir', type=str, default=config.datadir)
    parser.add_argument('--puff_sparsity', type=float, default=1.0)
    parser.add_argument('--radius_multiplier', type=float, default=1.0)
    parser.add_argument('--no_wind_vector', action='store_true')
    parser.add_argument('--invert_colors', action='store_true',
                        help='dark background')
    args = parser.parse_args()

    out_fname = args.out
    if out_fname is None:
        out_fname = os.path.join(args.outdir, f'{args.dataset}_plume.mp4')

    matplotlib.use('Agg')  # headless rendering
    animate_plume(
        dataset=args.dataset,
        out_fname=out_fname,
        t_start=args.t_start,
        duration=args.duration,
        t_end=args.t_end,
        env_dt=args.env_dt,
        fps=args.fps,
        speed=args.speed,
        xlim=tuple(args.xlim),
        ylim=tuple(args.ylim),
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        data_dir=args.data_dir,
        puff_sparsity=args.puff_sparsity,
        radius_multiplier=args.radius_multiplier,
        wind_vector=not args.no_wind_vector,
        invert_colors=args.invert_colors,
    )


if __name__ == '__main__':
    main()
