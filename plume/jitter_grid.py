"""
Grid-based wind jitter for the puff plume simulator.

The baseline simulator (sim_utils.manual_integrator) gives every puff its own
independent draw of lateral noise, so two puffs sitting on top of each other
scatter in opposite directions and the plume is a smooth Gaussian spread.
Here the arena is instead cut into square cells that each carry their own
jitter, so puffs that share a cell are translocated *together* and the plume
develops coherent filaments and voids.

The jitter is the curl of a scalar streamfunction psi living on the cell
corners (nodes):

    u = +d(psi)/dy      v = -d(psi)/dx

Inside a cell psi is the bilinear interpolant of its four corner values, and
u, v are that interpolant's exact derivatives.  Two properties follow, and
they are the reason for going through psi instead of drawing (u, v) per cell
directly:

  * du/dx + dv/dy == 0 at every point, so the field cannot pile puffs up on
    the convergent side of a cell boundary nor thin them out on the divergent
    side.  Cell seams shear and swirl the plume, they never accumulate it.
  * the velocity component normal to a cell edge is continuous across it, so
    a puff crossing a seam does not jump.  (The tangential component may
    step; `smooth` > 0 damps that if it bothers you.)

Each node's psi is an Ornstein-Uhlenbeck process with standard deviation
psi_std and correlation time `tau`, integrated exactly, so eddies live for
~tau seconds rather than being redrawn white on every step.

Choosing `sigma` (m/s)
----------------------
A puff's lateral displacement variance grows like 2*D*t.  For this field
D = sigma^2 * tau_eff, where tau_eff = 1/(1/tau + |U|/cell) is shortened by
the mean wind sweeping the puff through cells.  The baseline per-puff white
noise instead gives D = wind_y_var^2 * dt / 2.  So to land near the baseline
plume width:

    sigma ~ wind_y_var * sqrt(dt / (2*tau_eff))

e.g. wind_y_var=0.5, dt=0.01, tau=0.5, cell=0.5, |U|=0.5 gives tau_eff=0.33
and sigma ~ 0.06.  Treat that as a starting point and calibrate against the
baseline datasets: sigma of the same order as wind_magnitude will tear the
plume apart and can reverse the local flow.
"""
import json
import numpy as np


def binomial_smooth(field, passes):
    """Separable [1 2 1]/4 smoothing of a 2D field, replicating at the edges."""
    for _ in range(passes):
        f = np.pad(field, 1, mode='edge')
        field = 0.25*f[:-2, 1:-1] + 0.5*f[1:-1, 1:-1] + 0.25*f[2:, 1:-1]
        f = np.pad(field, 1, mode='edge')
        field = 0.25*f[1:-1, :-2] + 0.5*f[1:-1, 1:-1] + 0.25*f[1:-1, 2:]
    return field


class Lattice():
    """Regular grid over the arena. Locates points and evaluates curl(psi).

    psi lives on the *nodes* (cell corners), so the field arrays are
    (nx+1, ny+1).  x_max/y_max are snapped up to a whole number of cells.
    """

    def __init__(self, x_min, x_max, y_min, y_max, cell):
        self.cell = float(cell)
        self.x_min = float(x_min)
        self.y_min = float(y_min)
        self.nx = max(1, int(np.ceil((x_max - self.x_min)/self.cell)))
        self.ny = max(1, int(np.ceil((y_max - self.y_min)/self.cell)))
        self.x_max = self.x_min + self.nx*self.cell
        self.y_max = self.y_min + self.ny*self.cell

    @property
    def shape(self):
        return (self.nx + 1, self.ny + 1)

    def locate(self, x, y):
        """Cell indices plus in-cell coordinates in [0,1].

        Points outside the lattice clamp onto the boundary cell rather than
        extrapolating, so a puff (or agent) that strays out sees the edge
        field instead of a runaway one.
        """
        fx = (np.asarray(x, dtype=np.float64) - self.x_min)/self.cell
        fy = (np.asarray(y, dtype=np.float64) - self.y_min)/self.cell
        i = np.clip(np.floor(fx), 0, self.nx - 1).astype(np.int64)
        j = np.clip(np.floor(fy), 0, self.ny - 1).astype(np.int64)
        s = np.clip(fx - i, 0.0, 1.0)
        t = np.clip(fy - j, 0.0, 1.0)
        return i, j, s, t

    def curl(self, psi, x, y):
        """(u, v) = (+dpsi/dy, -dpsi/dx) of the bilinear interpolant of psi."""
        i, j, s, t = self.locate(x, y)
        p00 = psi[i, j]
        p10 = psi[i + 1, j]
        p01 = psi[i, j + 1]
        p11 = psi[i + 1, j + 1]
        u = ((1.0 - s)*(p01 - p00) + s*(p11 - p10))/self.cell
        v = -((1.0 - t)*(p10 - p00) + t*(p11 - p01))/self.cell
        return u, v

    def to_meta(self):
        return {'x_min': self.x_min, 'x_max': self.x_max,
                'y_min': self.y_min, 'y_max': self.y_max,
                'cell': self.cell, 'nx': self.nx, 'ny': self.ny}

    @classmethod
    def from_meta(cls, meta):
        return cls(meta['x_min'], meta['x_max'], meta['y_min'], meta['y_max'],
                   meta['cell'])


class JitterGrid():
    """Evolving divergence-free jitter field, plus a tape of what it did.

    :sigma: std of the perturbation velocity, m/s (0 disables the field)
    :tau: eddy lifetime, seconds
    :smooth: binomial smoothing passes on psi; 0 = independent per-cell jitter
    :record_stride: tape every Nth sim step (N=1 keeps full 100Hz fidelity)
    """

    def __init__(self, sigma, tau, lattice, smooth=0, seed=None,
                 record_stride=1, sim_dt=0.01):
        self.lattice = lattice
        self.sigma = float(sigma)
        self.tau = float(tau)
        self.smooth = int(smooth)
        self.seed = seed
        self.sim_dt = float(sim_dt)
        self.record_stride = int(record_stride)
        self.rng = np.random.RandomState(seed)

        # Smoothing shrinks the noise variance; renormalize so that a call to
        # _noise() always delivers ~unit variance per node.
        self._noise_norm = 1.0
        if self.smooth > 0:
            draws = [binomial_smooth(self.rng.normal(size=self.lattice.shape),
                                     self.smooth) for _ in range(16)]
            self._noise_norm = 1.0/np.stack(draws).std()

        # psi -> velocity gain depends on cell size and smoothing, and varies
        # within a cell. Measure it once over the whole lattice so that
        # `sigma` means m/s no matter how the grid is configured.
        xs = self.rng.uniform(lattice.x_min, lattice.x_max, size=4096)
        ys = self.rng.uniform(lattice.y_min, lattice.y_max, size=4096)
        samples = []
        for _ in range(32):
            u, v = lattice.curl(self._noise(), xs, ys)
            samples.append(np.concatenate([u, v]))
        self.psi_std = self.sigma/np.concatenate(samples).std()

        self.psi = self.psi_std*self._noise()
        self._frames = []
        self._tidxs = []

    def _noise(self):
        z = self.rng.normal(size=self.lattice.shape)
        if self.smooth > 0:
            z = binomial_smooth(z, self.smooth)*self._noise_norm
        return z

    def step(self, dt):
        """Exact OU update: holds std at psi_std for any dt/tau ratio."""
        a = np.exp(-dt/self.tau)
        self.psi = a*self.psi + np.sqrt(1.0 - a*a)*self.psi_std*self._noise()

    def velocity(self, x, y):
        """Perturbation velocity at puff/agent positions (arrays welcome)."""
        return self.lattice.curl(self.psi, x, y)

    def record(self, tidx):
        """Tape the field that the *next* integration step will use.

        Frame `tidx` therefore holds the field that transported the puffs into
        their tidx-labelled positions -- exactly the convention a row of
        data_wind follows, so an agent reading frame tidx and wind row tidx
        senses the same wind that moved the plume it is standing in.
        """
        if tidx % self.record_stride == 0:
            self._frames.append(self.psi.astype(np.float32))
            self._tidxs.append(int(tidx))

    def to_meta(self):
        meta = self.lattice.to_meta()
        meta.update({'sigma': self.sigma, 'tau': self.tau,
                     'smooth': self.smooth, 'seed': self.seed,
                     'psi_std': float(self.psi_std), 'sim_dt': self.sim_dt,
                     'record_stride': self.record_stride})
        return meta

    def save(self, prefix, verbose=True):
        """Write {prefix}.npy (psi frames) + {prefix}.json (lattice+params)."""
        if not self._frames:
            raise ValueError("Nothing recorded; call record() inside the sim loop")
        tidxs = np.asarray(self._tidxs)
        steps = np.diff(tidxs)
        assert len(steps) == 0 or np.all(steps == steps[0]), \
            "recorded tidxs must be evenly spaced"
        psi = np.stack(self._frames)
        np.save(f'{prefix}.npy', psi)
        meta = self.to_meta()
        meta.update({'tidx_start': int(tidxs[0]),
                     'tidx_stride': int(steps[0]) if len(steps) else self.record_stride,
                     'n_frames': int(psi.shape[0])})
        with open(f'{prefix}.json', 'w') as fp:
            json.dump(meta, fp, indent=2, sort_keys=True)
        if verbose:
            print(f"[JitterGrid] saved {psi.shape} psi frames "
                  f"({psi.nbytes/1e6:.1f} MB) to {prefix}.npy")
        return meta


class WindField():
    """A saved jitter field, replayed as (tidx, x, y) -> (du, dv).

    Loaded with mmap by default: the array stays on disk and the OS page cache
    is shared between the training subprocesses, so N envs do not cost N
    copies of the field.
    """

    def __init__(self, psi, lattice, tidx_start, tidx_stride, meta=None):
        self.psi = psi
        self.lattice = lattice
        self.tidx_start = int(tidx_start)
        self.tidx_stride = int(tidx_stride)
        self.meta = meta or {}

    @classmethod
    def load(cls, prefix, mmap=True):
        with open(f'{prefix}.json') as fp:
            meta = json.load(fp)
        psi = np.load(f'{prefix}.npy', mmap_mode='r' if mmap else None)
        return cls(psi, Lattice.from_meta(meta), meta['tidx_start'],
                   meta['tidx_stride'], meta)

    @property
    def frame_dt(self):
        """Seconds between taped frames."""
        return self.meta.get('sim_dt', 0.01)*self.tidx_stride

    @property
    def tidx_max(self):
        return self.tidx_start + self.tidx_stride*(self.psi.shape[0] - 1)

    def frame_index(self, tidx):
        k = int(round((tidx - self.tidx_start)/float(self.tidx_stride)))
        return int(np.clip(k, 0, self.psi.shape[0] - 1))

    def lookup(self, tidx, x, y, flipx=1.0):
        """Perturbation wind at a time and place.

        Frame `tidx` is the field that moved the puffs into their tidx
        positions (see JitterGrid.record), so pair this with data_wind's row
        for the same tidx.  A tidx that was not taped (record_stride > 1) or
        that falls outside the run snaps to the nearest stored frame.

        :flipx: -1.0 mirrors the field about y=0, to stay consistent with a
            plume that was flipped across the x-axis (see plume_env.flipx).
            Only meaningful for a lattice symmetric in y.
        """
        psi = np.asarray(self.psi[self.frame_index(tidx)], dtype=np.float64)
        u, v = self.lattice.curl(psi, x, flipx*np.asarray(y, dtype=np.float64))
        return u, flipx*v
