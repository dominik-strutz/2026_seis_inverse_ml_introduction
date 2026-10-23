"""Velocity models behind one call: ``travel_times(m_x, m_y)`` returns F.

With ``heterogeneity = 0`` the P speed is constant and F is the analytic straight-ray
travel time. Otherwise the speed is a **Gaussian random field** (smooth, irregular,
correlation length ``CORRELATION`` km), v = v_ref exp(h u) with u of unit standard
deviation, so one number ``h`` sets the strength. Travel times then come from an
**eikonal solve** (fast marching, one per station, using reciprocity: the solve starts
at the station and reads the time at every possible source). The result is tabulated on
a grid and read back with **bicubic interpolation**, so the lookup is smooth and
differentiable.

The table stores the mean slowness q_i(m) = T_i(m) / ||m - xi_i||, so that
F_i = ||m - xi_i|| q_i. Unlike T_i itself, q_i has no cone-shaped kink at the
station, so interpolation stays accurate right up to it.

Tables are cached per station, so moving or adding one station costs one solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as nnf

from .geometry import BOX

V_REF = 6.0  # background P speed (km/s)
CORRELATION = 15.0  # correlation length of the random field (km)
HETEROGENEITY = 0.25  # std of log(v / v_ref) of the heterogeneous model
SEED = 5  # default random field (chosen to show local minima)
_GRID = 101  # eikonal grid nodes per side (1 km spacing on the default map)
_FIELDS: dict = {}  # unit-variance random fields by seed
_TABLES: dict = {}  # solved per-station tables, shared so re-creating a model is free


def unit_field(seed: int, grid: int = _GRID) -> torch.Tensor:
    """Gaussian random field with zero mean and unit variance, shape (grid, grid)."""
    if (seed, grid) not in _FIELDS:
        n = 2 * grid  # padded, so the periodic FFT does not wrap the map onto itself
        gen = torch.Generator().manual_seed(1000 + seed)
        white = torch.randn(n, n, generator=gen, dtype=torch.float64)
        k = torch.fft.fftfreq(n, d=BOX / (grid - 1), dtype=torch.float64) * 2 * np.pi
        k2 = k[:, None] ** 2 + k[None, :] ** 2
        amplitude = torch.exp(-k2 * CORRELATION**2 / 8)  # sqrt of a Gaussian spectrum
        u = torch.fft.ifft2(torch.fft.fft2(white) * amplitude).real[:grid, :grid]
        _FIELDS[(seed, grid)] = (u - u.mean()) / u.std()
    return _FIELDS[(seed, grid)]


@dataclass
class Velocity:
    """A P-speed model on the map; build one with :func:`make_velocity`."""

    heterogeneity: float = 0.0  # standard deviation of log(v / v_ref)
    seed: int = 0
    v_ref: float = V_REF
    box: float = BOX

    @property
    def constant(self) -> bool:
        return self.heterogeneity == 0

    # --- the field itself ---------------------------------------------------
    def speed_grid(self) -> torch.Tensor:
        """P speed (km/s) on the eikonal grid, shape (grid, grid) indexed [y, x]."""
        if self.constant:
            return torch.full((_GRID, _GRID), self.v_ref, dtype=torch.float64)
        return self.v_ref * torch.exp(self.heterogeneity * unit_field(self.seed))

    def speed(self, x, y):
        """P speed (km/s) at points (bilinear read of the grid)."""
        x, y = torch.broadcast_tensors(
            *(torch.as_tensor(a, dtype=torch.float64) for a in (x, y))
        )
        v = lookup(self.speed_grid()[None], x, y, self.box, mode="bilinear")
        return v[:, 0].reshape(x.shape)

    # --- forward operator ---------------------------------------------------
    def _station_table(self, sx: float, sy: float) -> torch.Tensor:
        """Mean slowness q for one station from one eikonal solve: (grid, grid)."""
        import skfmm

        key = (self.seed, self.heterogeneity, round(sx, 3), round(sy, 3))
        if key not in _TABLES:
            if len(_TABLES) > 4000:
                _TABLES.clear()
            grid, h = _GRID, self.box / (_GRID - 1)
            nodes = np.linspace(0, self.box, grid)
            gx, gy = np.meshgrid(nodes, nodes)  # [iy, ix]
            speed = self.speed_grid().numpy()
            r0 = 2 * h  # the solve starts on a small circle around the station
            dist = np.hypot(gx - sx, gy - sy)
            s_station = 1 / float(self.speed(torch.tensor(sx), torch.tensor(sy)))
            t = skfmm.travel_time(dist - r0, speed, dx=h, order=2) + r0 * s_station
            q = np.where(dist < 3 * h, s_station, t / np.maximum(dist, 1e-9))
            _TABLES[key] = torch.from_numpy(q)
        return _TABLES[key]

    def travel_times(self, xi_x, xi_y):
        """Return F(m_x, m_y): travel times (s) from a source to every station.

        ``m_x`` and ``m_y`` are tensors of any common shape ``S``; the result has
        shape ``S + (N,)``. Gradients flow to ``m_x`` and ``m_y``.
        """
        xi_x = torch.as_tensor(xi_x, dtype=torch.float64)
        xi_y = torch.as_tensor(xi_y, dtype=torch.float64)
        n = len(xi_x)
        table = None
        if not self.constant:
            table = torch.stack(
                [
                    self._station_table(sx, sy)
                    for sx, sy in zip(xi_x.tolist(), xi_y.tolist(), strict=True)
                ]
            )

        def travel_times(m_x, m_y):
            m_x, m_y = torch.broadcast_tensors(
                *(torch.as_tensor(a, dtype=torch.float64) for a in (m_x, m_y))
            )
            shape = m_x.shape
            delta2 = (m_x[..., None] - xi_x) ** 2 + (m_y[..., None] - xi_y) ** 2
            distance = torch.sqrt(delta2 + 1e-12)  # keeps the gradient finite at xi
            if table is None:
                return distance / self.v_ref
            return distance * lookup(table, m_x, m_y, self.box).reshape(*shape, n)

        return travel_times

    def rays(self, xi_x, xi_y, source):
        """Ray paths from ``source`` to every station, (n_steps + 1, N, 2)."""
        F = self.travel_times(xi_x, xi_y)
        return trace_rays(F, xi_x, xi_y, source).detach()


def lookup(table, m_x, m_y, box: float = BOX, mode: str = "bicubic"):
    """Read per-station tables (N, grid, grid) at source positions: shape (P, N)."""
    n = table.shape[0]
    gx = (2 * m_x.reshape(-1) / box - 1).expand(n, -1)
    gy = (2 * m_y.reshape(-1) / box - 1).expand(n, -1)
    grid = torch.stack([gx, gy], dim=-1)[:, :, None, :]  # (N, P, 1, 2)
    q = nnf.grid_sample(
        table[:, None], grid, mode=mode, padding_mode="border", align_corners=True
    )
    return q[:, 0, :, 0].T  # (P, N)


def trace_rays(travel_times, xi_x, xi_y, source, step: float = 1.0, n_steps: int = 200):
    """Ray paths from ``source`` to every station, shape (n_steps + 1, N, 2).

    A ray follows the steepest descent of the station's travel-time field, so rays
    bend towards fast regions; in a constant model they are straight lines.
    """
    xi = torch.stack([xi_x, xi_y], dim=-1)
    p = torch.as_tensor(source, dtype=torch.float64).expand(len(xi), 2).clone()
    path = [p]
    for _ in range(n_steps):
        q = p.clone().requires_grad_()
        (g,) = torch.autograd.grad(travel_times(q[:, 0], q[:, 1]).diagonal().sum(), q)
        left = (p - xi).norm(dim=-1, keepdim=True)
        p = p - left.clamp(max=step) * g / g.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        path.append(p)
    return torch.stack(path)


def make_velocity(kind: str = "homogeneous", seed: int = SEED) -> Velocity:
    """``"homogeneous"`` is a constant P speed; ``"heterogeneous"`` is a random
    Gaussian field (strength ``HETEROGENEITY``) drawn with ``seed``."""
    if kind == "homogeneous":
        return Velocity(0.0, seed)
    if kind == "heterogeneous":
        return Velocity(HETEROGENEITY, seed)
    raise ValueError(f"unknown velocity model {kind!r}")
