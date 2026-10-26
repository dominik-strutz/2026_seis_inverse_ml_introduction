"""Default scene for the Lecture 1 optimiser playground, in the local map frame (km).

One deterministic, well spread network of ``N_STATIONS`` = 8 stations, a true source
and an optimiser start. Students then drag, add and remove everything on the map.
"""

import torch

BOX = 100.0  # side of the square map (km)
N_STATIONS = 8  # stations in the default network
MAX_STATIONS = 16  # most stations the playgrounds offer

DEFAULT_SOURCE = (60.0, 55.0)  # true source (km)
DEFAULT_START = (20.0, 25.0)  # optimiser start (km)


def _rand(*size):
    return torch.rand(*size, dtype=torch.float64)


def _dart_throwing(n, spacing, lo=5.0, hi=95.0, max_tries=20000):
    """Random points at least ``spacing`` apart (Poisson-disc-like)."""
    pts = []
    for _ in range(max_tries):
        p = lo + (hi - lo) * _rand(2)
        if all(torch.dist(p, q) >= spacing for q in pts):
            pts.append(p)
            if len(pts) == n:
                break
    pts = torch.stack(pts)
    return pts[:, 0], pts[:, 1]


def make_stations(n: int = N_STATIONS) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the default station coordinates ``(xi_x, xi_y)``, float64, in km:
    ``n`` irregular, well spread stations (deterministic).

    Stations come from one fixed sequence, so ``make_stations(5)`` is the first five
    of ``make_stations(8)``. The first 8 are the default network of the course;
    stations 9 to ``MAX_STATIONS`` follow at a smaller spacing.
    """
    if not 1 <= n <= MAX_STATIONS:
        raise ValueError(f"n must be between 1 and {MAX_STATIONS}")
    torch.manual_seed(7)
    xi_x, xi_y = _dart_throwing(N_STATIONS, spacing=25.0)
    if n > N_STATIONS:  # continue the same sequence, closer together
        pts = list(torch.stack([xi_x, xi_y], dim=1))
        for _ in range(20000):
            p = 5.0 + 90.0 * _rand(2)
            if all(torch.dist(p, q) >= 17.0 for q in pts):
                pts.append(p)
                if len(pts) == MAX_STATIONS:
                    break
        xi_x, xi_y = torch.stack(pts)[:, 0], torch.stack(pts)[:, 1]
    return xi_x[:n], xi_y[:n]
