"""Deterministic non-ML setup for the Lecture 1 optimiser playground."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .geometry import BOX
from .velocity import V_REF, Velocity

T0 = 12.3  # origin time (s): a nuisance parameter, removed by centring
_MAX_STATIONS = 512
# One fixed set of unit-normal pick errors; station i always receives noise[i], so
# moving the source or a station does not reshuffle the noise.
torch.manual_seed(0)
_NOISE = torch.randn(_MAX_STATIONS, dtype=torch.float64)


@dataclass(frozen=True)
class LocationProblem:
    """Tensors and physics needed to locate one synthetic earthquake."""

    xi_x: torch.Tensor
    xi_y: torch.Tensor
    d_obs: torch.Tensor
    m_true: torch.Tensor
    v_ref: float
    velocity: Velocity
    travel_times: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    rays: torch.Tensor
    box: float = BOX


def make_problem(
    xi_x: torch.Tensor,
    xi_y: torch.Tensor,
    m_true: torch.Tensor,
    sigma: float,
    velocity: Velocity,
) -> LocationProblem:
    """Synthetic P arrivals d_obs = t0 + F(m_true) + noise at the given stations.

    This stops before centring, defining a loss, choosing a batch or constructing an
    optimiser. Those operations stay visible in the notebook.
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    n = len(xi_x)
    if not 2 <= n <= _MAX_STATIONS:
        raise ValueError(f"need between 2 and {_MAX_STATIONS} stations")
    xi_x = torch.as_tensor(xi_x, dtype=torch.float64)
    xi_y = torch.as_tensor(xi_y, dtype=torch.float64)
    m_true = torch.as_tensor(m_true, dtype=torch.float64)
    travel_times = velocity.travel_times(xi_x, xi_y)
    d_obs = T0 + travel_times(m_true[0], m_true[1]) + sigma * _NOISE[:n]
    return LocationProblem(
        xi_x=xi_x,
        xi_y=xi_y,
        d_obs=d_obs,
        m_true=m_true,
        v_ref=V_REF,
        velocity=velocity,
        travel_times=travel_times,
        rays=velocity.rays(xi_x, xi_y, m_true.tolist()),
    )
