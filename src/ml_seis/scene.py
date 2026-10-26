"""The scene of the locator playground: stations, P-speed model, pick noise.

``Scene`` bundles what the physics needs to simulate and to invert picks: the
station coordinates, the P-speed model, the noise level, which stations work, and
the forward operator ``F(m_x, m_y)``. (``scenarios.py`` is the single synthetic
event of the first playground; this is the whole network of stations of the locator.)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from . import physics
from .velocity import Velocity, make_velocity


@dataclass(frozen=True)
class Scene:
    """Stations, P-speed model and pick noise; ``F(m_x, m_y)`` is the forward
    operator and ``live`` marks the stations that work (False: muted)."""

    xi_x: torch.Tensor
    xi_y: torch.Tensor
    velocity: Velocity
    sigma: float
    live: torch.Tensor
    F: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    @property
    def n_stations(self) -> int:
        return len(self.xi_x)


def make_scene(stations, velocity: str, field: int, sigma: float, muted=()) -> Scene:
    """A scene from station coordinates [[x, y], ...] (km), the name of the P-speed
    model (``"homogeneous"`` or ``"heterogeneous"``) and its random field, the pick
    noise (s) and the indices of muted stations."""
    xi_x, xi_y = torch.tensor(stations, dtype=torch.float64).reshape(-1, 2).T
    model = make_velocity(velocity, int(field))
    live = torch.ones(len(xi_x), dtype=torch.bool)
    live[[i for i in muted if i < len(xi_x)]] = False
    return Scene(xi_x, xi_y, model, float(sigma), live, model.travel_times(xi_x, xi_y))


def scene_from(config: dict) -> Scene:
    """The scene captured by the train button of the locator playground."""
    return make_scene(
        config["stations"],
        config["velocity"],
        config["field"],
        config["sigma"],
        config["muted"],
    )


def observe(scene: Scene, m) -> torch.Tensor:
    """Centred noisy picks at the scene's stations for sources ``m`` of shape (n, 2):
    ``physics.observe`` with the scene's operator, noise level and working stations."""
    return physics.observe(scene.F, m, scene.sigma, scene.live)


def centred_arrivals(scene: Scene, m) -> torch.Tensor:
    """Noise-free centred arrivals C F(m) at the scene's stations for sources ``m``
    of shape (..., 2); a muted station is left out of the mean and reads 0."""
    m = torch.as_tensor(m, dtype=torch.float64)
    F = scene.F(m[..., 0], m[..., 1])
    return (F - F[..., scene.live].mean(-1, keepdim=True)) * scene.live


def ray_paths(scene: Scene, source) -> torch.Tensor:
    """Ray paths from ``source`` (km) to every station of the scene's P-speed model,
    shape (N, steps, 2): straight lines for a constant speed, bent otherwise."""
    source = torch.as_tensor(source, dtype=torch.float64).tolist()
    return scene.velocity.rays(scene.xi_x, scene.xi_y, source).transpose(0, 1)
