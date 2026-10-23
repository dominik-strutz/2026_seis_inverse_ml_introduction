"""Shared non-ML helpers for the course: geometry, velocity, priors and map plots.

Networks, losses, optimisers and training loops are deliberately *not* here; they
live in the notebooks.
"""

from .geometry import BOX, DEFAULT_SOURCE, DEFAULT_START, make_stations
from .priors import PRIORS, log_prior, sample_prior
from .scenarios import LocationProblem, make_problem
from .velocity import V_REF, Velocity, lookup, make_velocity, trace_rays

__all__ = [
    "BOX",
    "LocationProblem",
    "PRIORS",
    "V_REF",
    "Velocity",
    "DEFAULT_SOURCE",
    "DEFAULT_START",
    "log_prior",
    "lookup",
    "make_problem",
    "make_stations",
    "make_velocity",
    "sample_prior",
    "trace_rays",
]
