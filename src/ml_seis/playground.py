"""Interface plumbing for the Lecture 1 optimiser playground.

The map editor (which also holds the settings in its header bar), reading the
scene back and sending the run to it. The objective, the optimiser and the
training loop are *not* here; they are written in the notebook.
"""

import marimo as mo
import torch

from .geometry import DEFAULT_SOURCE, DEFAULT_START, make_stations
from .mapwidget import MapEditor
from .plotting import STATUS_MESSAGES, evaluate_run
from .scenarios import make_problem
from .velocity import make_velocity


def make_editor(n_stations=5):
    """Draggable map pre-loaded with the first ``n_stations`` default stations,
    the source and the start."""
    xi_x, xi_y = make_stations(n_stations)
    return mo.ui.anywidget(
        MapEditor(
            stations=torch.stack([xi_x, xi_y], dim=1).tolist(),
            source=list(DEFAULT_SOURCE),
            start=list(DEFAULT_START),
        )
    )


def read_scene(editor):
    """Turn the dragged scene and the scene settings in the header into a
    synthetic problem and the optimiser start.

    Returns ``problem`` (stations ``xi_x``/``xi_y``, picks ``d_obs``, ``v_ref``,
    ``travel_times(m_x, m_y)``) and ``m_start``, a length-2 tensor in km.
    """
    xi_x, xi_y = torch.tensor(editor.value["stations"], dtype=torch.float64).T
    m_true = torch.tensor(editor.value["source"], dtype=torch.float64)
    m_start = torch.tensor(editor.value["start"], dtype=torch.float64)
    velocity = make_velocity(editor.value["velocity"], editor.value["field"])
    problem = make_problem(xi_x, xi_y, m_true, editor.value["sigma"], velocity)
    return problem, m_start


def optimiser_settings(editor):
    """The optimiser settings chosen in the header: method ("SGD" or "Adam"),
    step size, SGD momentum and number of steps."""
    v = editor.value
    return v["method"], v["lr"], v["momentum"], v["steps"]


def show_run(editor, problem, path, misfit):
    """Send the run (``path``: the list of iterates) to the maps and the loss
    curve; return a warning if it went wrong (else nothing). ``misfit(m)`` is
    the notebook's loss; it is evaluated along the path and on a grid here."""
    path = torch.stack(path)
    with torch.no_grad():
        loss = misfit(path).tolist()  # S at every point of the path
    run = evaluate_run(problem, path, loss, misfit)
    editor.show(problem.velocity, problem.rays, path, run.S_grid, run.m_grid, loss)
    if run.status != "ok":
        return mo.callout(STATUS_MESSAGES[run.status], kind="warn")
    return None
