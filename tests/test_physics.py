"""The physics module is what slides print, so it must stay self-contained."""

import inspect

import pytest
import torch

from ml_seis import physics
from ml_seis.slides import show_source

PUBLIC = [getattr(physics, name) for name in physics.__all__]


def test_public_api():
    names = {"center", "observe", "simulate_picks", "travel_times"}
    assert set(physics.__all__) == names


@pytest.mark.parametrize("function", PUBLIC, ids=lambda f: f.__name__)
def test_self_contained(function):
    """No closures, and only torch and sibling functions as globals."""
    used = inspect.getclosurevars(function)
    allowed = {"torch", *physics.__all__}
    assert not used.nonlocals
    assert set(used.globals) <= allowed
    assert "returns" in function.__doc__


def test_travel_times_shape_and_value():
    xi_x = torch.tensor([0.0, 30.0], dtype=torch.float64)
    xi_y = torch.tensor([0.0, 40.0], dtype=torch.float64)
    m = torch.zeros(5, dtype=torch.float64)
    F = physics.travel_times(m, m, xi_x, xi_y, v_ref=5.0)
    assert F.shape == (5, 2)
    assert torch.allclose(F[0], torch.tensor([0.0, 10.0], dtype=torch.float64))


def test_center_removes_origin_time():
    d = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float64)
    assert torch.allclose(physics.center(d), physics.center(d + 12.3))
    assert torch.allclose(physics.center(d).mean(-1), torch.zeros(1, dtype=d.dtype))


def test_simulate_picks_is_reproducible():
    xi = torch.tensor([10.0, 90.0, 40.0], dtype=torch.float64)
    m = torch.tensor(55.0, dtype=torch.float64)
    args = (m, m, xi, xi, 6.0, 0.1, 12.3)
    a = physics.simulate_picks(*args, generator=torch.Generator().manual_seed(0))
    b = physics.simulate_picks(*args, generator=torch.Generator().manual_seed(0))
    assert torch.equal(a, b)
    quiet = physics.simulate_picks(*args[:5], 0.0, 12.3)
    assert torch.allclose(quiet, 12.3 + physics.travel_times(m, m, xi, xi, 6.0))


def test_show_source_refuses_other_functions():
    show_source(physics.travel_times, physics.center)
    with pytest.raises(ValueError):
        show_source(inspect.getsource)


def test_observe_mutes_stations_and_centres():
    xi = torch.tensor([10.0, 90.0, 40.0, 70.0], dtype=torch.float64)
    live = torch.tensor([True, True, False, True])
    m = torch.tensor([[55.0, 45.0], [20.0, 80.0]], dtype=torch.float64)

    def F(m_x, m_y):
        return physics.travel_times(m_x, m_y, xi, xi, 6.0)

    d = physics.observe(F, m, 0.1, live, torch.Generator().manual_seed(0))
    assert d.shape == (2, 4)
    assert torch.all(d[:, 2] == 0)
    assert torch.allclose(d[:, live].mean(-1), torch.zeros(2, dtype=d.dtype))
