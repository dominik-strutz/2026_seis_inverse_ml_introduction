"""The record stepper: its range and its starting value."""

import pytest

from ml_seis.stepperwidget import Stepper


def test_starts_at_the_first_sample():
    s = Stepper(40, label="record")
    assert (s.start, s.stop, s.value, s.label) == (1, 40, 1, "record")


def test_start_value_is_clamped_into_the_range():
    assert Stepper(10, value=99).value == 10
    assert Stepper(10, value=-3).value == 1
    assert Stepper(10, start=0, value=0).value == 0


def test_empty_range_is_refused():
    with pytest.raises(ValueError):
        Stepper(0)
