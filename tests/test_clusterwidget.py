"""The Part 2c cluster playground: what it sends, and what it refuses."""

from types import SimpleNamespace

import pytest
import torch

from ml_seis.clusterwidget import TONE_ORDER, TONES, ClusterNaming, cluster_tones

CLASSES = ("LP", "TR", "VT", "TC")
N_TRAIN, N_TEST, DRAWS = 60, 20, 3


def fake_fits(ks=(2, 3, 5)):
    generator = torch.Generator().manual_seed(0)
    fits = {}
    for k in ks:
        fits[k] = SimpleNamespace(
            centres=torch.randn(k, 4, generator=generator),
            assign_train=torch.arange(N_TRAIN) % k,
            assign_test=torch.arange(N_TEST) % k,
            names=[torch.arange(k) % 4 for _ in range(DRAWS)],
            asked=[torch.arange(5 * k) for _ in range(DRAWS)],
        )
    return fits


def make(fits=None, **kwargs):
    generator = torch.Generator().manual_seed(1)
    return ClusterNaming(
        CLASSES,
        torch.randn(N_TRAIN, 4, generator=generator),
        torch.arange(N_TRAIN) % 4,
        torch.randn(N_TEST, 4, generator=generator),
        torch.arange(N_TEST) % 4,
        fake_fits() if fits is None else fits,
        **kwargs,
    )


def test_tables_are_sent_once_for_every_k_and_draw():
    widget = make(k=3, index_train=torch.arange(N_TRAIN) + 100)
    assert widget.classes == list(CLASSES) and len(widget.colours) == 4
    assert [fit["k"] for fit in widget.fits] == [2, 3, 5]
    assert widget.samples == DRAWS and widget.k == 3 and widget.sample == 0
    fit = widget.fits[2]  # k = 5
    assert len(fit["centres"]) == 5 and len(fit["centres"][0]) == 4
    assert len(fit["assign"]) == N_TRAIN and len(fit["assign_test"]) == N_TEST
    assert len(fit["names"]) == DRAWS and len(fit["asked"][0]) == 25
    assert widget.train["index"][0] == 100 and widget.test["index"][0] == 0
    # viridis, one colour per cluster of this k: for the scatter and for the focus
    assert len(fit["pale"]) == len(fit["deep"]) == 5
    assert all(c.startswith("#") for c in fit["pale"] + fit["deep"])
    assert widget.fits[0]["deep"][0] == fit["deep"][0]  # the same start of the map
    assert widget.fits[0]["deep"][1] != fit["deep"][1]  # spread over this k


def lightness(colour):
    return sum(int(colour[i : i + 2], 16) for i in (1, 3, 5))


def test_every_class_has_its_tones_from_dark_to_light():
    widget = make(k=3)
    assert len(widget.shades) == len(CLASSES)
    for base, ramp in zip(widget.colours, widget.shades, strict=True):
        assert len(ramp) == len(TONES) and ramp[TONES.index(0.0)] == base
        assert [lightness(c) for c in ramp] == sorted(lightness(c) for c in ramp)
        assert all(c.startswith("#") and len(c) == 7 for c in ramp)


def test_clusters_that_share_a_name_get_different_tones():
    widget = make(k=3)
    fit = widget.fits[2]  # k = 5: names 0 1 2 3 0, so clusters 0 and 4 share a name
    assert len(fit["tones"]) == DRAWS and len(fit["tones"][0]) == 5
    tones = fit["tones"][0]
    assert tones[0] != tones[4] and tones[0] == TONE_ORDER[0]  # the lower number first
    assert [tones[j] for j in (1, 2, 3)] == [TONE_ORDER[0]] * 3  # alone: class colour


def test_tones_follow_the_size_of_the_cluster_and_run_out_gracefully():
    # six clusters with one name: the largest keeps the class colour, the tones repeat
    assign = torch.tensor([0] * 1 + [1] * 5 + [2] * 3 + [3] * 2 + [4] * 4 + [5] * 6)
    tones = cluster_tones(torch.zeros(6, dtype=torch.long), torch.arange(21), assign, 6)
    assert tones[5] == TONE_ORDER[0] and tones[1] == TONE_ORDER[1]
    assert tones[4] == TONE_ORDER[2] and tones[2] == TONE_ORDER[3]
    assert tones[3] == TONE_ORDER[0] and tones[0] == TONE_ORDER[1]  # wrapped round


def test_a_cluster_without_a_looked_up_record_has_no_name():
    assign = torch.tensor([0, 0, 1, 1, 2, 2])
    names = torch.tensor([0, 1, 1])
    assert cluster_tones(names, torch.tensor([0, 2]), assign, 3) == [1, 1, -1]


def test_start_value_must_be_a_fitted_k():
    assert make(k=3).k == 3
    with pytest.raises(ValueError, match="no fit for k = 4"):
        make(k=4)


def test_every_fit_needs_the_same_number_of_draws():
    fits = fake_fits()
    fits[3].names = fits[3].names[:2]
    fits[3].asked = fits[3].asked[:2]
    with pytest.raises(ValueError, match="same number of draws"):
        make(fits, k=2)


def test_only_k_and_sample_are_meant_to_come_back():
    """The notebook reads ``k`` and ``sample``; the sent tables are not edited back."""
    widget = make(k=2)
    widget.k, widget.sample = 5, 2
    assert (widget.k, widget.sample) == (5, 2)


def test_explore_text_travels_with_the_widget():
    pairs = [("Clusters", "What does k change?"), ("Sample", "Which five?")]
    assert make(k=2, explore=pairs).explore == [list(p) for p in pairs]
    assert make(k=2).explore == []
