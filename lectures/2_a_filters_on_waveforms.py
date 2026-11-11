# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "anywidget>=0.9",
#     "h5py>=3.11",
#     "marimo>=0.25.2",
#     "matplotlib>=3.9",
#     "numpy>=2",
#     "scipy>=1.14",
#     "torch>=2.5",
#     "ml-seis",
# ]
#
# [tool.uv.sources]
# ml-seis.git = "https://github.com/dominik-strutz/2026_seis_inverse_ml_introduction"
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="full",
    layout_file=(
        "data:application/json;base64,eyJ0eXBlIjoic2xpZGVzIiwiZGF0YSI6eyJjZWxscyI6W"
        "3sidHlwZSI6InNraXAifSx7InR5cGUiOiJzbGlkZSIsInNob3dDb2RlIjpmYWxzZX0seyJ0eXB"
        "lIjoic2tpcCJ9LHsidHlwZSI6InNraXAifSx7InR5cGUiOiJza2lwIn0seyJ0eXBlIjoic2tpc"
        "CJ9LHsidHlwZSI6InNsaWRlIiwic2hvd0NvZGUiOmZhbHNlfSx7InR5cGUiOiJza2lwIn0seyJ"
        "0eXBlIjoic2xpZGUiLCJzaG93Q29kZSI6ZmFsc2V9LHsidHlwZSI6InNraXAifSx7InR5cGUiO"
        "iJza2lwIn0seyJ0eXBlIjoic2tpcCJ9LHsidHlwZSI6InNsaWRlIiwic2hvd0NvZGUiOmZhbHN"
        "lfSx7InR5cGUiOiJzbGlkZSIsInNob3dDb2RlIjpmYWxzZX0seyJ0eXBlIjoic2tpcCJ9LHsid"
        "HlwZSI6InNraXAifSx7InR5cGUiOiJzbGlkZSIsInNob3dDb2RlIjpmYWxzZX0seyJ0eXBlIjo"
        "ic2tpcCJ9LHsidHlwZSI6InNraXAifSx7InR5cGUiOiJzbGlkZSIsInNob3dDb2RlIjpmYWxzZ"
        "X0seyJ0eXBlIjoic2tpcCJ9LHsidHlwZSI6ImZyYWdtZW50Iiwic2hvd0NvZGUiOmZhbHNlfSx"
        "7InR5cGUiOiJzbGlkZSIsInNob3dDb2RlIjpmYWxzZX0seyJ0eXBlIjoiZnJhZ21lbnQiLCJza"
        "G93Q29kZSI6ZmFsc2V9LHsidHlwZSI6InNsaWRlIiwic2hvd0NvZGUiOmZhbHNlfSx7InR5cGU"
        "iOiJza2lwIn0seyJ0eXBlIjoiZnJhZ21lbnQiLCJzaG93Q29kZSI6ZmFsc2V9LHsidHlwZSI6I"
        "nNsaWRlIiwic2hvd0NvZGUiOmZhbHNlfSx7InR5cGUiOiJmcmFnbWVudCIsInNob3dDb2RlIjp"
        "mYWxzZX0seyJ0eXBlIjoic2tpcCJ9LHsidHlwZSI6InNraXAifSx7InR5cGUiOiJzbGlkZSIsI"
        "nNob3dDb2RlIjpmYWxzZX1dLCJkZWNrIjp7InRyYW5zaXRpb24iOiJub25lIiwidmVydGljYWx"
        "BbGlnbiI6InRvcCJ9fX0="
    ),
    css_file="lecture_slides.css",
    html_head_file="slides_head.html",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from ml_seis import waveform_metrics as metrics
    from ml_seis import waveform_plotting as plots
    from ml_seis.convwidget import ConvFilterPlayground
    from ml_seis.llaimawidget import LlaimaExplorer
    from ml_seis.slides import artifact, try_it
    from ml_seis.waveform_data import (
        LLAIMA_NAMES,
        llaima_cache_path,
        load_llaima,
        prepare_llaima,
    )
    from ml_seis.waveform_setup import (
        cache_ready,
        default_cache_dir,
        llaima_card,
        shape_card,
    )

    return (
        ConvFilterPlayground,
        F,
        LLAIMA_NAMES,
        LlaimaExplorer,
        artifact,
        cache_ready,
        default_cache_dir,
        llaima_cache_path,
        llaima_card,
        load_llaima,
        metrics,
        mo,
        nn,
        np,
        plots,
        prepare_llaima,
        shape_card,
        torch,
        try_it,
    )


@app.cell(hide_code=True)
def _(artifact, mo):
    mo.vstack(
        [
            mo.md(r"""
            # Part 2a: filters on waveforms

            ### Machine learning and inversion for seismologists

            https://github.com/dominik-strutz/2026_seis_inverse_ml_introduction
            """),
            artifact("title_figure_waveforms", width="65%"),
        ],
        gap=0,
    )
    return


@app.cell
def _(default_cache_dir, mo):
    cache_dir = default_cache_dir(mo.notebook_dir())
    mo.md(f"Downloaded data are cached in `{cache_dir}`.")
    return (cache_dir,)


@app.cell(hide_code=True)
def _(cache_dir, cache_ready, llaima_cache_path, mo):
    llaima_cache = llaima_cache_path(cache_dir)
    llaima_cached = cache_ready(llaima_cache)
    llaima_button = mo.ui.run_button(label="Download and prepare the Llaima data")
    # With the data on disk nothing is shown here.
    mo.vstack([] if llaima_cached else [llaima_button])
    return llaima_button, llaima_cache, llaima_cached


@app.cell(hide_code=True)
def _(
    cache_dir,
    llaima_button,
    llaima_cached,
    load_llaima,
    mo,
    prepare_llaima,
):
    mo.stop(
        not (llaima_cached or llaima_button.value),
        mo.md("*The Llaima records are not on this machine yet. Press the button.*"),
    )
    with mo.status.progress_bar(
        total=100, title="Downloading the Llaima records", remove_on_exit=True
    ) as _bar:
        _seen = [0]

        def _download_progress(fraction):
            _step = int(100 * fraction) - _seen[0]
            _seen[0] += _step
            if _step > 0:
                _bar.update(increment=_step)

        prepare_llaima(cache_dir, _download_progress)
        _bar.update(increment=0, subtitle="Preparing the records...")
        llaima = load_llaima(cache_dir)
    return (llaima,)


@app.cell(hide_code=True)
def _(LLAIMA_NAMES, llaima, llaima_cache, llaima_card, mo):
    mo.vstack(
        [
            llaima_card(mo, llaima, llaima_cache, LLAIMA_NAMES),
            mo.md(
                f"A batch of records is a tensor of shape `(batch, channel, sample)`: "
                f"the whole dataset is `{tuple(llaima.u.shape)}`."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(llaima, mo):
    mo.md(f"""
    ## Volcano records with labels

    * {len(llaima.label):,} records from one station on **Llaima volcano,
    Chile**.
    * vertical component, band-passed between 1 and 10 Hz
    * {llaima.u.shape[-1] / llaima.sampling_rate:.0f} s long at
    {llaima.sampling_rate:.0f} Hz => $T$ = {llaima.u.shape[-1]} samples.


    An expert at the observatory gave each record one of four labels:

    - **LP**: long-period events: fluid and gas moving, mostly low frequencies,
    - **TR**: tremor: a long, steady signal,
    - **VT**: volcano-tectonic events: brittle failure inside the volcano, like small
      earthquakes,
    - **TC**: tectonic earthquakes: regional events.
    """)
    return


@app.cell
def _(LlaimaExplorer, llaima, mo):
    explorer = mo.ui.anywidget(LlaimaExplorer(llaima))
    return (explorer,)


@app.cell(hide_code=True)
def _(explorer, mo):
    mo.vstack(
        [
            explorer,
            mo.md(r"""
    ### Three things to look for

    - **Frequency.** Which classes live in which frequency bands, and which overlap?
    - **Padding.** How much of a record is flat, and does it depend on the class?
    - **Balance.** How common is each class, and how are the records split?
        """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How the explorer works

    Each row is a class and each column a record, drawn as its waveform. The
    bars on the left count the records of a class, split into train, validation and
    test, and the train, validation and test chips filter the grid. The page buttons
    (or the arrow keys) show the next records, *shuffle* draws a new order, and a
    click on a record opens it in the panel below as its spectrum against the class
    spectra, its spectrogram, or its waveform (drag to zoom, click to reset).
    The tag under a record gives its *flat* share, the fraction of samples that
    repeat the previous one.
    """)
    return


@app.cell
def _(llaima, metrics):
    # per-class means over all records, for the numbers in the tasks
    _f = metrics.record_features(llaima.u[:, 0], llaima.sampling_rate)
    class_stats = {
        name: {
            key: float(_f[key][(llaima.label == k).numpy()].mean())
            for key in ("flat", "f_mean")
        }
        for k, name in enumerate(llaima.classes)
    }
    majority = int(llaima.counts().argmax())
    return class_stats, majority


@app.cell(hide_code=True)
def _(class_stats, llaima, majority, try_it):
    _fast = max(class_stats, key=lambda c: class_stats[c]["f_mean"])
    _slow = min(class_stats, key=lambda c: class_stats[c]["f_mean"])
    _share = 100 * llaima.counts()[majority] / len(llaima.label)
    try_it(
        [
            (
                "Frequency",
                "Which class has the highest typical frequency and which the lowest?",
                "Spectrum panel, open one record of each class and compare it with "
                "the class spectra.",
                f"On average `{_fast}` is highest ({class_stats[_fast]['f_mean']:.1f} "
                f"Hz) and `{_slow}` lowest ({class_stats[_slow]['f_mean']:.1f} Hz), "
                "but `LP` and `TC` overlap a lot, so frequency alone will not "
                "separate every class.",
            ),
            (
                "Padding",
                "Which classes are mostly flat padding?",
                "Page through a few records of every class and read the flat share "
                "under each record.",
                "Averaged over all records: "
                + ", ".join(
                    f"`{c}` {100 * class_stats[c]['flat']:.0f}%" for c in class_stats
                )
                + ". The padding differs between classes, which a network could "
                "exploit.",
            ),
            (
                "Balance",
                "How well does a network that always names the commonest class do?",
                "Read the bars, with and without the validation and test chips.",
                f"`{llaima.classes[majority]}` is {_share:.0f}% of all records, so "
                "always answering with it is right that often; `VT` is the rarest.",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(llaima, mo):
    _test = llaima.counts("test")
    mo.md(f"""
    ## Some disclaimers

    - The labels are **one expert's judgement**, not ground truth, and records may
      overlap.
    - The source gives no event identity or time, so near-duplicate events may sit in
      both training and test data. We cannot rule that out.
    - Short events are padded with a flat line: a network could partly classify by
      *how much is flat* rather than by the signal.
    - The classes are **imbalanced**: always answering with the commonest class would
      be right {100 * llaima.counts().max() / len(llaima.label):.0f}% of the time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A filter slides along the record

    * A **filter** is a short list of $K$ weights $w_0, \dots, w_{K-1}$ (a template).
    * At every position $t$ it takes a weighted sum of the $K$ samples of the record
    $\mathbf{u}$ under it:

    $$
    h_t = b + \sum_{\tau=0}^{K-1} w_\tau \, u_{t+\tau}
    $$

    * The same weights are used at every $t$ (*weight sharing*).
    * The output $\mathbf{h}$ is the **feature map**: large where the record looks
      like the template.

    **You know this operation.** The sum is the cross-correlation of the record with
    the template (a matched filter):

    $$
    h_t - b = \sum_\tau w_\tau \, u_{t+\tau} = (w \star u)_t
    $$


    $\mathbf{w}$ is not designed but **learned**.
    """)
    return


@app.cell
def _(F, llaima, mo, torch):
    # one filter, one record: the whole sliding is one call
    record = llaima.u[[llaima.pick("train", "TC")]]  # shape (1, 1, 3008): one record
    edge_filter = torch.tensor([[[-1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0]]]) / 3
    feature_map = F.conv1d(record, edge_filter)  # shape (1, 1, 3002)
    mo.md(
        f"The filter has shape `(out_channels, in_channels, K)`. "
        f"`{tuple(record.shape)}` → `{tuple(feature_map.shape)}`: the output is "
        f"$K-1$ samples shorter, because the filter has to fit inside the record. "
        f"The layers below pad the record so that the length stays the same."
    )
    return


@app.cell
def _(ConvFilterPlayground, llaima, mo):
    conv_playground = mo.ui.anywidget(ConvFilterPlayground(llaima))
    return (conv_playground,)


@app.cell(hide_code=True)
def _(conv_playground, mo):
    mo.vstack(
        [
            conv_playground,
            mo.md(r"""
    ### Four things to look for

    - **Template.** What kind of pattern does each preset respond to, and where on a
      record is its feature map large?
    - **Length.** What does the filter length $K$ change?
    - **Frequency band.** Which wiggles does a filter keep and which does it remove?
    - **Classes.** Can one filter respond differently to different classes?
        """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How the playground works

    The left column shows the whole record $\mathbf{u}$ with its feature map
    $\mathbf{h}$ underneath, and a zoom on the shaded window. **Drag** along any of
    these traces to move the filter, or press play. The right column is the filter:
    the row of presets sets the weights (*smooth*, *high-pass*, *band-pass*, edges,
    *slope*, *bump*), and **dragging the bars** draws your own.

    The bottom right shows the arithmetic at the marked position: one column per
    weight $w_\tau$ (its bar is the weight) and under it the product
    $w_\tau u_{t+\tau}$ with the sample under that weight. The products add up to
    $h_t$ (with the bias $b$). Point at a column to ring its sample in the zoom and
    read its numbers. The buttons choose the filter length $K$ and the record (the
    class and the next or previous record of it). The bias $b$ shifts the feature
    map up or down, and ReLU (on) cuts away its negative values: both come back with
    the network below.
    """)
    return


@app.cell(hide_code=True)
def _(try_it):
    try_it(
        [
            (
                "Template",
                "How do *slope* and *bump* respond to the same stretch of record?",
                "Preset *slope*, then *bump*, same record and position; scrub the "
                "filter along a record.",
                "*Slope* is large on steady ramps and *bump* is large at an isolated "
                "peak and zero on a ramp. Where the record is exactly flat (padding) "
                "every preset gives zero, so the feature map is also flat.",
            ),
            (
                "Length",
                "What does a longer filter do to a smoothing filter's feature map?",
                "Preset *smooth*, $K$ = 3, then 15, on an `LP` record.",
                "At $K$ = 3 the feature map is almost a copy of the record. At "
                "$K$ = 15 the fast wiggles average out; over the whole dataset about "
                "half the amplitude is left.",
            ),
            (
                "Frequency band",
                "Which class loses most under a high-pass filter?",
                "Preset *high-pass*, $K$ = 9, on one `TR` record and one `VT` record.",
                "A `TR` feature map is small next to its record (about a quarter of "
                "the amplitude on average), a `VT` one is large (about three "
                "quarters): `VT` carries more fast wiggles. Single records vary.",
            ),
            (
                "Classes",
                "Can you draw a filter that responds to one class and not another?",
                "Start from a preset, drag the bars, and compare the feature maps of "
                "records of two classes.",
                "A high-pass shape separates `TR` from `VT` in size, but no single "
                "filter separates all four: a network learns many at once.",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Many filters at once

    * A single filter finds a single pattern.
    * A network **learns** its filters and uses **many at once**.
    * The number of filters is also the number of **channels**
    """)
    return


@app.cell
def _(F, llaima, torch):
    # four fixed filters of length 9 as the channels of one layer
    _k = torch.arange(9.0) - 4
    bank = torch.stack(
        [
            torch.ones(9) / 9,  # smooth
            (_k == 0).float() - 1 / 9,  # high-pass
            torch.where(_k < 0, -1.0, 1.0) * (_k != 0) / 8,  # rise edge
            torch.cos(2 * torch.pi * _k / 4) * torch.hann_window(9, periodic=False),
        ]
    )[:, None, :]
    bank_record = llaima.u[llaima.pick("train", "VT")][None]  # (1, 1, 3008)
    bank_maps = F.conv1d(bank_record, bank, padding=4)  # (1, 4, 3008)
    bank_names = ["smooth", "high-pass", "rise edge", "band-pass"]
    return bank_maps, bank_names, bank_record


@app.cell(hide_code=True)
def _(bank_maps, bank_names, bank_record, llaima, np, plots):
    plots.plot_filter_bank(
        np.arange(500) / llaima.sampling_rate,
        bank_record[0, 0, :500],
        bank_maps[0, :, :500],
        bank_names,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pooling

    Between convolutions we shrink the time axis with **max-pooling**. Pooling by 4
    makes every feature map four times shorter, so each remaining unit stands for
    four samples.
    """)
    return


@app.cell(hide_code=True)
def _(artifact):
    artifact("pooling")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stacking Filters and Pooling

    A convolution block is many filters and a nonlinearity. Between blocks we
    max-pool, so each unit sees a longer stretch of the record.
    """)
    return


@app.cell
def _(WaveformCNN, llaima, metrics):
    cnn_spec = metrics.layer_spec(WaveformCNN(n_out=4).features, llaima.u.shape[-1])
    cnn_convs = [layer for layer in cnn_spec if layer["kind"] == "conv"]
    cnn_field = cnn_convs[-1]["field"]
    return cnn_field, cnn_spec


@app.cell(hide_code=True)
def _(artifact, cnn_spec, llaima):
    artifact(
        "cnn_stack",
        layers=cnn_spec,
        rate=llaima.sampling_rate,
        record_samples=llaima.u.shape[-1],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The whole network
    """)
    return


@app.cell(hide_code=True)
def _(WaveformCNN, artifact, cnn_spec, llaima):
    artifact(
        "cnn_architecture",
        layers=cnn_spec,
        base=cnn_spec[0]["filters"],
        kernel=cnn_spec[0]["kernel"],
        n_out=4,
        params=sum(p.numel() for p in WaveformCNN(n_out=4).parameters()),
        rate=llaima.sampling_rate,
        record_samples=llaima.u.shape[-1],
    )
    return


@app.cell
def _(nn):
    def conv_block(c_in, c_out, kernel):
        return nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2),
            nn.BatchNorm1d(c_out),
            nn.ReLU(),
        )

    class WaveformCNN(nn.Module):
        """Waveform (batch, 1, samples) -> n_out numbers per record."""

        def __init__(self, n_out, base=16, kernel=9, dropout=0.3):
            super().__init__()
            self.features = nn.Sequential(
                conv_block(1, base, kernel),
                nn.MaxPool1d(4),
                conv_block(base, 2 * base, kernel),
                nn.MaxPool1d(4),
                conv_block(2 * base, 4 * base, kernel),
                nn.MaxPool1d(4),
                conv_block(4 * base, 4 * base, kernel),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * base, n_out))

        def forward(self, u):
            return self.head(self.features(u))

    return (WaveformCNN,)


@app.cell(hide_code=True)
def _(WaveformCNN, cnn_field, llaima, metrics, mo, shape_card):
    mo.vstack(
        [
            mo.md("Follow one untrained record through the network:"),
            shape_card(
                mo,
                metrics.shape_table(WaveformCNN(n_out=4), llaima.u[:1]),
                cnn_field,
                llaima.sampling_rate,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Which filters?

    So far we picked or drew the filters. In Part 2b the gradient of Part 1 chooses
    them, so that the four logits give the labels.
    """)
    return


if __name__ == "__main__":
    app.run()
