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
        "lIjoic2tpcCJ9LHsidHlwZSI6InNraXAifSx7InR5cGUiOiJzbGlkZSIsInNob3dDb2RlIjpmY"
        "WxzZX0seyJ0eXBlIjoic2tpcCJ9LHsidHlwZSI6InNraXAifSx7InR5cGUiOiJza2lwIn0seyJ"
        "0eXBlIjoic2tpcCJ9LHsidHlwZSI6InNsaWRlIiwic2hvd0NvZGUiOmZhbHNlfSx7InR5cGUiO"
        "iJza2lwIn0seyJ0eXBlIjoic2xpZGUiLCJzaG93Q29kZSI6ZmFsc2V9LHsidHlwZSI6InNsaWR"
        "lIiwic2hvd0NvZGUiOmZhbHNlfSx7InR5cGUiOiJmcmFnbWVudCIsInNob3dDb2RlIjpmYWxzZ"
        "X0seyJ0eXBlIjoic2xpZGUiLCJzaG93Q29kZSI6ZmFsc2V9LHsidHlwZSI6InNraXAifSx7InR"
        "5cGUiOiJmcmFnbWVudCIsInNob3dDb2RlIjpmYWxzZX0seyJ0eXBlIjoic2xpZGUiLCJzaG93Q"
        "29kZSI6ZmFsc2V9LHsidHlwZSI6InNsaWRlIiwic2hvd0NvZGUiOmZhbHNlfSx7InR5cGUiOiJ"
        "mcmFnbWVudCIsInNob3dDb2RlIjpmYWxzZX0seyJ0eXBlIjoic2xpZGUiLCJzaG93Q29kZSI6Z"
        "mFsc2V9LHsidHlwZSI6ImZyYWdtZW50Iiwic2hvd0NvZGUiOmZhbHNlfSx7InR5cGUiOiJza2l"
        "wIn0seyJ0eXBlIjoic2xpZGUiLCJzaG93Q29kZSI6ZmFsc2V9LHsidHlwZSI6ImZyYWdtZW50I"
        "iwic2hvd0NvZGUiOmZhbHNlfSx7InR5cGUiOiJzbGlkZSIsInNob3dDb2RlIjpmYWxzZX0seyJ"
        "0eXBlIjoiZnJhZ21lbnQiLCJzaG93Q29kZSI6ZmFsc2V9LHsidHlwZSI6InNraXAifSx7InR5c"
        "GUiOiJza2lwIn0seyJ0eXBlIjoic2tpcCJ9LHsidHlwZSI6InNsaWRlIiwic2hvd0NvZGUiOmZ"
        "hbHNlfSx7InR5cGUiOiJmcmFnbWVudCIsInNob3dDb2RlIjpmYWxzZX0seyJ0eXBlIjoic2tpc"
        "CJ9LHsidHlwZSI6InNsaWRlIiwic2hvd0NvZGUiOmZhbHNlfSx7InR5cGUiOiJmcmFnbWVudCI"
        "sInNob3dDb2RlIjpmYWxzZX0seyJ0eXBlIjoic2xpZGUiLCJzaG93Q29kZSI6ZmFsc2V9LHsid"
        "HlwZSI6InNraXAifSx7InR5cGUiOiJza2lwIn0seyJ0eXBlIjoic2tpcCJ9LHsidHlwZSI6InN"
        "raXAifSx7InR5cGUiOiJzbGlkZSIsInNob3dDb2RlIjpmYWxzZX0seyJ0eXBlIjoic2xpZGUiL"
        "CJzaG93Q29kZSI6ZmFsc2V9LHsidHlwZSI6InNraXAifSx7InR5cGUiOiJzbGlkZSIsInNob3d"
        "Db2RlIjpmYWxzZX0seyJ0eXBlIjoic2tpcCJ9LHsidHlwZSI6InNraXAifSx7InR5cGUiOiJzb"
        "GlkZSIsInNob3dDb2RlIjpmYWxzZX0seyJ0eXBlIjoic2xpZGUiLCJzaG93Q29kZSI6ZmFsc2V"
        "9LHsidHlwZSI6ImZyYWdtZW50Iiwic2hvd0NvZGUiOmZhbHNlfSx7InR5cGUiOiJzbGlkZSIsI"
        "nNob3dDb2RlIjpmYWxzZX0seyJ0eXBlIjoiZnJhZ21lbnQiLCJzaG93Q29kZSI6ZmFsc2V9LHs"
        "idHlwZSI6InNraXAifSx7InR5cGUiOiJzbGlkZSIsInNob3dDb2RlIjpmYWxzZX0seyJ0eXBlI"
        "joic2xpZGUiLCJzaG93Q29kZSI6ZmFsc2V9LHsidHlwZSI6ImZyYWdtZW50In0seyJ0eXBlIjo"
        "ic2xpZGUiLCJzaG93Q29kZSI6ZmFsc2V9LHsidHlwZSI6InNsaWRlIiwic2hvd0NvZGUiOmZhb"
        "HNlfSx7InR5cGUiOiJza2lwIn1dLCJkZWNrIjp7InRyYW5zaXRpb24iOiJub25lIiwidmVydGl"
        "jYWxBbGlnbiI6InRvcCJ9fX0="
    ),
    css_file="lecture_slides.css",
    html_head_file="slides_head.html",
)


@app.cell
def _():
    from types import SimpleNamespace

    import marimo as mo
    import numpy as np
    import torch
    from torch import nn

    from ml_seis import waveform_metrics as metrics
    from ml_seis import waveform_plotting as plots
    from ml_seis.pickerwidget import PickerExplorer
    from ml_seis.slides import artifact
    from ml_seis.waveform_data import (
        ceed_cache_path,
        load_ceed,
        make_pick_targets,
        prepare_ceed,
    )
    from ml_seis.waveform_setup import ceed_card, pick_card

    return (
        PickerExplorer,
        SimpleNamespace,
        artifact,
        ceed_cache_path,
        ceed_card,
        load_ceed,
        make_pick_targets,
        metrics,
        mo,
        nn,
        np,
        pick_card,
        plots,
        prepare_ceed,
        torch,
    )


@app.cell(hide_code=True)
def _(artifact, mo):
    mo.vstack(
        [
            mo.md(r"""
            # Part 2d: picking phases

            ### Machine learning and inversion for seismologists

            https://github.com/dominik-strutz/2026_seis_inverse_ml_introduction
            """),
            artifact("title_figure_picking", width="65%"),
        ],
        gap=0,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Recap from Parts 2a to 2c and 1b

    The convolution block of 2a to 2c and the training loop of 1b. The block gains a
    `stride`, which comes later, and loses its bias: BatchNorm shifts the output
    anyway.
    """)
    return


@app.cell
def _(nn):
    def conv_block(c_in, c_out, kernel, stride=1, bias=False):
        return nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel, stride, padding=kernel // 2, bias=bias),
            nn.BatchNorm1d(c_out, eps=1e-3),
            nn.ReLU(),
        )

    return (conv_block,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One answer per sample

    A classifier returns one answer for a whole record. That does not help for picking
    phase arrivals.

    A picker has to say *when*: at every sample it returns the probability of **P**,
    of **S** and of **neither**. A classification for each time step.

    The answer for a record of $T$ samples has shape $3 \times T$
    """)
    return


@app.cell
def _(mo, torch):
    # more threads than physical cores can make the small convolutions many times
    # slower, and beyond six threads they gain nothing
    torch.set_num_threads(min(torch.get_num_threads(), 6))
    cache_dir = mo.notebook_dir().parent / "data" / "waveforms"
    mo.md(f"Downloaded data are cached in `{cache_dir}`.")
    return (cache_dir,)


@app.cell(hide_code=True)
def _(cache_dir, ceed_cache_path, mo):
    ceed_cache = ceed_cache_path(cache_dir)
    ceed_cached = ceed_cache.exists()
    ceed_button = mo.ui.run_button(label="Download and prepare the CEED subset")
    # With the data on disk nothing is shown here.
    mo.vstack(
        []
        if ceed_cached
        else [
            ceed_button,
            mo.md(
                "*One download of 361 MB. After it the notebook trains two networks "
                "on the CPU, which takes a few minutes.*"
            ),
        ]
    )
    return ceed_button, ceed_cache, ceed_cached


@app.cell(hide_code=True)
def _(cache_dir, ceed_button, ceed_cached, load_ceed, mo, prepare_ceed):
    mo.stop(
        not (ceed_cached or ceed_button.value),
        mo.md("*The CEED records are not on this machine yet. Press the button.*"),
    )
    with mo.status.progress_bar(
        total=100, title="Downloading the CEED year file", remove_on_exit=True
    ) as _bar:
        _seen = [0]

        def _download_progress(fraction):
            _step = int(100 * fraction) - _seen[0]
            _seen[0] += _step
            if _step > 0:
                _bar.update(increment=_step)

        prepare_ceed(cache_dir, progress=_download_progress)
        _bar.update(increment=0, subtitle="Preparing the records...")
        ceed = load_ceed(cache_dir)
    return (ceed,)


@app.cell(hide_code=True)
def _(ceed, ceed_cache, ceed_card, mo):
    mo.vstack([mo.md("The records of this part:"), ceed_card(mo, ceed, ceed_cache)])
    return


@app.cell(hide_code=True)
def _(ceed, mo):
    mo.md(f"""
    ## Southern California records with analyst picks

    {len(ceed.p):,} three-component records (E, N, Z) at 100 Hz from the Southern
    California Seismic Network during 2007 published in CEED dataset.

    Each has an analyst's P and S pick
    """)
    return


@app.cell
def _(PickerExplorer, ceed, mo):
    data_explorer = mo.ui.anywidget(PickerExplorer(ceed))
    return (data_explorer,)


@app.cell(hide_code=True)
def _(data_explorer, mo):
    mo.vstack(
        [
            data_explorer,
            mo.md(r"""
    ### Three things to look for

    - **Distance.** How do the S–P time and the look of the record change with the
      distance of the station?
    - **Quality.** Which traces could you pick with confidence, and which not?
    - **Earthquakes.** How many stations record one earthquake, and in which set are
      they?
        """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(ceed, mo):
    _events = [
        len(set(ceed.event_id[ceed.indices(s).numpy()]))
        for s in ("train", "validation", "test")
    ]
    mo.md(f"""
    ## Split by earthquake, not by trace

    Several stations record the same earthquake -> their waveforms are strongly
    related

    With traces of one earthquake in both training and test data, the test score
    would measure how well the network remembers the earthquake, not how well it picks

    We split by earthquake, not at random: {_events[0]} earthquakes for training,
    {_events[1]} for validation and {_events[2]} for testing

    The three sets share no earthquake. In 2a the Llaima records had no event
    identity, so there we could not do this
    """)
    return


@app.cell(hide_code=True)
def _(ceed, plots):
    plots.plot_split_by_event(ceed)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From a pick to a target at every sample

    An analyst pick is one sample index, but a network with an answer at every sample
    needs a target at every sample.

    Three classes: **P**, **S** and **neither**. Since the picks have some uncertainty,
    each pick is spread into a narrow Gaussian of width $\sigma$

    At every sample the three values add to one

    Training uses random 30.72 s windows that contain both picks: a phase can arrive
    anywhere in the window
    """)
    return


@app.cell
def _(mo):
    # only for the figure below: the width that trains is chosen in the Retrain form
    plot_width = mo.ui.slider(0.05, 0.5, 0.05, value=0.2, label="target width σ (s)")
    PICK_LENGTH = 3072  # 30.72 s at 100 Hz, divisible by 4**4 for the four strides
    return PICK_LENGTH, plot_width


@app.cell(hide_code=True)
def _(PICK_LENGTH, ceed, make_pick_targets, mo, plot_width, plots):
    _i = ceed.indices("train")[[1533]]  # a clean record: P clear on Z, S on all three
    _u, _p, _s = ceed.crop(_i, PICK_LENGTH, [0.5])
    _target = make_pick_targets(_p, _s, PICK_LENGTH, 100 * plot_width.value)
    _figure = mo.vstack(
        [plot_width, plots.plot_targets(_u[0], _target[0], _p[0], _s[0])],
        align="center",
    )
    # centred in the space under the slide text, vertically and horizontally
    mo.Html(
        f'<div style="display:flex;align-items:center;justify-content:center;'
        f'min-height:620px">{mo.as_html(_figure).text}</div>'
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PhaseNet

    **PhaseNet** (Zhu and Beroza, 2019) is a convolutional network shaped like a U:
    three components of 3072 samples go in, scores for P, S and neither at every
    sample come out

    Three ideas build the network:

    - the **encoder**, the left half of the U, compresses the record
    - the **decoder**, the right half, expands it again
    - **skip connections** across the U bring back the timing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The encoder

    Four times: a conv block, then a step that makes the record 4 times shorter

    3072 samples become 12 and the channels grow from 8 to 128. A unit at the bottom
    stands for 256 samples, 2.56 s
    """)
    return


@app.cell(hide_code=True)
def _(artifact):
    artifact("phasenet_architecture", stage=1, length=3072, rate=100)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Shrinking with a stride

    A convolution with **stride** $s$ applies its filter every $s$ samples instead of
    at every one: the record gets $s$ times shorter

    Each output unit is a weighted sum of the samples under the filter, and the
    weights are learned
    """)
    return


@app.cell(hide_code=True)
def _(artifact):
    artifact("strided_conv", stride=4, length=3072)
    return


@app.cell
def _(conv_block, mo, torch):
    _x = torch.zeros(1, 8, 3072)  # (batch, channel, sample)
    _y = conv_block(8, 8, 7, stride=4)(_x)
    mo.md(
        f"A conv block with `stride=4` turns `{tuple(_x.shape)}` into "
        f"`{tuple(_y.shape)}`: 4 times fewer samples, the same 8 channels."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The decoder

    Four times: make the record 4 times longer, then a conv block

    A last convolution over the channels of one sample, a $1\times 1$ convolution,
    turns 8 channels into the three scores at each of the 3072 samples
    """)
    return


@app.cell(hide_code=True)
def _(artifact):
    artifact("phasenet_architecture", stage=2, length=3072, rate=100)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Growing back: the transposed convolution

    The decoder has to make the record longer

    A **transposed convolution** with stride $s$ is the mirror of a convolution with
    the same stride $s$: every input unit writes a copy of the filter, scaled by its
    value, $s$ samples after the copy of the unit before

    PhaseNet's filter is longer than its stride, so neighbouring copies overlap and
    add, and the output has 3 extra samples that the code cuts
    """)
    return


@app.cell(hide_code=True)
def _(artifact):
    artifact("transposed_conv_overlap")
    return


@app.cell
def _(mo, nn, torch):
    _x = torch.zeros(1, 8, 768)
    _y = nn.ConvTranspose1d(8, 8, 7, stride=4)(_x)
    mo.md(
        f"`nn.ConvTranspose1d(8, 8, 7, stride=4)` turns `{tuple(_x.shape)}` into "
        f"`{tuple(_y.shape)}`: 4 times 768 plus 3 samples. The code cuts `[1:-2]` to "
        "get 3072 again."
    )
    return


@app.cell
def _(conv_block, nn, torch):
    def up_block(c_in, c_out):
        """Transposed convolution (filter 7, stride 4) with BatchNorm and ReLU."""
        return nn.Sequential(
            nn.ConvTranspose1d(c_in, c_out, 7, stride=4, bias=False),
            nn.BatchNorm1d(c_out, eps=1e-3),
            nn.ReLU(),
        )

    class PhaseNet(nn.Module):
        """PhaseNet (Zhu and Beroza 2019) as SeisBench defines it, for windows of
        3072 samples: (batch, 3, 3072) -> (batch, 3, 3072) scores for P, S, neither.

        The only choice is ``skips``. To change the network, change the layers."""

        def __init__(self, skips=True):
            super().__init__()
            self.skips = skips
            m = 2 if skips else 1  # a skip doubles the channels the decoder sees
            # encoder: a conv block, then a stride-4 conv block, four times
            self.inc = conv_block(3, 8, 7, bias=True)
            self.enc0 = conv_block(8, 8, 7)
            self.down0 = conv_block(8, 8, 7, stride=4)
            self.enc1 = conv_block(8, 16, 7)
            self.down1 = conv_block(16, 16, 7, stride=4)
            self.enc2 = conv_block(16, 32, 7)
            self.down2 = conv_block(32, 32, 7, stride=4)
            self.enc3 = conv_block(32, 64, 7)
            self.down3 = conv_block(64, 64, 7, stride=4)
            self.enc4 = conv_block(64, 128, 7)  # the code: 128 channels, 12 steps
            # decoder: a transposed conv block, the skip, a conv block, four times
            self.up3 = up_block(128, 64)
            self.dec3 = conv_block(m * 64, 64, 7)
            self.up2 = up_block(64, 32)
            self.dec2 = conv_block(m * 32, 32, 7)
            self.up1 = up_block(32, 16)
            self.dec1 = conv_block(m * 16, 16, 7)
            self.up0 = up_block(16, 8)
            self.dec0 = conv_block(m * 8, 8, 7)
            self.out = nn.Conv1d(8, 3, 1)  # 1x1: P, S, neither at every sample

        def forward(self, u):
            skip0 = self.enc0(self.inc(u))  # kept for the skip connections
            skip1 = self.enc1(self.down0(skip0))
            skip2 = self.enc2(self.down1(skip1))
            skip3 = self.enc3(self.down2(skip2))
            z = self.enc4(self.down3(skip3))  # the code
            # a transposed conv of filter 7 and stride 4 adds 3 samples: cut them
            h = self.up3(z)[:, :, 1:-2]
            h = self.dec3(torch.cat([skip3, h], dim=1) if self.skips else h)
            h = self.up2(h)[:, :, 1:-2]
            h = self.dec2(torch.cat([skip2, h], dim=1) if self.skips else h)
            h = self.up1(h)[:, :, 1:-2]
            h = self.dec1(torch.cat([skip1, h], dim=1) if self.skips else h)
            h = self.up0(h)[:, :, 1:-2]
            h = self.dec0(torch.cat([skip0, h], dim=1) if self.skips else h)
            return self.out(h)

    return (PhaseNet,)


@app.cell
def _(PhaseNet, mo, torch):
    _net, _flat = PhaseNet(), PhaseNet(skips=False)
    n_weights = sum(p.numel() for p in _net.parameters())
    n_weights_flat = sum(p.numel() for p in _flat.parameters())
    _scores = _net(torch.zeros(2, 3, 3072))
    mo.md(
        f"Scores for P, S and neither at every sample: `{tuple(_scores.shape)}`. "
        f"**{n_weights:,}** weights with skip connections, {n_weights_flat:,} without. "
        "SeisBench takes windows of 3001 samples, which needs a few padding tricks; "
        "3072 divides by 4 four times, so none are needed here."
    )
    return n_weights, n_weights_flat


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Skip connections across the U

    The output of each conv block of the encoder is copied across the U to the decoder,
    to the level of the same length

    This helps with getting the timing right, which is possible without but harder
    """)
    return


@app.cell(hide_code=True)
def _(artifact, n_weights, n_weights_flat):
    artifact(
        "phasenet_architecture",
        stage=3,
        params=n_weights,
        params_no_skips=n_weights_flat,
        length=3072,
        rate=100,
    )
    return


@app.cell
def _(mo, torch):
    _skip, _up = torch.zeros(1, 8, 3072), torch.zeros(1, 8, 3072)
    _both = torch.cat([_skip, _up], dim=1)  # along the channel axis
    mo.md(
        f"`torch.cat([skip, up], dim=1)`: `{tuple(_skip.shape)}` and "
        f"`{tuple(_up.shape)}` make `{tuple(_both.shape)}`; the next conv block "
        "takes 16 channels."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The loss at every sample

    The network returns scores $a_{c,t}$ for class $c$ (P, S, neither) at sample $t$

    A softmax across the classes gives probabilities $p_{c,t}$, and the loss is the
    cross-entropy of 2b with the soft target $y_{c,t}$, averaged over all samples

    $$
    S = -\frac{1}{T} \sum_{t=1}^{T} \sum_{c} y_{c,t} \log p_{c,t}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(artifact):
    artifact("pick_loss")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training the picker

    The loop of 2b, on mini-batches of traces

    - a random 30.72 s window of each trace that holds both picks
    - forward pass and the loss at every sample
    - backward pass and one step of Adam
    - after every epoch, the loss on the validation earthquakes

    We train two networks: **PhaseNet**, and the same network **without skip
    connections**
    """)
    return


@app.cell
def _(PICK_LENGTH, ceed, make_pick_targets, mo, torch):
    def loss_fn(scores, target):
        # soft-target cross-entropy over the class axis (P, S, neither)
        return -(target * torch.log_softmax(scores, dim=1)).sum(dim=1).mean()

    def train(model, width, title, epochs=12, batch=32):
        """Train ``model`` on the training traces; the loss after every epoch.

        ``width`` is the width of the Gaussian targets in seconds."""
        idx_train, idx_val = ceed.indices("train"), ceed.indices("validation")
        optimiser = torch.optim.Adam(model.parameters(), lr=2e-3)
        # the learning rate falls over the epochs like a cosine, as in 2b
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
        history = {"train": [], "validation": []}
        for _ in mo.status.progress_bar(
            range(epochs), title=title, remove_on_exit=True
        ):
            model.train()
            order = idx_train[torch.randperm(len(idx_train))]  # shuffle every epoch
            losses = []
            for i in range(0, len(order), batch):
                idx = order[i : i + batch]  # 1. a mini-batch of traces
                u, p, s = ceed.crop(idx, PICK_LENGTH, torch.rand(len(idx)))
                target = make_pick_targets(p, s, PICK_LENGTH, 100 * width)
                optimiser.zero_grad()
                loss = loss_fn(model(u), target)  # 2. forward pass and loss
                loss.backward()  # 3. backward pass: the gradient of Part 1a
                optimiser.step()  # 4. one step of Adam
                losses.append(loss.item())
            scheduler.step()
            model.eval()
            with torch.no_grad():  # 5. validation traces, never used for a step
                u, p, s = ceed.crop(idx_val, PICK_LENGTH, ceed.position_eval[idx_val])
                target = make_pick_targets(p, s, PICK_LENGTH, 100 * width)
                history["validation"].append(loss_fn(model(u), target).item())
            history["train"].append(sum(losses) / len(losses))
        return history

    return (train,)


@app.cell
def _(mo):
    pick_form = (
        mo.md("{width}")
        .batch(
            width=mo.ui.slider(0.05, 0.5, 0.05, value=0.2, label="target width σ (s)"),
        )
        .form(submit_button_label="Retrain")
    )
    return (pick_form,)


@app.cell
def _(PICK_LENGTH, PhaseNet, SimpleNamespace, ceed, mo, torch, train):
    @mo.cache  # a width that was trained once is not trained again
    def train_both(width):
        """PhaseNet with and without skip connections, and their test probabilities."""
        nets, history = {}, {}
        for name, skips in (("skips", True), ("no skips", False)):
            torch.manual_seed(0)
            nets[name] = PhaseNet(skips=skips)
            history[name] = train(nets[name], width, f"Training the picker ({name})")
        index = ceed.indices("test")  # earthquakes never seen in training
        u, p, s = ceed.crop(index, PICK_LENGTH, ceed.position_eval[index])
        prob = {}
        for name, net in nets.items():
            net.eval()
            with torch.no_grad():
                prob[name] = torch.softmax(net(u), dim=1).numpy()
        return SimpleNamespace(
            width=width, history=history, index=index.numpy(), u=u, p=p, s=s, prob=prob
        )

    return (train_both,)


@app.cell
def _(pick_form, train_both):
    # trains as soon as the notebook runs; the form retrains with another width
    run = train_both(pick_form.value["width"] if pick_form.value else 0.2)
    return (run,)


@app.cell(hide_code=True)
def _(mo, pick_form, plots, run):
    mo.vstack(
        [
            mo.md("## Training and validation loss"),
            pick_form,
            plots.plot_history_comparison(run.history),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From probabilities to picks

    For every test trace the picker returns a probability curve for P and for S

    A phase is **picked** at the highest sample of its curve if the peak reaches a
    threshold, and **missed** otherwise

    The error of a pick is *predicted minus analyst* time

    A high threshold misses more phases but trusts the picks it keeps, a low one
    picks nearly everything, including wrong peaks
    """)
    return


@app.cell
def _(PickerExplorer, SimpleNamespace, ceed, mo, run):
    results_explorer = mo.ui.anywidget(
        PickerExplorer(
            ceed,
            SimpleNamespace(index=run.index, u=run.u, p=run.p, s=run.s, prob=run.prob),
        )
    )
    return (results_explorer,)


@app.cell(hide_code=True)
def _(mo, results_explorer):
    mo.vstack(
        [
            results_explorer,
            mo.md(r"""
    ### Three things to look for

    - **Where the curve peaks.** How close is the peak to the analyst pick, and how
      high is it?
    - **Failures.** What is going on in a trace the picker gets seconds wrong?
    - **Skips.** What does the network without skip connections get right, and what
      wrong?
        """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    pick_threshold = mo.ui.slider(0.05, 0.9, 0.05, value=0.2, label="pick threshold")
    return (pick_threshold,)


@app.cell
def _(SimpleNamespace, metrics, pick_threshold, run):
    errors = {}  # per network: the picks, the errors of P and S and their summaries
    for _name, _prob in run.prob.items():
        _picks, _ = metrics.extract_picks(_prob, pick_threshold.value)
        _p, _s = metrics.pick_residuals(_picks, run.p, run.s, 100.0)
        errors[_name] = SimpleNamespace(
            picks=_picks,
            p=_p,
            s=_s,
            summary_p=metrics.pick_summary(_p),
            summary_s=metrics.pick_summary(_s),
        )
    return (errors,)


@app.cell(hide_code=True)
def _(errors, mo, pick_card, pick_threshold, plots):
    mo.vstack(
        [
            mo.md("## Errors of the picks"),
            pick_threshold,
            pick_card(mo, {k: (v.summary_p, v.summary_s) for k, v in errors.items()}),
            plots.plot_pick_residuals({k: (v.p, v.s) for k, v in errors.items()}),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Predicted against analyst

    Picks on the diagonal agree with the analyst

    Points far from it are a different phase, another event or ambient noise, not a
    noisy pick

    The S–P time, the difference of the two, is what a locator uses
    """)
    return


@app.cell(hide_code=True)
def _(errors, plots, run):
    plots.plot_pick_scatter(errors["skips"].picks, run.p, run.s)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Errors against signal and distance

    Most records here are weak, so a cloud of small errors at low signal-to-noise
    ratio is expected

    The question is where the tail of large errors comes from, and whether the model
    or the analyst pick deserves the blame
    """)
    return


@app.cell(hide_code=True)
def _(ceed, errors, plots, run):
    plots.plot_residual_trends(
        ceed.snr[run.index],
        ceed.distance_km[run.index],
        errors["skips"].p,
        errors["skips"].s,
    )
    return


@app.cell
def _(errors, np):
    # a robust spread of the P errors: the tail would dominate the rms
    sigma_hat = float(1.4826 * np.nanmedian(np.abs(errors["skips"].p)))
    return (sigma_hat,)


@app.cell(hide_code=True)
def _(mo, sigma_hat):
    mo.md(f"""
    ## The spread of the pick errors

    In Part 1 every pick had an error
    $\\varepsilon_i \\sim \\mathcal{{N}}(0, \\sigma^2)$

    For this picker the spread of the P errors on the test traces is about
    **{sigma_hat:.2f} s** (a robust estimate from the median absolute error)

    This is the $\\sigma$ of the locator: the location uncertainty of Part 1 scales
    with it

    The large errors are not a wider Gaussian but a different phase, so the median,
    not the rms, tells the typical pick
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SeisBench: plug and play

    **SeisBench** (Woollam et al., 2022) is an open-source Python framework with one
    interface for many pickers

    Pretrained PhaseNet and EQTransformer for several regions and instruments (for
    example weights trained on STEAD, INSTANCE, SCEDC and ETHZ data), benchmark
    datasets and a common data format

    `annotate` returns probability curves like ours; `classify` returns the picks
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    SeisBench in three lines:

    ```python
    import seisbench.models as sbm

    model = sbm.PhaseNet.from_pretrained("original")
    probabilities = model.annotate(stream)  # P, S and noise curves
    picks = model.classify(stream)  # the picks
    ```
    """)
    return


@app.cell(hide_code=True)
def _(artifact):
    artifact("course_pipeline")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Limits

    - **Labels.** The analyst picks are both the target and the yardstick, and they
      are uncertain themselves
    - **Other data.** A picker degrades on other regions and instruments; our one
      network-year says nothing about that
    - **Errors.** Pick errors are heavy-tailed, not the Gaussian of Part 1
    - **One earthquake at a time.** Overlapping earthquakes in one window are
      missed
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sources

    - Zhu and Beroza (2019), *PhaseNet*, Geophys. J. Int. 216, 261–273,
      doi:10.1093/gji/ggy423, arXiv:1803.03211
    - Woollam et al. (2022), *SeisBench*, Seismol. Res. Lett. 93, 1695–1709,
      doi:10.1785/0220210324, arXiv:2111.00786
    - Münchmeyer et al. (2022), *Which picker fits my data?*, J. Geophys. Res. Solid
      Earth 127, e2021JB023499, doi:10.1029/2021JB023499, arXiv:2110.13671
    - Zhu et al. (2022), *Earthquake phase association using a Bayesian Gaussian
      mixture model*, J. Geophys. Res. Solid Earth 127, e2021JB023249,
      doi:10.1029/2021JB023249
    - Tan et al. (2021), *Machine-learning-based high-resolution earthquake catalog
      ... Central Italy sequence*, The Seismic Record 1, 11–19,
      doi:10.1785/0320210001
    - Mousavi and Beroza (2022), *Deep-learning seismology*, Science 377, eabm4470,
      doi:10.1126/science.abm4470
    """)
    return


if __name__ == "__main__":
    app.run()
