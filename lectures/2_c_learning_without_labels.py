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
        "CJ9LHsidHlwZSI6InNraXAifSx7InR5cGUiOiJza2lwIn0seyJ0eXBlIjoic2tpcCJ9LHsidHl"
        "wZSI6InNsaWRlIiwic2hvd0NvZGUiOmZhbHNlfSx7InR5cGUiOiJmcmFnbWVudCIsInNob3dDb"
        "2RlIjpmYWxzZX0seyJ0eXBlIjoic2xpZGUiLCJzaG93Q29kZSI6ZmFsc2V9LHsidHlwZSI6ImZ"
        "yYWdtZW50Iiwic2hvd0NvZGUiOmZhbHNlfSx7InR5cGUiOiJmcmFnbWVudCIsInNob3dDb2RlI"
        "jpmYWxzZX0seyJ0eXBlIjoic2xpZGUiLCJzaG93Q29kZSI6ZmFsc2V9LHsidHlwZSI6ImZyYWd"
        "tZW50Iiwic2hvd0NvZGUiOmZhbHNlfSx7InR5cGUiOiJzbGlkZSIsInNob3dDb2RlIjpmYWxzZ"
        "X0seyJ0eXBlIjoiZnJhZ21lbnQiLCJzaG93Q29kZSI6ZmFsc2V9LHsidHlwZSI6InNraXAifSx"
        "7InR5cGUiOiJza2lwIn0seyJ0eXBlIjoic2tpcCJ9LHsidHlwZSI6InNsaWRlIiwic2hvd0NvZ"
        "GUiOmZhbHNlfSx7InR5cGUiOiJza2lwIn0seyJ0eXBlIjoic2tpcCJ9LHsidHlwZSI6InNraXA"
        "ifSx7InR5cGUiOiJmcmFnbWVudCIsInNob3dDb2RlIjpmYWxzZX0seyJ0eXBlIjoic2tpcCJ9L"
        "HsidHlwZSI6InNsaWRlIiwic2hvd0NvZGUiOmZhbHNlfSx7InR5cGUiOiJza2lwIn0seyJ0eXB"
        "lIjoic2tpcCJ9LHsidHlwZSI6InNsaWRlIiwic2hvd0NvZGUiOmZhbHNlfSx7InR5cGUiOiJmc"
        "mFnbWVudCIsInNob3dDb2RlIjpmYWxzZX0seyJ0eXBlIjoic2xpZGUiLCJzaG93Q29kZSI6ZmF"
        "sc2V9LHsidHlwZSI6InNsaWRlIiwic2hvd0NvZGUiOmZhbHNlfSx7InR5cGUiOiJza2lwIn0se"
        "yJ0eXBlIjoic2tpcCJ9LHsidHlwZSI6InNraXAifSx7InR5cGUiOiJza2lwIn0seyJ0eXBlIjo"
        "ic2xpZGUiLCJzaG93Q29kZSI6ZmFsc2V9LHsidHlwZSI6InNsaWRlIiwic2hvd0NvZGUiOmZhb"
        "HNlfSx7InR5cGUiOiJza2lwIn0seyJ0eXBlIjoic2tpcCJ9LHsidHlwZSI6InNsaWRlIiwic2h"
        "vd0NvZGUiOmZhbHNlfSx7InR5cGUiOiJmcmFnbWVudCIsInNob3dDb2RlIjpmYWxzZX0seyJ0e"
        "XBlIjoic2tpcCJ9LHsidHlwZSI6InNsaWRlIiwic2hvd0NvZGUiOmZhbHNlfSx7InR5cGUiOiJ"
        "za2lwIn0seyJ0eXBlIjoic2tpcCJ9LHsidHlwZSI6InNsaWRlIiwic2hvd0NvZGUiOmZhbHNlf"
        "V0sImRlY2siOnsidHJhbnNpdGlvbiI6Im5vbmUiLCJ2ZXJ0aWNhbEFsaWduIjoidG9wIn19fQ="
        "="
    ),
    css_file="lecture_slides.css",
    html_head_file="slides_head.html",
)


@app.cell
def _():
    from types import SimpleNamespace

    import marimo as mo
    import torch
    from torch import nn

    from ml_seis import waveform_metrics as metrics
    from ml_seis import waveform_plotting as plots
    from ml_seis.autoencoderwidget import AutoencoderFlow
    from ml_seis.clusterwidget import ClusterNaming
    from ml_seis.slides import artifact, try_it
    from ml_seis.stepperwidget import Stepper
    from ml_seis.waveform_data import (
        LLAIMA_NAMES,
        LLAIMA_WINDOW,
        llaima_cache_path,
        load_llaima,
        prepare_llaima,
    )
    from ml_seis.waveform_setup import cache_ready, default_cache_dir, llaima_card

    return (
        AutoencoderFlow,
        ClusterNaming,
        LLAIMA_NAMES,
        LLAIMA_WINDOW,
        SimpleNamespace,
        Stepper,
        artifact,
        cache_ready,
        default_cache_dir,
        llaima_cache_path,
        llaima_card,
        load_llaima,
        metrics,
        mo,
        nn,
        plots,
        prepare_llaima,
        torch,
        try_it,
    )


@app.cell(hide_code=True)
def _(artifact, mo):
    mo.vstack(
        [
            mo.md(r"""
            # Part 2c: learning without labels

            ### Machine learning and inversion for seismologists

            https://github.com/dominik-strutz/2026_seis_inverse_ml_introduction
            """),
            artifact("title_figure_waveforms", width="65%"),
        ],
        gap=0,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Recap from Parts 2a, 2b and 1b

    The network of 2b: blocks of filters, batch normalisation and ReLU with
    max-pooling in between, an average over time and a linear layer, 8 filters in
    every block. In 2b the linear layer returned one number per class; here it
    returns 4 numbers, the latent vector. The training loop of 2b is further down,
    written for a pair of an input and a target.
    """)
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

        def __init__(self, n_out, width=8, kernel=9):
            super().__init__()
            self.features = nn.Sequential(
                conv_block(1, width, kernel),
                nn.MaxPool1d(4),
                conv_block(width, width, kernel),
                nn.MaxPool1d(4),
                conv_block(width, width, kernel),
                nn.MaxPool1d(4),
                conv_block(width, width, kernel),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )
            self.head = nn.Linear(width, n_out)

        def forward(self, u):
            return self.head(self.features(u))

    return WaveformCNN, conv_block


@app.cell
def _(default_cache_dir, mo, torch):
    # more threads than physical cores can make the small convolutions many times
    # slower, and beyond six threads they gain nothing
    torch.set_num_threads(min(torch.get_num_threads(), 6))
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
            mo.md("The Llaima records of 2a and 2b, with the same split:"),
            llaima_card(mo, llaima, llaima_cache, LLAIMA_NAMES),
        ]
    )
    return


@app.cell(hide_code=True)
def _(LLAIMA_WINDOW, llaima, mo, spec_cover):
    _seconds = LLAIMA_WINDOW / llaima.sampling_rate
    _dropped = len(llaima.indices("train")) - len(
        llaima.indices("train", min_live=spec_cover)
    )
    mo.md(f"""
    ### The first {_seconds:.0f} s

    As in 2b, the network gets only the **first {_seconds:.0f} s** of every record
    ({LLAIMA_WINDOW} samples): the onset is in it, and no flat padding that would
    give the class away. {_dropped} training records with less signal are left out.

    The mean of every record is removed, as ObsPy does before it computes a
    spectrogram, and all records share one scale.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Take away the labels

    Instead of predicting labels we now compress (encode) every record into a short
    **latent vector** $\mathbf{z}$ (here four numbers) and decompress (decode) it
    again. The decoded result is compared to the record, or to a product derived from
    it

    $$
    \mathbf{z} = \text{encoder}(\mathbf{u})
    \qquad
    \hat{\mathbf{s}} = \text{decoder}(\mathbf{z})
    \qquad
    S = L(\mathbf{s}, \hat{\mathbf{s}})
    $$

    Here the loss is the mean squared error over the $n$ records of a batch

    $$
    L(\mathbf{s}, \hat{\mathbf{s}}) = \frac{1}{n} \sum_{i=1}^{n}
    \lVert \mathbf{s}_i - \hat{\mathbf{s}}_i \rVert^2
    $$

    A network trained like this is an **autoencoder**
    """)
    return


@app.cell(hide_code=True)
def _(artifact):
    artifact("autoencoder")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Transposed convolution

    A convolution with a stride, or pooling afterwards, makes a sequence shorter. A
    **transposed convolution** does the reverse: it makes it longer
    """)
    return


@app.cell(hide_code=True)
def _(artifact):
    artifact("transposed_conv", height=600)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md(r"**General**, kernel $K$ and stride $s$"),
                    mo.md(r"""
                    $$
                    h^{\text{out}}_j = \sum_t w_{j - s t} \, h^{\text{in}}_t
                    $$
                    """),
                    mo.md(r"$w_\tau = 0$ outside $\tau = 0, \dots, K - 1$"),
                ],
                gap=1.5,
                align="center",
            ),
            mo.vstack(
                [
                    mo.md(r"**Special case**, $K = s = 5$ (the figure)"),
                    mo.md(r"""
                    $$
                    h^{\text{out}}_{5t + \tau} = w_\tau \, h^{\text{in}}_t
                    \qquad \tau = 0, \dots, 4
                    $$
                    """),
                    mo.md("no overlap: each value becomes five"),
                ],
                gap=1.5,
                align="center",
            ),
        ],
        justify="space-around",
        align="start",
        gap=4,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A convolution of kernel 1

    The last layer of the decoder turns 8 channels into the 16 frequencies of the
    spectrogram

    A **convolution of kernel 1** looks at one step at a time: the same linear layer
    maps the 8 channel values of a step to 16 numbers

    $$
    \hat{s}_{f,t} = b_f + \sum_{c=1}^{8} W_{f,c} \, h_{c,t}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(artifact):
    artifact("conv_1x1")
    return


@app.cell(hide_code=True)
def _(mo, spec_shape):
    mo.md(f"""
    ## Waveform in, spectrogram out

    The input is the record itself. The target is its **spectrogram** in decibels.

    We do not rebuild the samples themselves: four numbers cannot store the phase of
    every oscillation, so the spectrogram leaves the phase out on purpose

    The decoder writes the {spec_shape[0]} frequencies as channels.
    """)
    return


@app.cell(hide_code=True)
def _(
    llaima,
    llaima_spec,
    plots,
    spec_fmax,
    spec_fmin,
    spec_hop,
    spec_window,
    to_db,
    train_examples,
    u,
):
    plots.plot_spectrogram_row(
        to_db(llaima_spec),
        llaima.classes,
        train_examples,
        spec_hop,
        llaima.sampling_rate,
        spec_window,
        spec_fmax,
        waves=u[list(train_examples), 0],
        fmin=spec_fmin,
    )
    return


@app.cell
def _(LLAIMA_WINDOW, llaima, metrics):
    # the recipe lives in ml_seis.waveform_metrics, where the explorer of 2a and 2b
    # takes it from too: the pictures there are the target the network rebuilds here
    spec_window, spec_hop = metrics.SPEC_WINDOW, metrics.SPEC_HOP  # 2.6 s, every 0.5 s
    # frequencies of the first and the last row (Hz): 0.4 to 12.5 Hz in pairs
    _freq, _ = metrics.spectrogram_axes(spec_window, llaima.sampling_rate)
    spec_fmin, spec_fmax = float(_freq[0]), float(_freq[-1])
    window = llaima.window()  # the first 15 s of every record, (N, 1, 750)
    # a 128-sample window that starts half a second before the 15 s end reads past
    # them, one window minus one hop in all: so much more of the record, live, so
    # that no spectrogram time bin ever falls outside the signal
    spec_cover = LLAIMA_WINDOW + spec_window - spec_hop
    spec_read = llaima.u[:, 0, :spec_cover]
    train_idx = llaima.indices("train", min_live=spec_cover)
    val_idx = llaima.indices("validation", min_live=spec_cover)
    test_idx = llaima.indices("test", min_live=spec_cover)
    # the mean of every record removed (the records carry an offset, which is no
    # signal; ObsPy's spectrogram does the same), then one common scale for all records
    window = window - window.mean(-1, keepdim=True)
    u = window / window[train_idx].std()
    return (
        spec_cover,
        spec_fmax,
        spec_fmin,
        spec_hop,
        spec_read,
        spec_window,
        test_idx,
        train_idx,
        u,
        val_idx,
    )


@app.cell
def _(llaima, metrics, spec_read, train_idx):
    # power in decibels, as ObsPy computes it but without zero padding and with every
    # two neighbouring frequencies averaged: 16 frequencies from 0.4 to 12.5 Hz
    # (``ml_seis.waveform_metrics.spectrogram``, also the picture of the explorer)
    llaima_spec = metrics.spectrogram(spec_read)
    # every frequency is standardised with the training records, as any input or
    # target of a network is: mean 0 and standard deviation 1 (the average spectrum
    # alone would otherwise be most of what the network has to rebuild)
    spec_mean = llaima_spec[train_idx].mean((0, 2))[:, None]  # (frequency, 1), in dB
    spec_std = llaima_spec[train_idx].std((0, 2))[:, None]
    llaima_spec = (llaima_spec - spec_mean) / spec_std

    def to_db(s):
        """The standardised numbers of the network back in decibels: the picture
        ObsPy draws. For (frequency, time bin) or (batch, frequency, time bin)."""
        return s * spec_std + spec_mean

    spec_shape = tuple(llaima_spec.shape[1:])

    # the first record of every class among the training records
    train_examples = [
        int(train_idx[llaima.label[train_idx] == k][0])
        for k in range(len(llaima.classes))
    ]
    return llaima_spec, spec_shape, to_db, train_examples


@app.cell
def _(WaveformCNN, conv_block, nn):
    class Autoencoder(nn.Module):
        """(batch, 1, samples) -> (batch, channels, steps), through the latent vector z.

        The encoder is the CNN of 2b: its ``n_latent`` output numbers are the latent
        vector. The decoder spreads them over 6 steps of ``width`` channels, writes
        each step as five steps (a transposed convolution with kernel and stride 5,
        so the copies do not overlap), filters them and writes the ``channels``
        frequencies of every step."""

        def __init__(self, channels, width=8, kernel=9, n_latent=4):
            super().__init__()
            self.encoder = WaveformCNN(n_latent, width, kernel)
            self.decoder = nn.Sequential(
                nn.Linear(n_latent, 6 * width),
                nn.ReLU(),
                nn.Unflatten(1, (width, 6)),
                conv_block(width, width, 3),
                nn.ConvTranspose1d(width, width, 5, stride=5),  # 6 steps -> 30 steps
                conv_block(width, width, 5),
                nn.Conv1d(width, channels, 1),
            )

        def forward(self, u):
            return self.decoder(self.encoder(u))

    return (Autoencoder,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training an autoencoder

    The loop of 2b: the network reads the waveform, and its spectrogram is the target

    - forward pass and squared-error loss
    - backward pass and one step of Adam
    - after every epoch, the loss on the validation records
    - the **learning rate** falls smoothly over the epochs, as in 2b

    No label enters anywhere

    The test records stay untouched until the network is trained
    """)
    return


@app.cell
def _(mo, nn, torch):
    def train(model, u, s, idx_train, idx_val, epochs=30):
        """Train ``model`` to rebuild the spectrograms ``s[idx_train]`` of the
        waveforms ``u[idx_train]``.

        Returns the training and validation loss after every epoch."""
        loss_fn = nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=2e-3)
        # the learning rate falls from 2e-3 to about 0 over the epochs, like a cosine
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
        history = {"train": [], "validation": []}
        for _ in mo.status.progress_bar(
            range(epochs), title="Training", remove_on_exit=True
        ):
            model.train()
            order = idx_train[torch.randperm(len(idx_train))]  # shuffle every epoch
            losses = []
            for i in range(0, len(order), 32):
                batch = u[order[i : i + 32]]  # 1. a mini-batch of 32 records
                target = s[order[i : i + 32]]  # ... and their spectrograms
                optimiser.zero_grad()
                loss = loss_fn(model(batch), target)  # 2. forward pass and loss
                loss.backward()  # 3. backward pass: the gradient of Part 1a
                optimiser.step()  # 4. one step of Adam
                losses.append(loss.item())
            scheduler.step()  # a smaller learning rate for the next epoch
            model.eval()
            with torch.no_grad():  # 5. the validation records, never used for a step
                validation = loss_fn(model(u[idx_val]), s[idx_val]).item()
            history["train"].append(sum(losses) / len(losses))
            history["validation"].append(validation)
        return history

    return (train,)


@app.cell(hide_code=True)
def _(mo):
    ae_form = (
        mo.md("{seed} {epochs} {width}")
        .batch(
            seed=mo.ui.number(0, 99, 1, value=0, label="seed"),
            epochs=mo.ui.slider(1, 40, 1, value=30, label="epochs"),
            width=mo.ui.dropdown([4, 8, 16], value=8, label="filters"),
        )
        .form(submit_button_label="Retrain")
    )
    return (ae_form,)


@app.cell
def _(
    Autoencoder,
    SimpleNamespace,
    ae_form,
    llaima_spec,
    mo,
    test_idx,
    torch,
    train,
    train_idx,
    u,
    val_idx,
):
    @mo.cache  # settings that were trained once are not trained again
    def train_autoencoder(settings):
        """Train an autoencoder on the Llaima records with the form's settings."""
        torch.manual_seed(settings["seed"])
        model = Autoencoder(llaima_spec.shape[1], width=settings["width"])
        history = train(
            model, u, llaima_spec, train_idx, val_idx, epochs=settings["epochs"]
        )

        # the test records, looked at for the first time
        model.eval()
        with torch.no_grad():
            latent = model.encoder(u[test_idx])  # (records, 4)
            rebuilt = model.decoder(latent)
            s_test = llaima_spec[test_idx]
        return SimpleNamespace(
            model=model,
            history=history,
            settings=settings,
            latent=latent,
            rebuilt=rebuilt,
            error=float(((rebuilt - s_test) ** 2).sum() / (s_test**2).sum()),
        )

    # trains as soon as the notebook runs; the form retrains with other settings
    ae_run = train_autoencoder(ae_form.value or {"seed": 0, "epochs": 30, "width": 8})
    return (ae_run,)


@app.cell(hide_code=True)
def _(ae_form, ae_run, mo, plots):
    mo.vstack(
        [
            ae_form,
            plots.plot_history(
                ae_run.history, ylabel="reconstruction error", logy=True
            ),
            mo.md(
                f"On the test records the rebuild misses "
                f"**{100 * ae_run.error:.0f}%** of the spectrogram's squared "
                f"variation, from a latent vector of {ae_run.latent.shape[1]} numbers."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(Stepper, llaima, mo, test_idx, torch):
    _fewest = int(torch.bincount(llaima.label[test_idx]).min())
    rec_index = mo.ui.anywidget(Stepper(_fewest, label="record"))
    return (rec_index,)


@app.cell(hide_code=True)
def _(
    ae_run,
    llaima,
    llaima_spec,
    mo,
    plots,
    rec_index,
    spec_fmax,
    spec_fmin,
    spec_hop,
    spec_window,
    test_idx,
    to_db,
    u,
):
    # the n-th test record of every class, as positions among the test records
    _pos = [
        int(
            (llaima.label[test_idx] == c)
            .nonzero()
            .squeeze(1)[rec_index.value["value"] - 1]
        )
        for c in range(len(llaima.classes))
    ]
    mo.vstack(
        [
            rec_index,
            plots.plot_reconstruction(
                u[test_idx[_pos], 0],
                to_db(llaima_spec[test_idx[_pos]]),
                to_db(ae_run.rebuilt[_pos]),
                llaima.classes,
                ae_run.latent[_pos],
                spec_hop,
                llaima.sampling_rate,
                spec_window,
                spec_fmax,
                spec_fmin,
            ),
            mo.md(r"""
    ### Three things to look for

    - **Kept.** Which parts of the spectrogram survive a latent vector of four numbers?
    - **Lost.** What does the original have that the rebuild does not?
    - **Classes.** Which class is rebuilt best, which worst, and what do the four
      numbers of the latent vector do from one class to the next?
            """),
        ],
        align="center",
        justify="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How the playground works

    The network trains as soon as the notebook runs, with the settings of the form
    above the curve. Change them and press **Retrain**; settings you already trained
    show at once. The playground shows one test record of each class, one row each: its
    waveform, its spectrogram and the spectrogram the decoder builds from the four
    numbers of its latent vector, on one colour scale. *Record* steps through the test
    records, all four rows at once.
    """)
    return


@app.cell(hide_code=True)
def _(try_it):
    try_it(
        [
            (
                "Kept",
                "What survives?",
                "Compare the TR and LP rows, then step through a few records.",
                "The broad band where the energy sits and how it changes slowly "
                "along the record. For the steady tremor TR that is most of it: "
                "about 18% of the squared variation of its spectrogram is missed.",
            ),
            (
                "Lost",
                "What is missing from the rebuild?",
                "Compare the LP and TC rows, then step through several records.",
                "The fine texture: separate patches of energy, short bursts, the "
                "exact time of the onset. The rebuild is smooth in time, because "
                "the average over time in the encoder leaves the latent vector no "
                "way to say when. About 52% (LP) and 40% (TC) of the squared variation "
                "is missed.",
            ),
            (
                "Classes",
                "Which class is rebuilt worst?",
                "The defaults; four records of each class.",
                "Over five seeds the share missed is 17 to 18% for TR, 27 to 28% "
                "for VT, 40 to 41% for TC and 52 to 54% for LP. A steady tremor has "
                "about the same spectrum all along the record and is cheap to "
                "describe; LP and TC differ from record to record and cost more.",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What the autoencoder is for

    The decoder and the target are only the exercise: they force the encoder to put what
    matters about a record into $\mathbf{z}$. Afterwards we keep $\mathbf{z}$
    """)
    return


@app.cell(hide_code=True)
def _(
    AutoencoderFlow,
    ae_run,
    llaima,
    llaima_spec,
    mo,
    spec_fmax,
    spec_fmin,
    test_idx,
    to_db,
    u,
):
    # real numbers of the trained autoencoder, for the test records: the waveform, its
    # latent vector, the spectrogram the decoder rebuilds from it and the target
    mo.ui.anywidget(
        AutoencoderFlow(
            llaima.classes,
            u[test_idx, 0],
            ae_run.latent,
            llaima.label[test_idx],
            to_db(llaima_spec[test_idx]),
            to_db(ae_run.rebuilt),
            index=test_idx,
            duration=u.shape[-1] / llaima.sampling_rate,
            fmin=spec_fmin,
            fmax=spec_fmax,
        )
    )
    return


@app.cell(hide_code=True)
def _(ae_run, mo, plots):
    mo.vstack(
        [
            mo.md(r"""
            ## Where the latent vector puts the records
            """),
            mo.hstack(
                [plots.plot_corner(ae_run.latent, title="no labels", size=8.0)],
                justify="center",
            ),
        ],
        gap=0,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A few labels are enough

    Records of one kind should lie close together in the latent space. Three steps
    use that, with a handful of labels

    - **k-means** finds $k$ clusters of latent vectors, without labels: it alternates
      between giving every record to its nearest centre and moving every centre to
      the mean of its records
    - each cluster is **named** by looking up the true label of five of its records,
      the majority wins
    - a record takes the name of its nearest cluster
    """)
    return


@app.cell
def _(torch):
    def kmeans(z, k, restarts=10, iters=50):
        """Lloyd's algorithm with k-means++ starts; keeps the best of ``restarts``."""
        torch.manual_seed(0)
        best_inertia, best_centres = float("inf"), None
        for _ in range(restarts):
            centres = z[torch.randint(len(z), (1,))]
            for _ in range(k - 1):
                d2 = torch.cdist(z, centres).min(1).values ** 2
                pick = torch.multinomial(d2 / d2.sum(), 1)
                centres = torch.cat([centres, z[pick]])
            for _ in range(iters):
                assign = torch.cdist(z, centres).argmin(1)
                centres = torch.stack(
                    [
                        z[assign == j].mean(0) if (assign == j).any() else centres[j]
                        for j in range(k)
                    ]
                )
            inertia = torch.cdist(z, centres).min(1).values.pow(2).sum().item()
            if inertia < best_inertia:
                best_inertia, best_centres = inertia, centres
        return best_centres

    def nearest(z, centres):
        return torch.cdist(z, centres).argmin(1)

    def name_clusters(assign, label, k, per_cluster, n_classes, seed=0):
        """Give each cluster the majority label of a few of its records.

        Returns the name of every cluster and the indices of the records whose
        label was looked up."""
        torch.manual_seed(seed)
        names, asked = torch.zeros(k, dtype=torch.long), []
        for j in range(k):
            members = (assign == j).nonzero().squeeze(1)
            members = members[torch.randperm(len(members))][:per_cluster]
            if len(members):
                names[j] = torch.bincount(label[members], minlength=n_classes).argmax()
            asked.append(members)
        return names, torch.cat(asked)

    return kmeans, name_clusters, nearest


@app.cell
def _(
    SimpleNamespace,
    ae_run,
    kmeans,
    llaima,
    name_clusters,
    nearest,
    torch,
    train_idx,
    u,
):
    with torch.no_grad():
        latent_train = ae_run.model.encoder(u[train_idx])
    label_train = llaima.label[train_idx]

    # Every number of clusters, and 30 draws of the five labelled records of each
    # cluster, are made here once, so that the playground below switches at once.
    cluster_fits = {}
    for _k in range(2, 17):
        _centres = kmeans(latent_train, _k)  # k-means on the training latent vectors
        _assign = nearest(latent_train, _centres)
        # five labelled records per cluster name it (a draw is one seed)
        _named = [
            name_clusters(_assign, label_train, _k, 5, len(llaima.classes), seed=_s)
            for _s in range(30)
        ]
        cluster_fits[_k] = SimpleNamespace(
            centres=_centres,
            assign_train=_assign,
            assign_test=nearest(ae_run.latent, _centres),
            names=[_names for _names, _ in _named],
            asked=[_asked for _, _asked in _named],
        )
    return cluster_fits, latent_train


@app.cell(hide_code=True)
def _(
    ClusterNaming,
    ae_run,
    cluster_fits,
    latent_train,
    llaima,
    mo,
    test_idx,
    train_idx,
):
    # the playground: k and the draw are set in it, the rest of the notebook reads them
    cluster_widget = mo.ui.anywidget(
        ClusterNaming(
            llaima.classes,
            latent_train,
            llaima.label[train_idx],
            ae_run.latent,
            llaima.label[test_idx],
            cluster_fits,
            index_train=train_idx,
            index_test=test_idx,
            # drawn in the widget's free corner: the widget is too tall for text below
            explore=[
                (
                    "Clusters",
                    "What does the number of clusters change, and when do the labels "
                    "stop improving?",
                ),
                (
                    "Classes",
                    "Does every class get a cluster of its own, and which do not?",
                ),
                (
                    "Mixed clusters",
                    "Which clusters are not unanimous, and what do the true labels of "
                    "all their records say?",
                ),
                (
                    "Sample",
                    "Does the name of a cluster depend on which five records were "
                    "looked up?",
                ),
            ],
        )
    )
    return (cluster_widget,)


@app.cell
def _(cluster_fits, cluster_widget, llaima, test_idx, torch):
    cl_k = cluster_widget.value["k"]  # the number of clusters chosen above
    cl_sample = cluster_widget.value["sample"]  # which draw of the labelled records
    centres = cluster_fits[cl_k].centres
    assign_test = cluster_fits[cl_k].assign_test
    # a cluster is named by the majority of its five labelled records, and a test
    # record takes the name of its nearest cluster
    cluster_names = cluster_fits[cl_k].names[cl_sample]
    recovered = cluster_names[assign_test]
    cluster_accuracy = float((recovered == llaima.label[test_idx]).float().mean())
    # the share of the commonest class among the test records
    test_majority = float(torch.bincount(llaima.label[test_idx]).max()) / len(test_idx)
    return (
        assign_test,
        centres,
        cl_k,
        cluster_accuracy,
        cluster_names,
        recovered,
        test_majority,
    )


@app.cell(hide_code=True)
def _(cluster_widget, mo):
    mo.vstack([cluster_widget])
    return


@app.cell(hide_code=True)
def _(
    ae_run,
    centres,
    cluster_accuracy,
    cluster_names,
    llaima,
    mo,
    plots,
    recovered,
    test_majority,
):
    mo.vstack(
        [
            mo.md(r"""
            ## Every record gets the label of its cluster
            """),
            mo.hstack(
                [
                    plots.plot_corner(
                        ae_run.latent,
                        recovered,
                        llaima.classes,
                        centres,
                        "numbers: cluster centres",
                        size=8.0,
                    )
                ],
                justify="center",
            ),
            mo.md(
                f"{len(set(cluster_names.tolist()))} of the "
                f"{len(llaima.classes)} classes received a cluster, and "
                f"**{100 * cluster_accuracy:.0f}%** of the test records get their "
                f"right label; always answering with the commonest class gives "
                f"{100 * test_majority:.0f}%."
            ),
        ],
        gap=0,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How the playground works

    *Clusters* is the $k$ of k-means, run on the latent vectors of the trained
    autoencoder (train it again above with a new seed to see how much depends on the
    network). *Sample* is which five records of every cluster are looked up; 30
    samples are drawn in advance and the first is the one the rest of the notebook
    starts from. *Horizontal* and *vertical* choose which two of the four latent
    numbers the panels show: the clusters are made from all four, so overlap in one
    view is no surprise. *Records* chooses the training or the test records.

    The three panels show the same records at the same places. **Clusters, found
    without labels**: each record in the colour of its cluster (the colours mean
    nothing), the cluster centres as numbered discs and, for training records, the
    records whose label was looked up ringed. **Clusters, named**: each record in the
    colour of the label its cluster gives it (clusters with the same name differ in
    how light the colour is); with test records a ring marks the wrong ones. **True
    labels**: the colour of the true label, which the method never uses.

    The table has one row per cluster: the true labels of its five records (the
    vote), the name the majority gives it and how many of the five agree, then the
    number of its test records and the share of them that get the right label. The
    numbers at the top count the labels spent, the classes that received a cluster
    and the share of test records with the right label, next to the share of the
    commonest class.

    Point at a cluster (a disc, a row or a vote dot) to mark it in all three panels,
    with the true labels of all its records below; point at a record to see its label
    and its latent vector. A click keeps a cluster in focus; click again, or press
    Escape, to let it go.
    """)
    return


@app.cell(hide_code=True)
def _(try_it):
    try_it(
        [
            (
                "Clusters",
                "When do the labels stop improving?",
                "*Clusters* 4, 8 and 16 with *sample* 1, on the autoencoder with the "
                "defaults; read *test records right*.",
                "Over five seeds, 4 clusters name 65 to 81% of the test records "
                "right, 8 clusters 83 to 88% and 16 clusters 84 to 88%. Past about "
                "8 clusters it stops improving.",
            ),
            (
                "Classes",
                "Does every class get a cluster of its own?",
                "*Clusters* 4, then 8; read *classes named*, look at the names in the "
                "right figure and, below, at the corner plot and the confusion matrix.",
                "At 4 clusters the rare class VT usually gets no cluster of its own: "
                "with sample 1, four of five seeds name 2 or 3 of the 4 classes and "
                "none of the VT records are right, because k-means follows the large "
                "spread of LP and TC, not the classes (one seed does find VT). At 8 "
                "clusters all four classes are named in nearly every draw, and 62 to "
                "74% of the VT records are right.",
            ),
            (
                "Mixed clusters",
                "Which clusters are not unanimous, and where do two classes share a "
                "cluster?",
                "*Clusters* 8 with *sample* 1; look at the votes in the right figure, "
                "point at the clusters that are not unanimous, and compare the "
                "middle panel with the one showing the true labels.",
                "With the defaults, one cluster wins its vote 5 of 5, five win it "
                "4 of 5 and two only 3 of 5; at 4 clusters the weakest vote is a "
                "split of 2, 2 and 1. LP and TC overlap: cluster 0 (named TC) holds "
                "10 LP and 5 TR records among its 42 test records and cluster 2 "
                "(named TC) 11 LP among its 81, while the VT cluster 7 and the TR "
                "cluster 5 are nearly pure.",
            ),
            (
                "Sample",
                "Does the name of a cluster depend on which five records were looked "
                "up?",
                "*Clusters* 4, then 8 and 16; step *sample* through its 30 values and "
                "watch *test records right*.",
                "With the defaults, over the 30 samples the share of test records "
                "right runs from 53 to 80% at 4 clusters (median 80%), from 60 to "
                "88% at 8 (median 88%) and from 78 to 87% at 16 (median 86%). A bad "
                "sample names a cluster after a minority: at 8 clusters, sample 9 "
                "names cluster 4 TC, although 413 of its 458 training records are "
                "LP, and cluster 2 LP, although 291 of its 381 are TC; only 60% of "
                "the test records are right, against 88% for sample 1.",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where it goes wrong

    Each row is a true label, each column the label a cluster gave

    A record is right when it lies on the diagonal
    """)
    return


@app.cell(hide_code=True)
def _(llaima, metrics, plots, recovered, test_idx):
    _cm = metrics.confusion_matrix(
        llaima.label[test_idx], recovered, len(llaima.classes)
    )
    plots.plot_confusion(
        _cm, llaima.classes, metrics.class_report(_cm), figsize=(11.0, 4.6)
    )
    return


@app.cell(hide_code=True)
def _(assign_test, cl_k, cluster_names, llaima, mo, torch):
    cl_pick = mo.ui.dropdown(
        {
            f"{j}: {llaima.classes[int(cluster_names[j])]} ({n} records)": j
            for j, n in enumerate(torch.bincount(assign_test, minlength=cl_k).tolist())
        },
        value=f"0: {llaima.classes[int(cluster_names[0])]} "
        f"({int((assign_test == 0).sum())} records)",
        label="cluster",
    )
    cl_again = mo.ui.button(value=0, on_click=lambda v: v + 1, label="Another sample")
    return cl_again, cl_pick


@app.cell(hide_code=True)
def _(
    assign_test,
    cl_again,
    cl_pick,
    llaima,
    llaima_spec,
    mo,
    plots,
    spec_fmax,
    spec_fmin,
    spec_hop,
    spec_window,
    test_idx,
    to_db,
    torch,
    u,
):
    _members = (assign_test == cl_pick.value).nonzero().squeeze(1)
    _generator = torch.Generator().manual_seed(cl_again.value)
    _shown = _members[torch.randperm(len(_members), generator=_generator)][:6]
    mo.vstack(
        [
            mo.hstack([cl_pick, cl_again], justify="center"),
            mo.hstack(
                [
                    plots.plot_cluster_members(
                        u[test_idx[_shown], 0],
                        to_db(llaima_spec[test_idx[_shown]]),
                        llaima.label[test_idx[_shown]],
                        llaima.classes,
                        spec_hop,
                        llaima.sampling_rate,
                        spec_window,
                        spec_fmax,
                        spec_fmin,
                        figsize=(13.0, 5.0),
                    )
                ],
                justify="center",
            ),
            mo.md(r"""
    ### Three things to look for

    - **Members.** What do the records of one cluster have in common, in waveform and
      in spectrogram?
    - **Mixed.** In which clusters do the true labels disagree, and what do the
      odd ones out look like?
    - **Same name.** Do clusters that received the same label look alike?
            """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How the playground works

    *Cluster* picks one of the clusters of the playground above, with the label it
    received and the number of test records in it. The figure shows six of its
    records, each titled with its **true** label: waveform above, spectrogram below.
    *Another sample* draws six others. Changing *clusters* above rebuilds the list.
    """)
    return


@app.cell(hide_code=True)
def _(try_it):
    try_it(
        [
            (
                "Members",
                "What do the records of one cluster have in common?",
                "*Clusters* 8; look at the clusters named VT and TR, then at a TC one.",
                "With the defaults, the VT cluster has a sharp onset and a decaying "
                "coda in every record; the TR cluster is steady along the record, "
                "with its energy at low frequency.",
            ),
            (
                "Mixed",
                "Which cluster is the most mixed?",
                "*Clusters* 8; press *Another sample* a few times in each cluster.",
                "With the defaults, cluster 0 (named TC) shows LP records among its "
                "members, 10 of its 42 test records; press the button a few "
                "times to meet them.",
            ),
            (
                "Same name",
                "Do clusters with the same label look alike?",
                "*Clusters* 8; compare the clusters named LP, then those named TC.",
                "Relative to the average record, the two LP clusters both stand out "
                "at about 4 Hz, two of the four TC clusters at about 9 and 12 Hz "
                "while the other two are close to the average record. One label is "
                "several kinds of record, which is why more clusters than classes "
                "help.",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A variational autoencoder

    Our latent vector is a point: one record gives one $\mathbf{z}$, and nothing says
    what lies between two points

    A **variational autoencoder** gives every record a distribution instead

    $$
    q(\mathbf{z} \mid \mathbf{u}) = \mathcal{N}\big(\boldsymbol{\mu}(\mathbf{u}),
      \mathbf{C}_Z(\mathbf{u})\big)
    $$

    and trains on the rebuild error plus a penalty that pulls every $q$ towards a
    standard Gaussian

    $$
    S = \text{rebuild error} + \mathrm{KL}\big(q(\mathbf{z} \mid \mathbf{u}) \,\Vert\,
    \mathcal{N}(\mathbf{0}, \mathbf{I})\big)
    $$

    That gives us what a plain autoencoder lacks:

    - draw $\mathbf{z}$ from the standard Gaussian and decode it: **new** spectrograms
    - a density over latent vectors: how **usual** is a record, and which ones are not
    - the step of Part 1c again: the network returns a distribution, not a point
    """)
    return


if __name__ == "__main__":
    app.run()
