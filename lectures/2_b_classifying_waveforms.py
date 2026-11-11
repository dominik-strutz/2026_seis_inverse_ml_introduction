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
        "CJ9LHsidHlwZSI6InNraXAifSx7InR5cGUiOiJza2lwIn0seyJ0eXBlIjoic2xpZGUiLCJzaG9"
        "3Q29kZSI6ZmFsc2V9LHsidHlwZSI6ImZyYWdtZW50Iiwic2hvd0NvZGUiOmZhbHNlfSx7InR5c"
        "GUiOiJzbGlkZSIsInNob3dDb2RlIjpmYWxzZX0seyJ0eXBlIjoic2xpZGUiLCJzaG93Q29kZSI"
        "6ZmFsc2V9LHsidHlwZSI6InNraXAifSx7InR5cGUiOiJzbGlkZSIsInNob3dDb2RlIjpmYWxzZ"
        "X0seyJ0eXBlIjoic2tpcCJ9LHsidHlwZSI6InNsaWRlIiwic2hvd0NvZGUiOmZhbHNlfSx7InR"
        "5cGUiOiJmcmFnbWVudCIsInNob3dDb2RlIjpmYWxzZX0seyJ0eXBlIjoic2tpcCJ9LHsidHlwZ"
        "SI6ImZyYWdtZW50Iiwic2hvd0NvZGUiOmZhbHNlfSx7InR5cGUiOiJza2lwIn0seyJ0eXBlIjo"
        "ic2xpZGUiLCJzaG93Q29kZSI6ZmFsc2V9LHsidHlwZSI6ImZyYWdtZW50Iiwic2hvd0NvZGUiO"
        "mZhbHNlfSx7InR5cGUiOiJza2lwIn0seyJ0eXBlIjoic2xpZGUiLCJzaG93Q29kZSI6ZmFsc2V"
        "9LHsidHlwZSI6ImZyYWdtZW50Iiwic2hvd0NvZGUiOmZhbHNlfSx7InR5cGUiOiJzbGlkZSIsI"
        "nNob3dDb2RlIjpmYWxzZX0seyJ0eXBlIjoiZnJhZ21lbnQiLCJzaG93Q29kZSI6ZmFsc2V9LHs"
        "idHlwZSI6InNsaWRlIiwic2hvd0NvZGUiOmZhbHNlfSx7InR5cGUiOiJmcmFnbWVudCIsInNob"
        "3dDb2RlIjpmYWxzZX0seyJ0eXBlIjoic2xpZGUiLCJzaG93Q29kZSI6ZmFsc2V9LHsidHlwZSI"
        "6InNraXAifSx7InR5cGUiOiJzbGlkZSIsInNob3dDb2RlIjpmYWxzZX0seyJ0eXBlIjoic2tpc"
        "CJ9LHsidHlwZSI6ImZyYWdtZW50Iiwic2hvd0NvZGUiOmZhbHNlfSx7InR5cGUiOiJza2lwIn0"
        "seyJ0eXBlIjoic2tpcCJ9LHsidHlwZSI6InNsaWRlIiwic2hvd0NvZGUiOmZhbHNlfSx7InR5c"
        "GUiOiJza2lwIn0seyJ0eXBlIjoic2xpZGUiLCJzaG93Q29kZSI6ZmFsc2V9LHsidHlwZSI6ImZ"
        "yYWdtZW50Iiwic2hvd0NvZGUiOmZhbHNlfSx7InR5cGUiOiJzbGlkZSIsInNob3dDb2RlIjpmY"
        "WxzZX0seyJ0eXBlIjoiZnJhZ21lbnQiLCJzaG93Q29kZSI6ZmFsc2V9LHsidHlwZSI6InNsaWR"
        "lIiwic2hvd0NvZGUiOmZhbHNlfSx7InR5cGUiOiJmcmFnbWVudCIsInNob3dDb2RlIjpmYWxzZ"
        "X0seyJ0eXBlIjoic2xpZGUiLCJzaG93Q29kZSI6ZmFsc2V9LHsidHlwZSI6ImZyYWdtZW50Iiw"
        "ic2hvd0NvZGUiOmZhbHNlfSx7InR5cGUiOiJzbGlkZSIsInNob3dDb2RlIjpmYWxzZX0seyJ0e"
        "XBlIjoiZnJhZ21lbnQiLCJzaG93Q29kZSI6ZmFsc2V9LHsidHlwZSI6InNsaWRlIiwic2hvd0N"
        "vZGUiOmZhbHNlfSx7InR5cGUiOiJmcmFnbWVudCIsInNob3dDb2RlIjpmYWxzZX0seyJ0eXBlI"
        "joic2xpZGUiLCJzaG93Q29kZSI6ZmFsc2V9LHsidHlwZSI6InNsaWRlIiwic2hvd0NvZGUiOmZ"
        "hbHNlfV0sImRlY2siOnsidHJhbnNpdGlvbiI6Im5vbmUiLCJ2ZXJ0aWNhbEFsaWduIjoidG9wI"
        "n19fQ=="
    ),
    css_file="lecture_slides.css",
    html_head_file="slides_head.html",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import torch
    from torch import nn

    from ml_seis import waveform_metrics as metrics
    from ml_seis import waveform_plotting as plots
    from ml_seis.llaimawidget import LlaimaExplorer
    from ml_seis.slides import artifact, try_it
    from ml_seis.waveform_data import (
        LLAIMA_NAMES,
        LLAIMA_WINDOW,
        llaima_cache_path,
        load_llaima,
        prepare_llaima,
    )
    from ml_seis.waveform_setup import cache_ready, llaima_card

    return (
        LLAIMA_NAMES,
        LLAIMA_WINDOW,
        LlaimaExplorer,
        artifact,
        cache_ready,
        llaima_cache_path,
        llaima_card,
        load_llaima,
        metrics,
        mo,
        nn,
        np,
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
            # Part 2b: training a waveform classifier

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
    ### Recap from Part 2a

    The network of 2a: blocks of filters, batch normalisation and ReLU with max-pooling
    in between, an average over time and a linear layer. It returns one number per
    class. Here it is narrower, 8 filters in every block and no dropout, so that we
    can look at everything it learned afterwards. The training loop of Part 1b comes
    below, written for batches of records.
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

    return (WaveformCNN,)


@app.cell
def _(mo, torch):
    # more threads than physical cores can make the small convolutions many times
    # slower, and beyond six threads they gain nothing
    torch.set_num_threads(min(torch.get_num_threads(), 6))
    cache_dir = mo.notebook_dir().parent / "data" / "waveforms"
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
            mo.md("The Llaima records of 2a, with the same split:"),
            llaima_card(mo, llaima, llaima_cache, LLAIMA_NAMES),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One logit per class

    The four Llaima labels are **mutually exclusive**: every record has exactly one

    The network returns **one logit per class**: four real numbers $a_c$, positive or
    negative

    **Softmax** turns the logits into probabilities that add to one

    $$
    p_c = \frac{e^{a_c}}{\sum_{j} e^{a_j}}
    $$

    With only two classes, one logit is enough: the **sigmoid**
    """)
    return


@app.cell(hide_code=True)
def _(plots):
    plots.plot_sigmoid_softmax()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cross-entropy: the loss of a classifier

    Record $i$ has the label $c_i$; the network is good if it gives that class a
      **high probability**

    Training minimises the mean negative log-likelihood of the labels

    $$
    S = -\frac{1}{n} \sum_{i=1}^{n} \log p_{c_i}(\mathbf{u}_i)
    $$

    In machine learning this is called **cross-entropy**

    A confident wrong answer costs much more than a vague one
    """)
    return


@app.cell(hide_code=True)
def _(artifact):
    artifact("softmax_loss")
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
    ### Three things to look for: what could name a record's class?

    - **Frequency.** Which classes live in which frequency bands, and which overlap?
    - **The end of a record.** What comes after the signal, and does it depend on
      the class?
    - **Balance.** How common is each class?
        """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(llaima, try_it):
    # the share of the records of each class that end in a flat padding
    _padded = llaima.n_live < llaima.u.shape[-1]
    _share = {
        c: 100 * float(_padded[llaima.label == k].float().mean())
        for k, c in enumerate(llaima.classes)
    }
    _counts = llaima.counts("train")
    _rare = int(_counts.argmin())
    try_it(
        [
            (
                "The end of a record",
                "Which classes end in a flat line, and how often?",
                "Page through a few records of every class and read the flat share "
                "under each record.",
                "Records that end in padding: "
                + ", ".join(f"`{c}` {s:.0f}%" for c, s in _share.items())
                + ". The flat line is padding, added to records shorter than 60 s.",
            ),
            (
                "Balance",
                "Which class is the rarest in the training records?",
                "Read the bars on the left of the explorer, or filter the grid by "
                "train.",
                f"`{llaima.classes[_rare]}` with {int(_counts[_rare])} of "
                f"{int(_counts.sum())} training records.",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The padding is a shortcut

    Records shorter than the 60 s window are padded with a flat line
    """)
    return


@app.cell(hide_code=True)
def _(LLAIMA_WINDOW, llaima, plots):
    plots.plot_live_length(
        llaima.n_live, llaima.label, llaima.classes, llaima.sampling_rate, LLAIMA_WINDOW
    )
    return


@app.cell
def _(llaima, metrics):
    # classify by how long the signal lasts, nothing else (bins of 3 s)
    _train, _test = llaima.indices("train"), llaima.indices("test")
    length_pred = metrics.length_baseline(
        llaima.n_live[_train],
        llaima.label[_train],
        llaima.n_live[_test],
        list(range(150, 3008, 150)),
        len(llaima.classes),
    )
    length_accuracy = float((length_pred == llaima.label[_test].numpy()).mean())
    return (length_accuracy,)


@app.cell(hide_code=True)
def _(LLAIMA_WINDOW, length_accuracy, llaima, mo):
    _seconds = LLAIMA_WINDOW / llaima.sampling_rate
    _dropped = len(llaima.indices("train")) - len(
        llaima.indices("train", min_live=LLAIMA_WINDOW)
    )
    mo.md(f"""
    ## Cutting out the shortcut

    The length alone names the class: a rule that sees only the length of the
      signal (bins of 3 s) is right for **{100 * length_accuracy:.0f}%** of the test
      records

    Network gets only the **first {_seconds:.0f} s** of every record
      ({LLAIMA_WINDOW} samples)

    The onset is in it (usually within the first 5 s) and it holds no padding

    {_dropped} training records with less signal are left out
    """)
    return


@app.cell
def _(LLAIMA_WINDOW, llaima):
    window = llaima.window()  # the first 15 s of every record, (N, 1, 750)
    train_idx = llaima.indices("train", min_live=LLAIMA_WINDOW)
    val_idx = llaima.indices("validation", min_live=LLAIMA_WINDOW)
    test_idx = llaima.indices("test", min_live=LLAIMA_WINDOW)
    return test_idx, train_idx, val_idx, window


@app.cell(hide_code=True)
def _(llaima, mo):
    _counts = llaima.counts("train")
    _weights = _counts.sum() / (len(_counts) * _counts)
    _list = ", ".join(
        f"`{c}` {float(w):.2f}" for c, w in zip(llaima.classes, _weights, strict=True)
    )
    mo.md(rf"""
    ## When one class is rare

    `{llaima.classes[0]}` and `{llaima.classes[3]}` are common
      ({int(_counts[0])} and {int(_counts[3])} training records),
      `{llaima.classes[2]}` is rare ({int(_counts[2])})

    The network can lower the loss a lot by getting the common classes right

    **Class weights** are the usual remedy: a record of class $c$ counts
      $w_c = n / (4\,n_c)$ times, with $n_c$ the number of training records of
      that class

    $$
    S_w = -\frac{{1}}{{n}} \sum_{{i=1}}^{{n}} w_{{c_i}} \log p_{{c_i}}(\mathbf{{u}}_i)
    $$

    Here the weights are {_list}

    They move errors from one class to another more than they remove them: we train
      **without** them and read the per-class numbers instead

    Keep in mind the training data is also a prior distribution!
    """)
    return


@app.cell(hide_code=True)
def _(llaima, plots):
    plots.plot_class_balance(llaima)
    return


@app.cell
def _(llaima, metrics, np, test_idx):
    # Three confusion matrices of the 535 test records (rows: label, columns:
    # prediction). The first two are made up, to look like real classifiers.
    cm_good = np.array([[185, 3, 1, 7], [5, 63, 0, 5], [6, 0, 33, 4], [9, 2, 1, 211]])
    cm_never_vt = np.array(
        [[185, 3, 0, 8], [5, 63, 0, 5], [27, 0, 0, 16], [9, 2, 0, 212]]
    )
    # the third is real: the network that always answers with the commonest class
    commonest = int(llaima.counts().argmax())
    cm_always = metrics.confusion_matrix(
        llaima.label[test_idx], [commonest] * len(test_idx), len(llaima.classes)
    )
    return cm_always, cm_good, cm_never_vt, commonest


@app.cell(hide_code=True)
def _(cm_good, llaima, metrics, mo):
    _vt = llaima.classes.index("VT")
    _report = metrics.class_report(cm_good)
    mo.md(f"""
    ## Accuracy, recall and precision

    **Confusion matrix**: one row per label, one column per answer of the network;
      each entry counts the test records with that label that got that answer, and
      the right answers sit on the diagonal

    **Accuracy**: of all test records, the share the network answers correctly
      (the diagonal over everything)

    **Recall** of a class: of all records that truly have this label, the share the
      network finds (the diagonal entry over its row). It asks: when this event
      happens, do we catch it?

    **Precision** of a class: of all records the network calls this class, the share
      that truly have this label (the diagonal entry over its column). It asks: when
      the network raises this alarm, can we trust it?

    An example with made-up numbers for {int(cm_good.sum())} test records: `VT` has
    **recall {100 * _report["recall"][_vt]:.0f}%** (the row: {cm_good[_vt, _vt]} of
    {cm_good[_vt].sum()} found) and **precision {100 * _report["precision"][_vt]:.0f}%**
    (the column: {cm_good[_vt, _vt]} of {cm_good[:, _vt].sum()} called `VT` are
    right)
    """)
    return


@app.cell(hide_code=True)
def _(cm_good, llaima, metrics, plots):
    plots.plot_confusion(
        cm_good,
        llaima.classes,
        metrics.class_report(cm_good),
        highlight=llaima.classes.index("VT"),
    )
    return


@app.cell(hide_code=True)
def _(cm_never_vt, llaima, metrics, mo):
    _vt = llaima.classes.index("VT")
    _report = metrics.class_report(cm_never_vt)
    mo.md(f"""
    ## A rare class can hide

    This network never answers `VT`, and still has an accuracy of
      **{100 * _report["accuracy"]:.0f}%**

    Recall of `VT` is {100 * _report["recall"][_vt]:.0f}%: all
      {cm_never_vt[_vt].sum()} records of the class are called something else
    """)
    return


@app.cell(hide_code=True)
def _(cm_never_vt, llaima, metrics, plots):
    plots.plot_confusion(
        cm_never_vt,
        llaima.classes,
        metrics.class_report(cm_never_vt),
    )
    return


@app.cell(hide_code=True)
def _(cm_always, commonest, llaima, metrics, mo):
    _report = metrics.class_report(cm_always)
    _name = llaima.classes[commonest]
    mo.md(f"""
    ## Always answering {_name}

    `{_name}` is the commonest class, {100 * _report["accuracy"]:.0f}% of the
      test records

    A network that always says `{_name}` has an accuracy of
      **{100 * _report["accuracy"]:.0f}%** without looking at a waveform, recall
      {100 * _report["recall"][commonest]:.0f}% for `{_name}` and 0% for every other
      class
    """)
    return


@app.cell(hide_code=True)
def _(cm_always, llaima, metrics, plots):
    plots.plot_confusion(cm_always, llaima.classes, metrics.class_report(cm_always))
    return


@app.cell(hide_code=True)
def _(LLAIMA_WINDOW, llaima, mo):
    _seconds = LLAIMA_WINDOW / llaima.sampling_rate
    mo.md(f"""
    ## Training a classifier

    The loop of Part 1b, on random mini-batches of the {_seconds:.0f} s windows:

    - forward pass and cross-entropy loss
    - backward pass and one step of Adam
    - after every epoch, the loss on the validation records
    - the **learning rate** falls smoothly from $10^{{-3}}$ to nearly zero over the
      epochs, so the last steps are small and the final network does not depend on
      the last mini-batches

    **Batch normalisation** rescales the output of each filter and helps a
    convolutional network train

    The test records stay untouched until the network is trained.
    """)
    return


@app.cell
def _(mo, nn, torch):
    def train(model, u, label, idx_train, idx_val, epochs=40, augment=None):
        """Train ``model`` on the records ``u[idx_train]`` with their ``label``.

        ``augment(idx)`` may give altered versions of those records instead. Returns
        the training and validation loss after every epoch."""
        loss_fn = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        # the learning rate falls from 1e-3 to about 0 over the epochs, like a cosine
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
        history = {"train": [], "validation": []}
        for _ in mo.status.progress_bar(
            range(epochs), title="Training", remove_on_exit=True
        ):
            model.train()
            order = idx_train[torch.randperm(len(idx_train))]  # shuffle every epoch
            losses = []
            for i in range(0, len(order), 64):
                idx = order[i : i + 64]  # 1. a mini-batch of 64 records
                u_batch = u[idx] if augment is None else augment(idx)
                optimiser.zero_grad()
                loss = loss_fn(model(u_batch), label[idx])  # 2. forward pass and loss
                loss.backward()  # 3. backward pass: the gradient of Part 1a
                optimiser.step()  # 4. one step of Adam
                losses.append(loss.item())
            scheduler.step()  # a smaller learning rate for the next epoch
            model.eval()
            with torch.no_grad():  # 5. the validation records, never used for a step
                validation = loss_fn(model(u[idx_val]), label[idx_val]).item()
            history["train"].append(sum(losses) / len(losses))
            history["validation"].append(validation)
        return history

    def classify(model, u):
        """The class probabilities of the records ``u``, (N, classes)."""
        model.eval()
        with torch.no_grad():
            return torch.softmax(model(u), dim=1)

    return classify, train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Augmentation

    A network always fits the training records a little better than new ones

    Augmentation works against that: every epoch the training windows are altered a
      little, and **the class does not change**

    * **Shift**: the window starts a random 0 to 1 s later, the onset is not at a fixed
      time

    * **Flip**: the polarity is reversed at random, the sign of a record depends on
      the station

    Validation and test windows are never altered
    """)
    return


@app.cell
def _(llaima, mo, test_idx):
    # test records to page through, one class after the other: LP, TR, VT, TC, LP, ...
    _by_class = [test_idx[llaima.label[test_idx] == k][:10] for k in range(4)]
    showcase = [int(c[i]) for i in range(10) for c in _by_class]
    record_slider = mo.ui.slider(0, len(showcase) - 1, label="record", show_value=True)
    return record_slider, showcase


@app.cell(hide_code=True)
def _(llaima, mo, plots, record_slider, showcase):
    _j = showcase[record_slider.value]
    mo.vstack(
        [
            record_slider,
            plots.plot_augmentations(
                llaima.u[_j, 0],
                llaima.sampling_rate,
                max_shift=50,
                seed=_j,
                name=f"({llaima.classes[int(llaima.label[_j])]})",
            ),
        ]
    )
    return


@app.cell
def _(LLAIMA_WINDOW, llaima, torch, window):
    def augment(idx, shift=True, flip=True):
        """The training windows of the records ``idx``, altered at random."""
        u = window[idx]
        if shift:  # start 0 to 50 samples (1 s) later, never inside the padding
            room = (llaima.n_live[idx] - LLAIMA_WINDOW).clamp(max=50)
            start = (torch.rand(len(idx)) * (room + 1)).long()
            columns = start[:, None] + torch.arange(LLAIMA_WINDOW)
            u = torch.gather(llaima.u[idx, 0], 1, columns)[:, None]
        if flip:  # reverse the polarity of about half of the records
            u = u * torch.where(torch.rand(len(idx), 1, 1) < 0.5, -1.0, 1.0)
        return u

    return (augment,)


@app.cell
def _(mo):
    settings_form = mo.ui.batch(
        mo.md("**Augmentation** {shift} {flip}"),
        {
            "shift": mo.ui.switch(True, label="shift"),
            "flip": mo.ui.switch(True, label="flip"),
        },
    ).form(submit_button_label="Retrain")
    return (settings_form,)


@app.cell
def _(
    WaveformCNN,
    augment,
    llaima,
    mo,
    plots,
    settings_form,
    torch,
    train,
    train_idx,
    val_idx,
    window,
):
    # trains as soon as the notebook runs; the form retrains with other settings
    settings = settings_form.value or {"shift": True, "flip": True}
    torch.manual_seed(0)
    model = WaveformCNN(n_out=4)
    history = train(
        model,
        window,
        llaima.label,
        train_idx,
        val_idx,
        augment=lambda idx: augment(idx, **settings),
    )
    run_name = "augmented" if any(settings.values()) else "plain"
    mo.vstack(
        [
            mo.md("## Training and validation loss"),
            settings_form,
            plots.plot_history_comparison({run_name: history}),
        ]
    )
    return (model,)


@app.cell
def _(classify, llaima, metrics, model, test_idx, window):
    test_label = llaima.label[test_idx]
    test_prob = classify(model, window[test_idx])
    cm = metrics.confusion_matrix(test_label, test_prob.argmax(1), len(llaima.classes))
    report = metrics.class_report(cm)
    return cm, report, test_label, test_prob


@app.cell(hide_code=True)
def _(llaima, mo, report):
    _worst = int(report["recall"].argmin())
    mo.md(f"""
    ## Reading the result

    **Accuracy {100 * report["accuracy"]:.1f}%** on the test records, against
      {100 * report["majority_accuracy"]:.1f}% for always answering with the commonest
      class

    Recall is lowest for `{llaima.classes[_worst]}`
      ({100 * report["recall"][_worst]:.0f}%)

    Precision and recall of the same class can differ: that is the imbalance at work
    """)
    return


@app.cell(hide_code=True)
def _(cm, llaima, plots, report):
    plots.plot_confusion(cm, llaima.classes, report)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The most confident mistakes

    The most informative failures are those where the network was sure and wrong

    Look at the record itself: is the label doubtful, or is the signal unusual?
    """)
    return


@app.cell(hide_code=True)
def _(llaima, metrics, plots, test_idx, test_label, test_prob, window):
    plots.plot_probability_gallery(
        window[test_idx],
        test_prob,
        test_label,
        llaima.classes,
        metrics.confident_mistakes(test_prob, test_label, k=4),
        llaima.sampling_rate,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What did the filters learn?

    The filters of the first block are short waveforms of their own

    We can draw them like the filters of 2a and compare them with the presets there

    Which are smooth, which look like edges, which oscillate?
    """)
    return


@app.cell(hide_code=True)
def _(llaima, model, plots):
    plots.plot_filters(model.features[0][0].weight.detach(), llaima.sampling_rate)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where the blocks react

    A feature map says where in the record a filter's pattern is present

    Each row is one filter, block by block; deeper blocks see longer structures, at
      a coarser time resolution

    The last block is averaged over time into the features
    """)
    return


@app.cell(hide_code=True)
def _(artifact, classify, llaima, metrics, model, showcase, window):
    # the 40 test records of the sliders above, with the maps of all four blocks
    artifact(
        "cnn_activations",
        **metrics.activation_records(
            model.features,
            window[showcase],
            llaima.label[showcase],
            classify(model, window[showcase]),
            llaima.classes,
            n_blocks=4,
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Internal Features

    The last block ends in 8 numbers per record, the **features**: the average over
      time of its 8 feature maps

    The linear layer sees only these 8 numbers, so they are the network's
      description of the record

    We extract them and leave out the linear layer
    """)
    return


@app.cell(hide_code=True)
def _(LLAIMA_WINDOW, WaveformCNN, artifact, metrics):
    artifact(
        "cnn_features",
        layers=metrics.layer_spec(WaveformCNN(n_out=4).features, LLAIMA_WINDOW),
        record_samples=LLAIMA_WINDOW,
        n_out=4,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Internal Features

    Project the 8 numbers onto their two strongest directions of variation

    Each test record is coloured by its label

    Where classes overlap, the network is unsure
    """)
    return


@app.cell(hide_code=True)
def _(llaima, metrics, model, plots, test_idx, test_label, torch, window):
    model.eval()
    with torch.no_grad():
        _features = model.features(window[test_idx])
    plots.plot_embedding(
        metrics.pca_2d(_features),
        test_label,
        llaima.classes,
        title="features after the average over time (test records)",
    )
    return


@app.cell(hide_code=True)
def _(explorer, mo):
    # the same widget as above, so a selection made in one place shows in the other
    mo.vstack(
        [
            mo.md("## Back to the data explorer"),
            explorer,
            mo.md(
                "Which classes overlap here, and are they the ones that overlap in the "
                "network's own coordinates?"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, train_idx):
    mo.md(f"""
    ## Labels cost expert time

    - This classifier needed {len(train_idx):,} labelled training records
    - Part 2c asks what a network can learn from the records alone
    """)
    return


if __name__ == "__main__":
    app.run()
