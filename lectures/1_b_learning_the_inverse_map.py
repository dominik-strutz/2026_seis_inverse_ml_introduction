# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "anywidget>=0.9",
#     "marimo>=0.25.2",
#     "matplotlib>=3.9",
#     "scikit-fmm>=2024.5",
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

__generated_with = "0.25.1"
app = marimo.App(
    width="full",
    layout_file=(
        "data:application/json;base64,eyJ0eXBlIjoic2xpZGVzIiwiZGF0YSI6eyJjZWxscyI6W"
        "3sidHlwZSI6InNraXAifSx7InR5cGUiOiJzbGlkZSIsInNob3dDb2RlIjpmYWxzZX0seyJ0eXB"
        "lIjoic2tpcCJ9LHsidHlwZSI6InNraXAifSx7InR5cGUiOiJzbGlkZSIsInNob3dDb2RlIjpmY"
        "WxzZX0seyJ0eXBlIjoiZnJhZ21lbnQiLCJzaG93Q29kZSI6ZmFsc2V9LHsidHlwZSI6InNraXA"
        "ifSx7InR5cGUiOiJza2lwIn0seyJ0eXBlIjoic2xpZGUiLCJzaG93Q29kZSI6ZmFsc2V9LHsid"
        "HlwZSI6InNraXAifSx7InR5cGUiOiJza2lwIn0seyJ0eXBlIjoic2tpcCJ9LHsidHlwZSI6InN"
        "saWRlIiwic2hvd0NvZGUiOmZhbHNlfSx7InR5cGUiOiJzbGlkZSIsInNob3dDb2RlIjpmYWxzZ"
        "X0seyJ0eXBlIjoiZnJhZ21lbnQiLCJzaG93Q29kZSI6ZmFsc2V9LHsidHlwZSI6InNsaWRlIiw"
        "ic2hvd0NvZGUiOmZhbHNlfSx7InR5cGUiOiJmcmFnbWVudCIsInNob3dDb2RlIjpmYWxzZX0se"
        "yJ0eXBlIjoic2tpcCJ9LHsidHlwZSI6InNraXAifSx7InR5cGUiOiJza2lwIn0seyJ0eXBlIjo"
        "ic2xpZGUiLCJzaG93Q29kZSI6ZmFsc2V9LHsidHlwZSI6ImZyYWdtZW50Iiwic2hvd0NvZGUiO"
        "mZhbHNlfSx7InR5cGUiOiJzbGlkZSIsInNob3dDb2RlIjpmYWxzZX0seyJ0eXBlIjoiZnJhZ21"
        "lbnQiLCJzaG93Q29kZSI6ZmFsc2V9LHsidHlwZSI6InNsaWRlIiwic2hvd0NvZGUiOmZhbHNlf"
        "Sx7InR5cGUiOiJza2lwIn0seyJ0eXBlIjoiZnJhZ21lbnQiLCJzaG93Q29kZSI6ZmFsc2V9LHs"
        "idHlwZSI6InNsaWRlIiwic2hvd0NvZGUiOmZhbHNlfSx7InR5cGUiOiJzbGlkZSIsInNob3dDb"
        "2RlIjpmYWxzZX0seyJ0eXBlIjoiZnJhZ21lbnQiLCJzaG93Q29kZSI6ZmFsc2V9LHsidHlwZSI"
        "6InNraXAifSx7InR5cGUiOiJzbGlkZSIsInNob3dDb2RlIjpmYWxzZX0seyJ0eXBlIjoiZnJhZ"
        "21lbnQiLCJzaG93Q29kZSI6ZmFsc2V9LHsidHlwZSI6InNraXAifSx7InR5cGUiOiJzbGlkZSI"
        "sInNob3dDb2RlIjpmYWxzZX0seyJ0eXBlIjoiZnJhZ21lbnQiLCJzaG93Q29kZSI6ZmFsc2V9L"
        "HsidHlwZSI6InNsaWRlIiwic2hvd0NvZGUiOmZhbHNlfSx7InR5cGUiOiJza2lwIn0seyJ0eXB"
        "lIjoic2tpcCJ9LHsidHlwZSI6InNraXAifSx7InR5cGUiOiJzbGlkZSIsInNob3dDb2RlIjpmY"
        "WxzZX0seyJ0eXBlIjoic2tpcCJ9LHsidHlwZSI6InNraXAifSx7InR5cGUiOiJza2lwIn1dLCJ"
        "kZWNrIjp7InRyYW5zaXRpb24iOiJub25lIiwidmVydGljYWxBbGlnbiI6InRvcCJ9fX0="
    ),
    css_file="lecture_slides.css",
    html_head_file="slides_head.html",
)


@app.cell(hide_code=True)
def _():
    from functools import partial
    from types import SimpleNamespace

    import marimo as mo
    import torch
    from torch import nn

    from ml_seis import learning_plotting as lp
    from ml_seis import make_stations, physics, sample_prior
    from ml_seis.locatorwidget import (
        DEFAULTS,
        N_TEST,
        LocatorPlayground,
        Recorder,
        station_list,
    )
    from ml_seis.mlpstructurewidget import MLPStructure
    from ml_seis.mlpwidget import MLPPlayground
    from ml_seis.physics import center, simulate_picks, travel_times
    from ml_seis.plotting import mlp_structure_figure, prior_figure
    from ml_seis.posterior import posterior_rms
    from ml_seis.runtime import torch_threads
    from ml_seis.scene import observe, scene_from
    from ml_seis.slides import artifact, show_source, try_it

    return (
        DEFAULTS,
        LocatorPlayground,
        MLPPlayground,
        MLPStructure,
        N_TEST,
        Recorder,
        SimpleNamespace,
        artifact,
        center,
        lp,
        make_stations,
        mlp_structure_figure,
        mo,
        nn,
        observe,
        partial,
        physics,
        posterior_rms,
        prior_figure,
        sample_prior,
        scene_from,
        show_source,
        simulate_picks,
        station_list,
        torch,
        torch_threads,
        travel_times,
        try_it,
    )


@app.cell(hide_code=True)
def _(artifact, mo):
    mo.vstack(
        [
            mo.md(r"""
            # Part 1b: learning the inverse map

            ### Machine learning and inversion for seismologists
            """),
            artifact("title_figure", width="65%"),
        ],
        gap=0,
    )
    return


@app.cell(hide_code=True)
def _(center, mo, show_source, simulate_picks, travel_times):
    mo.vstack(
        [
            mo.md("Physics from Part 1a, unchanged:"),
            show_source(travel_times, center, simulate_picks),
        ]
    )
    return


@app.cell
def _(DEFAULTS, scene_from, station_list):
    # the playground's opening state: 5 stations, 1000 events, two layers of 16
    default_config = DEFAULTS | {"stations": station_list(5)}
    default_scene = scene_from(default_config)
    return default_config, default_scene


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What is a multilayer perceptron?

    The network that will learn $\tilde{\mathbf{d}} \mapsto \mathbf{m}$ is a
    **multilayer perceptron** (MLP): a chain of layers, each one a matrix, a bias
    and a fixed nonlinear function $\varphi$,

    $$
    \mathbf{h}^{(\ell)} = \varphi\!\left(\mathbf{W}^{(\ell)}\,
    \mathbf{h}^{(\ell-1)} + \mathbf{b}^{(\ell)}\right)
    \qquad \ell = 1, \dots, L
    $$

    started at $\mathbf{h}^{(0)} = \tilde{\mathbf{d}}$ and ended by a linear
    layer, $\mathbf{m} = \mathbf{W}^{(L+1)} \mathbf{h}^{(L)} +
    \mathbf{b}^{(L+1)}$.

    The entries of $\mathbf{W}$ and $\mathbf{b}$ are trainable
    **parameters**. Training adjusts them through stochastic gradient descent.
    """)
    return


@app.cell(hide_code=True)
def _(MLPStructure, mo, nn, torch):
    # a tiny example, 3 inputs and 2 outputs (m_x, m_y); the +/- buttons resize it
    torch.manual_seed(1)
    small_net = nn.Sequential(
        nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)
    )

    mo.ui.anywidget(
        MLPStructure.from_net(
            small_net,
            in_labels=["d\u0303_1", "d\u0303_2", "d\u0303_3"],
            out_labels=["m_x", "m_y"],
            editable=True,
        )
    ).style({"zoom": "0.7"})  # smaller on the slide
    return


@app.cell(hide_code=True)
def _(MLPPlayground, mo):
    mlp_playground = mo.ui.anywidget(MLPPlayground())
    runs = {}  # finished runs by settings: pressing train with old settings replays one
    return mlp_playground, runs


@app.cell
def _(mlp_playground, nn, runs, torch):
    # the drawn train and validation points (pairs x -> y) and the settings of the last
    # training: a press of train, or any change while RETRAIN is on change
    x_train, y_train, x_val, y_val, settings = mlp_playground.training_setup()
    if settings.key not in runs:
        torch.manual_seed(settings.seed)  # reset draws a new seed: new initial weights

        # Linear + activation for each hidden layer, then a linear output layer
        dims = [1, *settings.widths, 1]
        layers = []
        for n_in, n_out in zip(dims[:-1], dims[1:], strict=True):
            layers += [nn.Linear(n_in, n_out), settings.activation()]
        net = nn.Sequential(*layers[:-1])
        # fused: one kernel per step, faster for a network this small
        opt = torch.optim.Adam(net.parameters(), lr=settings.lr, fused=True)

        runs[settings.key] = mlp_playground.recorder(net, settings)  # snapshots
        for epoch in range(1, settings.epochs + 1):
            with (
                torch.no_grad()
            ):  # the validation points are only looked at, never fitted
                val_mse = ((net(x_val) - y_val) ** 2).mean()
            opt.zero_grad()
            mse = ((net(x_train) - y_train) ** 2).mean()  # S, averaged over the points
            mse.backward()
            opt.step()
            runs[settings.key].log(epoch, mse, val_mse)
    recorder = runs[settings.key]
    return (recorder,)


@app.cell(hide_code=True)
def _(mlp_playground, mo):
    mo.vstack(
        [
            mlp_playground,
            mo.md(r"""
    ### Four things to look for

    - **Flexibility.** What do width, depth and the activation each add to the curve a
      network can draw?
    - **Initialisation.** Does the same network always find the same fit?
    - **Held-out points.** What do the validation points show that the train loss
      cannot?
    - **Data.** How many points before the two losses agree?
        """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How the playground works

    One number in, one number out. Draw any curve $y(x)$ on the output plot (or pick an
    example under *data*) and a network is trained to pass through your points, with the
    mean squared error as $S$. **+ / −** set the **depth** and the **width** of each
    hidden layer; the header sets the **activation** $\varphi$, the step size and the
    number of epochs, and **RETRAIN** says whether the network trains again at every
    change (*on change*) or only when you press **TRAIN** (*on button*). Each tile is
    one neuron and shows its output $h(x)$. ▶ plays the training back from the initial
    weights, ⏭ steps, ↺ draws new initial weights. Hover
    an edge, a matrix cell or a neuron to see what it does, click to pin it.

    The **train** points (●) are what the network is fitted to. The **validation**
    points (○) are never fitted: only their error is computed, after every epoch, and
    drawn next to the train loss. Choose the set under *data* before you draw
    (shift-drag erases). The idea comes from the [TensorFlow
    Playground](https://playground.tensorflow.org).
    """)
    return


@app.cell(hide_code=True)
def _(mlp_playground, recorder):
    mlp_playground.show_run(recorder)  # the playground above plays it back
    return


@app.cell(hide_code=True)
def _(try_it):
    try_it(
        [
            (
                "Flexibility",
                "What does a second layer add that a wider one does not?",
                "Start with one unit and add more, then add a layer; step (⏭) through.",
                "One unit makes one bend. Each neuron of a second layer bends a mix of "
                "the layer before, so the pieces multiply.",
            ),
            (
                "Flexibility",
                "What shape does each activation build?",
                "Switch between ReLU, tanh and linear.",
                "ReLU: straight pieces. tanh: smooth steps. linear: a straight line, "
                "however wide or deep.",
            ),
            (
                "Initialisation",
                "Is every neuron used?",
                "Press ↺ a few times, with ReLU.",
                "New initial weights, a different fit. Some neurons are *dead*: flat "
                "tiles, zero output for every $x$, no gradient.",
            ),
            (
                "Held-out points",
                "When does the network chase the noise?",
                "Load *sine*, width 16, three layers, 3000 epochs.",
                "The train loss falls far below the noise in the data, the validation "
                "loss does not follow: overfitting.",
            ),
            (
                "Held-out points",
                "Where can the network be wrong without the losses showing it?",
                "Choose *validation* and draw where there are no train points.",
                "The fit is only checked where validation points are.",
            ),
            (
                "Data",
                "How many train points make the losses agree?",
                "Move TRAIN from 6 to 200 points.",
                "With few, the train loss goes to zero while the validation loss stays "
                "high; with many, the two come together.",
            ),
            (
                "Data",
                "How many validation points can you trust?",
                "Set VALIDATION to 3, then to 200.",
                "A few points give a number you cannot trust.",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What the curve fit showed

    - **Flexibility.** Networks can fit complex curves from basic components.
    - **Overfitting.** Networks can fit the noise in their training points.

    In the next example the network takes the arrival times at $N$ stations and
    returns 2 values (the epicentre). Data is generated synthetically.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The prior: where we expect events

    To fit a neural network the data can either be collected from real events or,
    if we understand the system sufficiently well, be generated synthetically.
    Since ground truth locations for earthquakes are hard to come by and seismic
    wave propagation is well understood, we will generate our dataset here
    synthetically. But we need to be fully aware that this is always a (big)
    simplification.

    Where we expect events *before* seeing any arrival times is described by the
    **prior distribution** $\rho_M(\mathbf{m})$.
    * The `uniform` prior puts events anywhere on the map
    * The `ridgecrest` prior is inspired by the 2019 Ridgecrest aftershocks
    """)
    return


@app.cell(hide_code=True)
def _(make_stations, prior_figure, sample_prior, torch):
    torch.manual_seed(0)
    prior_figure(
        "uniform",
        "ridgecrest",
        sample_prior("uniform", 1000),
        sample_prior("ridgecrest", 1000),
        *make_stations(5),  # the 5 stations the locator playground opens with
        titles=("uniform prior: 1000 events", "ridgecrest prior: 1000 events"),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From a source to a training pair

    To generate synthetic data, draw a source from the prior, then simulate the
    noisy centered picks it would produce:
    $$
    \mathbf{m} \sim \rho_M
    \qquad
    \tilde{\mathbf{d}} =
    \mathbf{C}\,\big(\mathbf{F}(\mathbf{m}) + \boldsymbol{\varepsilon}\big)
    $$

    The data set we use to train the network is a long list of pairs
    $(\tilde{\mathbf{d}}, \mathbf{m})$. Here we have the luxury that the simulator
    can make as many as we want.
    """)
    return


@app.cell(hide_code=True)
def _(default_config, default_scene, lp):
    lp.dataset_figure(default_scene, "uniform", n_events=default_config["n_train"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The simulator in code

    `observe` is the simulator: picks at the stations for a batch of sources, with fresh
    noise, centred over the working stations (a **muted** one reads 0). `make_dataset`
    draws training and validation events from the training prior, each from its own
    random stream, so the validation events stay fixed when the number of training
    events changes. It reads everything from `config`, the settings the playground's
    **train** button captured.
    """)
    return


@app.cell(hide_code=True)
def _(physics, show_source):
    show_source(physics.observe)
    return


@app.cell
def _(physics, sample_prior, scene_from, torch):
    N_VAL = 300  # validation events

    def make_dataset(config):
        """Training and validation events for the scene captured by "train"."""
        scene = scene_from(config)  # stations, forward operator F, sigma
        torch.manual_seed(0)  # the same events every time
        m_train = sample_prior(config["train_prior"], config["n_train"])
        d_train = physics.observe(scene.F, m_train, scene.sigma, scene.live)
        # validation has its own seed, so it does not change with n_train
        torch.manual_seed(1)
        m_val = sample_prior(config["train_prior"], N_VAL)
        d_val = physics.observe(scene.F, m_val, scene.sigma, scene.live)
        return scene, m_train, d_train, m_val, d_val

    return N_VAL, make_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train, validation, test

    A network that fits its training events does not necessarily generalise to unseen
    events. To test this, the dataset is split into three sets with different jobs.

    - **Train**: the network is fitted to these and to nothing else
    - **Validation**: held back during training, used to adapt training process
    - **Test**: used once, at the end, for an honest estimate on events the network
      has never seen
    """)
    return


@app.cell(hide_code=True)
def _(N_TEST, N_VAL, default_config, default_scene, lp):
    lp.split_figure(default_scene, (default_config["n_train"], N_VAL, N_TEST))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Picks in, location out

    The default network is a multilayer perceptron, with one input per station and
    two outputs, the epicentre:

    $$
    \hat{\mathbf{m}} = f\big(\tilde{\mathbf{d}}\big)
    $$

    This one has two hidden layers of 16 tanh neurons.
    """)
    return


@app.cell(hide_code=True)
def _(mlp_structure_figure, nn, torch):
    # the playground's opening network: 5 picks in, two hidden layers, (m_x, m_y) out
    torch.manual_seed(0)  # the same random initial weights every time
    locator = nn.Sequential(
        nn.Linear(5, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 2)
    )

    # draw it as a graph plus its weight matrices
    mlp_structure_figure(
        locator,
        in_labels=[rf"$\tilde{{d}}_{i}$" for i in range(1, 6)],
        out_labels=[r"$m_x$", r"$m_y$"],
        max_nodes=8,  # a "⋮" stands for the other neurons of a wide layer
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Put numbers on one scale

    * Network weights start at order one, so a network learns best when its inputs
      and outputs are of order one too.
    * Picks are seconds, locations tens of kilometres. Both are shifted
    and divided by their standard deviation

    $$
    \tilde{\mathbf{d}} \mapsto
    \frac{\tilde{\mathbf{d}}}{\mathrm{std}(\tilde{\mathbf{d}})}
    \qquad
    \mathbf{m} \mapsto
    \frac{\mathbf{m} - \mathrm{mean}(\mathbf{m})}{\mathrm{std}(\mathbf{m})}
    $$

    with the mean and standard deviation of the **training** events only.
    Validation and test events must be treated exactly like new data.
    """)
    return


@app.cell
def _(torch):
    class Scaling:
        """Picks (s) and locations (km) on a scale of about one, from the training
        set only. With ``normalise=False`` the numbers go in and out as they are."""

        def __init__(self, d_train, m_train, normalise=True):
            ones = torch.ones(2, dtype=torch.float64)
            self.d_scale = d_train.std() if normalise else 1.0
            self.m_mean = m_train.mean(0) if normalise else 0 * ones
            self.m_scale = m_train.std(0) if normalise else ones

        def inputs(self, d):  # centred picks -> network input
            return (d / self.d_scale).float()

        def targets(self, m):  # locations -> what the network should output
            return ((m - self.m_mean) / self.m_scale).float()

        def locations(self, z):  # network output -> locations (km)
            return self.m_mean + self.m_scale * z.double()

    return (Scaling,)


@app.cell(hide_code=True)
def _(Scaling, default_config, lp, make_dataset):
    _, _m, _d, _, _ = make_dataset(default_config)
    _scaling = Scaling(_d, _m)
    lp.scaling_figure(_d, _scaling.inputs(_d), _m, _scaling.targets(_m))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What the network is trained on

    Gradient descent with unknown weights and biases
    $\mathbf{W}, \mathbf{b}$. The misfit is the mean
    squared error of the network's answer, in the scaled units,

    $$
    S = \mathrm{mean}\,\big\Vert \hat{\mathbf{m}} - \mathbf{m} \big\Vert_2^2
    $$

    and autodiff supplies $\nabla S$ for every weight. The optimiser (SGD or Adam) steps
    against it with step size $\eta$,

    $$
    \mathbf{W} \leftarrow \mathbf{W} - \eta\,\nabla_{\mathbf{W}} S
    $$

    We report the RMS location error in km, which is easier to read than $S$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Epochs and mini-batches

    A step after all events would be slow. To avoid this, events are shuffled and
    cut into **mini-batches**: one batch, one step. One pass through all events is
    an **epoch**, with a new shuffle each time.

    - Small batches: noisy steps that can shake the weights out of a poor local
      minimum.
    - Larger batches: less noisy gradient, still cheaper (memory and compute) than
      all at once
    - One batch: the exact gradient, straight into the nearest minimum, expensive
      per step
    """)
    return


@app.cell(hide_code=True)
def _(artifact):
    artifact("minibatch_epoch", width="100%")
    return


@app.cell
def _(
    Recorder,
    Scaling,
    SimpleNamespace,
    make_dataset,
    mo,
    nn,
    partial,
    torch,
    torch_threads,
):
    ACTIVATIONS = {
        "ReLU": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "linear": nn.Identity,
    }
    OPTIMISERS = {
        "SGD": torch.optim.SGD,
        "Adam": partial(torch.optim.Adam, fused=True),  # one kernel per step
    }

    @mo.cache  # only a new press of "train" trains again
    @torch_threads(1)  # tiny matrices: one thread is faster than eight
    def train(config):
        """Train a locator on the scene and settings captured by "train"."""
        # ---- 1. Data: simulate events and put them on a scale of about one ----
        scene, m_train, d_train, m_val, d_val = make_dataset(config)
        scaling = Scaling(d_train, m_train, config["normalise"])  # training set only
        x_train = scaling.inputs(d_train)  # network input: centred picks
        y_train = scaling.targets(m_train)  # network target: scaled locations

        # ---- 2. Model: N picks in, (m_x, m_y) out ----
        torch.manual_seed(0)  # the same initial weights and batches every time
        dims = [scene.n_stations, *config["widths"], 2]  # layer sizes
        layers = []
        for n_in, n_out in zip(dims[:-1], dims[1:], strict=True):
            layers += [nn.Linear(n_in, n_out), ACTIVATIONS[config["activation"]]()]
        net = nn.Sequential(*layers[:-1])  # drop the last activation: linear output

        # ---- 3. Optimiser: how the weights are updated from the gradient ----
        opt = OPTIMISERS[config["optimiser"]](net.parameters(), lr=config["lr"])

        # ---- 4. Error in km, for the curves (not what the network is trained on) ----
        def rms_km(d, m):
            """Root-mean-square location error (km) of the network on picks d."""
            with torch.no_grad():  # only looking, no gradient needed
                m_hat = scaling.locations(net(scaling.inputs(d)))  # back to km
            return ((m_hat - m) ** 2).sum(-1).mean().sqrt().item()

        n_train = len(m_train)
        batch_size = config["batch_size"] or n_train  # 0: the whole training set
        recorder = Recorder(net, config["epochs"])  # curves and snapshots to replay
        recorder.log(0, rms_km(d_train, m_train), rms_km(d_val, m_val))  # untrained

        # ---- 5. Training loop ----
        for epoch in range(1, config["epochs"] + 1):  # one epoch = one pass over data
            # shuffle the events, then cut them into mini-batches of indices
            for idx in torch.randperm(n_train).split(batch_size):
                # forward: the network's answer for this mini-batch
                y_hat = net(x_train[idx])

                # loss: mean squared error S against the scaled true locations
                S = ((y_hat - y_train[idx]) ** 2).sum(-1).mean()

                # backward: gradient of S with respect to every weight
                opt.zero_grad()  # forget the gradient of the previous step
                S.backward()

                # update: one optimiser step along the gradient
                opt.step()

            # after each epoch: measure progress on train and validation events
            recorder.log(epoch, rms_km(d_train, m_train), rms_km(d_val, m_val))
            if recorder.diverged:  # loss blew up: stop early
                break

        return SimpleNamespace(
            net=net,
            scaling=scaling,
            recorder=recorder,
            config=config,
            m_train=m_train,
        )

    return (train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The best possible locator

    - The data say how likely each location is: the **likelihood**.
    - Combined with the **prior**, it gives the **posterior**: the full
      probabilistic answer for one event.
    - Under a squared error the best single answer is the **posterior mean**, no
      network can beat this
    - The floor is not zero, because the picks are noisy.

    The figure uses noisier picks than usual, so that the prior can be seen.
    """)
    return


@app.cell(hide_code=True)
def _(default_scene, lp):
    lp.posterior_figure(default_scene, "ridgecrest", sigma=1.0)  # noisier: prior shows
    return


@app.cell
def _(mo, physics, posterior_rms, sample_prior, scene_from, torch):
    data_keys = "stations muted velocity field sigma train_prior".split()

    @mo.cache
    def error_floor(config):
        """RMS error (km) of the posterior mean on many fresh events: the best any
        locator trained with MSE can do on average. The validation events are too
        few (a handful of bad misses move the RMS by a km), so it uses its own."""
        scene = scene_from(config)
        torch.manual_seed(2)  # not the training or validation events
        m = sample_prior(config["train_prior"], 10_000)
        d = physics.observe(scene.F, m, scene.sigma, scene.live)
        return posterior_rms(
            scene.F, d, m, scene.sigma, config["train_prior"], scene.live
        )

    return data_keys, error_floor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading learning curves

    There are generally three kinds of behaviour you can see in learning curves:

    - **Underfit**: both errors stay high, well above
      the floor. The network is too simple.
    - **A good fit**: both errors fall and level off a little above the floor.
    - **Overfit**: the training error keeps falling
      while the validation error turns up.
    """)
    return


@app.cell(hide_code=True)
def _(data_keys, default_config, error_floor, lp, mo, train):
    _cases = [
        ("underfit: no hidden layer", default_config | {"widths": []}),
        ("a good fit: the defaults", default_config),
        (
            "overfit: 30 events, 3 × 128",
            default_config
            | {"n_train": 30, "widths": [128] * 3, "epochs": 300, "batch_size": 0},
        ),
    ]
    _runs = [
        (title, train(config).recorder)
        for title, config in mo.status.progress_bar(
            _cases, title="Training the three networks", remove_on_exit=True
        )
    ]
    lp.learning_curves_figure(
        [(title, r.rms_train, r.rms_val, r.best_epoch) for title, r in _runs],
        error_floor({k: default_config[k] for k in data_keys}),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A network knows only its simulator

    Training and validation error come from the same simulator, so neither can show that
    the simulator is wrong. The real world can differ in three ways:

    - **The prior**: events where the training prior put none.
    - **The physics**: a P-speed model other than the one that made the training picks.
    - **The stations**: one moved or had a failure, so the same input now means
      something else.
    """)
    return


@app.cell(hide_code=True)
def _(LocatorPlayground, mo):
    locator_playground = mo.ui.anywidget(LocatorPlayground())
    return (locator_playground,)


@app.cell(hide_code=True)
def _(data_keys, error_floor, locator_playground, train):
    config = locator_playground.train_config()  # what the last press captured
    run = train(config)  # cached: the same config never trains twice
    # the best possible error depends on the data only, not on the network
    floor = error_floor({k: config[k] for k in data_keys})
    return floor, run


@app.cell
def _(run, torch):
    def locate(d):
        """The trained network as a locator: centred picks (n, N) -> (n, 2) km."""
        with torch.no_grad():
            return run.scaling.locations(run.net(run.scaling.inputs(d)))

    return (locate,)


@app.cell(hide_code=True)
def _(locator_playground, mo):
    mo.vstack([locator_playground])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How the playground works

    Work from the top down.

    - **Scene.** Stations (+ / −; drag to move, double-click empty ground to add one,
      shift-click to mute a dead one), P-speed model and pick noise $\sigma$.
    - **Data.** Number of training events, training and test prior.
    - **Network.** Hidden layers, one width per layer, activation.
    - **Training.** Optimiser, step size $\eta$, batch size, epochs and *rescale*.

    The network **retrains as soon as you change a setting**, in well under a second,
    and the run plays back (▶ pauses, ⏭ steps, the bar scrubs). Set **RETRAIN** to *on
    button* to keep a trained network and change only the data, then press **TRAIN** for
    a new one. Runs that would take more than a few seconds always wait for the button.

    **Two maps** share the scene; each shows a background of your choice: the P speed
    (with the **rays** from the test event), the prior (with the training events), the
    **error** of the network for a source anywhere (*arrows* show where its answers are
    pulled), or the **posterior** or **likelihood** of the test event (68 % and 95 %
    regions, ◆ its mean). **Drag the test event** (star): the orange dot is the
    network's answer, the small dots are its answers for 64 noise draws of the same
    event. The network you chose is drawn under the playground: the learned weights, or
    the initial weights (grey) while a retraining is pending.

    With *on button*, moving or muting a station or switching the P speed changes the
    *data* but not the network. The posterior keeps the P speed and $\sigma$ of the
    training (dashed rays), while the data follow the new scene (solid).
    """)
    return


@app.cell(hide_code=True)
def _(try_it):
    # the page opens with 5 stations, 1000 events, 2 hidden layers of 16, tanh, Adam, 64
    try_it(
        [
            (
                "Capacity",
                "What is the smallest network that reaches the floor?",
                "Hidden layers 0, then one layer of width 4, then the defaults.",
                "Both errors stay above the dashed line while the network is too "
                "small. With a map on *error* and *arrows* on, answers are pulled "
                "towards the middle.",
            ),
            (
                "Data",
                "How many events stop a big network overfitting?",
                "Three layers of width 128, epochs 300, batch all. Training events "
                "30, then 300, then 1000.",
                "The training error keeps falling, the validation error turns up "
                "(▼ marks its minimum: the epoch to stop at). More events close the "
                "gap.",
            ),
            (
                "Data",
                "Do more stations help as much as more events?",
                "Stations 3, then 12.",
                "The dashed line and the posterior move: how much the data allow.",
            ),
            (
                "Preparation",
                "What does *rescale* buy?",
                "Switch *rescale* off.",
                "The same network barely learns in the same time.",
            ),
            (
                "Preparation",
                "How big can a step be?",
                "$\\eta$ = 0.3, then smaller. Batch *all* against 16.",
                "Big steps: the errors jump about and never settle. Batch *all*: one "
                "step per epoch, against many.",
            ),
            (
                "The floor",
                "What does the data add to the prior?",
                "One map on *posterior*, the other on *likelihood*; drag the star.",
                "The prior pulls the posterior towards where events were drawn; ◆ "
                "is its mean.",
            ),
            (
                "The floor",
                "What does the network answer when two places fit?",
                "Drag the five stations onto a horizontal line, put the star about "
                "20 km off it, one map on *posterior*, and drag the star along the "
                "line.",
                "Two lobes of equal weight under *uniform* (*ridgecrest* shifts mass "
                "but the second lobe stays). The network's one answer lies between "
                "them.",
            ),
            (
                "Shift",
                "Where on the map can you trust the network?",
                "Training prior *ridgecrest*, test prior *uniform*; one map on "
                "*error*, the other on *prior*.",
                "Good where the training events were, poor in the corners.",
            ),
            (
                "Shift",
                "What if the physics differ?",
                "RETRAIN *on button*, P speed *heterogeneous*, one map on *P speed*.",
                "The true rays bend, the dashed rays are what the posterior assumes. "
                "Network and posterior are wrong in the same way.",
            ),
            (
                "Shift",
                "What if a station moves or dies?",
                "RETRAIN *on button*, then drag a station or shift-click it.",
                "The network was trained for the old layout and has no way to know.",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(floor, locate, locator_playground, observe, run):
    # replay the run on the scene as it is now and send it to the playground above
    locator_playground.show(run, floor, locate, observe)
    return


if __name__ == "__main__":
    app.run()
