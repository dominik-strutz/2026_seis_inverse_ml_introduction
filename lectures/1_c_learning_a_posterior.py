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

__generated_with = "0.25.2"
app = marimo.App(
    width="full",
    layout_file="layouts/1_c_learning_a_posterior.slides.json",
    css_file="lecture_slides.css",
    html_head_file="slides_head.html",
)


@app.cell(hide_code=True)
def _():
    from functools import cache, partial
    from types import SimpleNamespace

    import marimo as mo
    import torch
    from torch import nn
    from torch.distributions import (
        AffineTransform,
        Categorical,
        MixtureSameFamily,
        MultivariateNormal,
        TransformedDistribution,
    )

    from ml_seis import physics, sample_prior
    from ml_seis.combinewidget import CombineFigure
    from ml_seis.learning_plotting import cost_figure
    from ml_seis.locatorwidget import LocatorPlayground, Recorder
    from ml_seis.physics import center, travel_times
    from ml_seis.posterior import (
        grid_posterior,
        posterior_rms,
        posterior_samples,
    )
    from ml_seis.runtime import torch_threads
    from ml_seis.scene import make_scene, observe, scene_from
    from ml_seis.slides import artifact, show_source, try_it
    from ml_seis.summarieswidget import SummariesFigure
    from ml_seis.velocity import SEED

    return (
        AffineTransform,
        Categorical,
        CombineFigure,
        LocatorPlayground,
        MixtureSameFamily,
        MultivariateNormal,
        Recorder,
        SEED,
        SimpleNamespace,
        SummariesFigure,
        TransformedDistribution,
        artifact,
        cache,
        center,
        cost_figure,
        grid_posterior,
        make_scene,
        mo,
        nn,
        observe,
        partial,
        physics,
        posterior_rms,
        posterior_samples,
        sample_prior,
        scene_from,
        show_source,
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
            # Part 1c: from a point to a distribution

            ### Machine learning and inversion for seismologists
            """),
            artifact("title_figure", width="65%"),
        ],
        gap=0,
    )
    return


@app.cell(hide_code=True)
def _(center, mo, physics, show_source, travel_times):
    mo.vstack(
        [
            mo.md("Physics from Parts 1a and 1b, unchanged:"),
            show_source(travel_times, center, physics.observe),
        ]
    )
    return


@app.cell(hide_code=True)
def _(nn, partial, torch):
    # From Part 1b: the activations and optimisers the playground offers.
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
    return ACTIVATIONS, OPTIMISERS


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

    return (make_dataset,)


@app.cell
def _(mo, physics, posterior_rms, sample_prior, scene_from, torch):
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

    return (error_floor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The loss picks the summary

    The posterior $p(\mathbf{m} \mid \tilde{\mathbf{d}})$ describes how likely it is
    for the source to be at $\mathbf{m}$ given our prior knowledge and what we
    measured. A network outputs one point $\hat{\mathbf{m}}$, so the **loss decides
    which point**: training minimises the expected loss $S(\hat{\mathbf{m}})$, with
    $\mathbb{E}$ the average over sources $\mathbf{m}$ drawn from the posterior.

    **MSE** (mean squared error): big misses cost far more than small ones

    $$
    S(\hat{\mathbf{m}}) = \mathbb{E}\left[\lVert \mathbf{m} - \hat{\mathbf{m}}
    \rVert^2_2 \right]
    \quad\Rightarrow\quad
    \hat{\mathbf{m}} = \mathbb{E}[\mathbf{m}]
    $$

    The **mean**. With two lobes it lands between them, where the source is not.

    **MAE** (mean absolute error): a miss costs in proportion to its size

    $$
    S(\hat{\mathbf{m}}) = \mathbb{E}\left[\lVert \mathbf{m} - \hat{\mathbf{m}}
    \rVert_1\right]
    \quad\Rightarrow\quad
    \hat{\mathbf{m}} = \operatorname{median}[\mathbf{m}]
    $$

    The **median**, of $m_x$ and $m_y$ separately: half of the posterior lies on
    either side. With two lobes it sits inside one.

    **MAP** (maximum a posteriori): no averaging. With prior density $\rho_M$ from
    before

    $$
    \hat{\mathbf{m}} = \arg\min_{\mathbf{m}}
    \left[\frac{S(\mathbf{m})}{\sigma^2} - \log \rho_M(\mathbf{m})\right]
    \quad\Rightarrow\quad
    \hat{\mathbf{m}} = \arg\max_{\mathbf{m}}\,
    p(\mathbf{m} \mid \tilde{\mathbf{d}})
    $$

    The highest **peak** of the posterior.

    If the posterior is one symmetric peak, all three agree. For multimodal
    distributions they do not, and **none of them is wrong, they answer different
    questions**. No single point can say "it is either here or there": that needs the
    whole density, which is what we learn next.
    """)
    return


@app.cell
def _(SEED, grid_posterior, make_scene, observe, torch):
    # one scene for the rest of the lecture: five stations on an east-west line
    line_stations = [[10.0 + 20.0 * i, 50.0] for i in range(5)]
    line = make_scene(line_stations, "homogeneous", SEED, sigma=0.3)
    m_event = torch.tensor([75.0, 65.0], dtype=torch.float64)  # the event (km)
    torch.manual_seed(3)
    d_event = observe(line, m_event[None])  # (1, N)
    post_event = grid_posterior(line.F, d_event[0], line.sigma, "ridgecrest")
    return line, line_stations, m_event, post_event


@app.cell(hide_code=True)
def _(SummariesFigure, line, m_event, mo, post_event):
    mo.vstack(
        [
            mo.ui.anywidget(
                SummariesFigure.from_posterior(
                    post_event, m_event, line.xi_x, line.xi_y
                )
            )
        ],
        justify="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A Gaussian mixture

    Describe the answer as a density: $K$ Gaussian bumps,

    $$
    q(\mathbf{m}) = \sum_{k=1}^{K} w_k\,
    \mathcal{N}(\mathbf{m};\, \boldsymbol{\mu}_k, \mathbf{L}_k \mathbf{L}_k^\top),
    \qquad \mathbf{w} = \operatorname{softmax}(\mathbf{a})
    $$

    Per component: one logit $a_k$, two numbers for the mean, three for the
    lower-triangular $\mathbf{L}_k$. **6 x K numbers** in total.

    The loss is the **negative log-likelihood** of the truth,

    $$
    S = -\frac{1}{n} \sum_{j=1}^{n} \log q(\mathbf{m}_j)
    $$

    which is small only when $q$ puts its mass on the true locations.
    """)
    return


@app.cell(hide_code=True)
def _(artifact):
    artifact("gmm")
    return


@app.cell
def _(
    AffineTransform,
    Categorical,
    MixtureSameFamily,
    MultivariateNormal,
    TransformedDistribution,
    nn,
    torch,
):
    # skip the argument checks of torch.distributions: the 6K numbers below are
    # valid by construction, and the checks cost about a tenth of the training time
    torch.distributions.Distribution.set_default_validate_args(False)

    def mixture(z, K):
        """A Gaussian mixture of K components from 6K numbers z (..., 6K):
        K logits, K means (2 numbers each), K Cholesky factors (3 each)."""
        logits, loc, chol = z.split([K, 2 * K, 3 * K], dim=-1)
        # chol: 3 numbers per component for the lower-triangular factor L of its
        # covariance, Sigma = L L^T: L = [[a, 0], [b, c]] with a, c > 0 (softplus
        # keeps them positive), so Sigma is always a valid covariance
        a, b, c = chol.unflatten(-1, (K, 3)).unbind(-1)
        diag_x = nn.functional.softplus(a) + 1e-3  # positive diagonal
        diag_y = nn.functional.softplus(c) + 1e-3
        L = torch.stack([diag_x, torch.zeros_like(b), b, diag_y], -1)
        return MixtureSameFamily(
            Categorical(logits=logits),
            MultivariateNormal(
                loc.unflatten(-1, (K, 2)), scale_tril=L.unflatten(-1, (2, 2))
            ),
        )

    def in_km(q, m_mean, m_scale):
        """A density over scaled units (about one) as a density over km."""
        to_km = AffineTransform(m_mean.float(), m_scale.float(), event_dim=1)
        return TransformedDistribution(q, to_km)

    return in_km, mixture


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fit a mixture to one posterior

    No network yet. The target is a multimodal posterior: the same five stations, but a
    heterogeneous velocity model, so the data constrain the source along a curved band
    with several clusters rather than two clean lobes (flat prior). Fit $q$ to
    samples of it by gradient descent on the NLL. Keep 2000 more samples apart: they
    play the role of the validation events, and the NLL on them is how well $q$
    carries over to samples it was not fitted to.
    """)
    return


@app.cell
def _(
    grid_posterior,
    line_stations,
    make_scene,
    observe,
    posterior_samples,
    torch,
):
    # the same array in a heterogeneous velocity model, a flat prior
    hetero = make_scene(line_stations, "heterogeneous", 7, sigma=0.8)
    m_hard = torch.tensor([60.0, 30.0], dtype=torch.float64)
    torch.manual_seed(3)
    d_hard = observe(hetero, m_hard[None])
    post_hard = grid_posterior(hetero.F, d_hard[0], hetero.sigma)
    # up to 1600 samples of it to fit to (the slider picks how many) and 2000
    # to score on, so the held-out NLL is a steady number rather than a noisy
    # one
    torch.manual_seed(1)
    _samples = posterior_samples(post_hard, 3600)
    m_fit, m_held = _samples[:1600], _samples[1600:]
    return m_fit, m_held, post_hard


@app.cell
def _(cache, in_km, m_fit, mixture, torch, torch_threads):
    @cache  # each (n, K) is fitted once
    @torch_threads(1)
    def fit_mixture(n, K, steps=400):
        """Fit a K-component mixture to the first n of the samples m_fit by
        minimising the NLL."""
        m = m_fit[:n]
        m_mean, m_scale = m.mean(0), m.std(0)
        x = ((m - m_mean) / m_scale).float()  # scaled to about one
        torch.manual_seed(0)  # independent of other cells
        start = x[torch.randperm(len(x))[:K]]  # means start on samples
        # Cholesky factors start at a round bump: softplus(-1) = 0.31 on the
        # diagonal
        chol = torch.tensor([-1.0, 0.0, -1.0]).repeat(K)
        z = torch.cat([torch.zeros(K), start.flatten(), chol]).requires_grad_()
        opt = torch.optim.Adam([z], lr=0.03)
        for _ in range(steps):
            opt.zero_grad()
            S = -mixture(z, K).log_prob(x).mean()  # NLL
            S.backward()
            opt.step()
        return in_km(mixture(z.detach(), K), m_mean, m_scale)

    return (fit_mixture,)


@app.cell(hide_code=True)
def _(fit_mixture, m_fit, m_held, mo, post_hard):
    from ml_seis.mixturewidget import MixtureFitFigure

    # every (n, K) once, in one cell nothing below depends on, so the figure
    # stays put; the slider stops at these numbers of fitted samples
    fits = {
        n: {K: fit_mixture(n, K) for K in (1, 2, 3, 4, 6, 8, 12)}
        for n in (25, 100, 200, 400, 1600)
    }
    mixture_figure = mo.ui.anywidget(
        MixtureFitFigure.from_fits(post_hard, m_fit, m_held, fits)
    )
    mo.vstack(
        [
            mixture_figure,
            mo.md(r"""
    ### Three things to look for

    - **One against several.** What does a single component do to a curved band
      with several clusters?
    - **Fitted against held out.** How do the NLL of the fitted and of the held-out
      samples change as $K$ grows, and where do they part?
    - **Few against many.** What do few fitted samples do that many do not?
        """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(try_it):
    try_it(
        [
            (
                "One against several",
                "Where does a single Gaussian put its mass?",
                "Slider $K$ = 1, then 2, 3, 4.",
                "$K$ = 1: one wide ellipse laid over the band, covering much empty "
                "ground. Each added component follows the curve better.",
            ),
            (
                "Fitted against held out",
                "Which $K$ would you choose, and why not the largest?",
                "Move $K$ from 1 to 12 and read the two NLL curves.",
                "Both curves fall until about $K$ = 4. After that the fitted NLL "
                "keeps creeping down while the held-out NLL stops improving and "
                "rises: the extra components learn the fitted samples, not the "
                "density.",
            ),
            (
                "Few against many",
                "What do few fitted samples do that many do not?",
                "Slider SAMPLES = 25 with $K$ = 8, then SAMPLES = 1600.",
                "25: the fitted NLL falls to 5.2 while the held-out NLL runs "
                "away to 29 — the 8 components learn the 25 samples, not the "
                "density. 1600: the two curves stay close for every $K$.",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Let the network output the mixture

    The last layer now gives $6K$ numbers instead of two, so every set of picks
    gets its own mixture $q(\mathbf{m} \mid \tilde{\mathbf{d}})$: a **mixture
    density network**. Two lines of the training loop change, the output size and
    the loss,

    $$
    S = -\frac{1}{n} \sum_{j=1}^{n} \log q(\mathbf{m}_j \mid \tilde{\mathbf{d}}_j)
    $$

    which is smallest when $q$ is the posterior.
    """)
    return


@app.cell
def _(
    ACTIVATIONS,
    OPTIMISERS,
    Recorder,
    Scaling,
    SimpleNamespace,
    make_dataset,
    mixture,
    mo,
    nn,
    torch,
    torch_threads,
):
    @mo.cache  # only a new press of "train" trains again
    @torch_threads(1)
    def train_density(setup):
        """Train a locator that outputs one point (MSE) or a mixture of K Gaussians
        (NLL), on the scene and settings captured by "train"."""
        scene, m_train, d_train, m_val, d_val = make_dataset(setup)
        scaling = Scaling(d_train, m_train, setup["normalise"])
        x_train, y_train = scaling.inputs(d_train), scaling.targets(m_train)
        point, K = setup["output"] == "point", setup["n_components"]

        # the MLP of the first half; the last layer gives 2 numbers or 6K
        torch.manual_seed(setup.get("seed", 0))  # the ensemble below varies it
        dims = [scene.n_stations, *setup["widths"], 2 if point else 6 * K]
        layers = []
        for n_in, n_out in zip(dims[:-1], dims[1:], strict=True):
            layers += [nn.Linear(n_in, n_out), ACTIVATIONS[setup["activation"]]()]
        net = nn.Sequential(*layers[:-1])
        opt = OPTIMISERS[setup["optimiser"]](net.parameters(), lr=setup["lr"])

        def density(x):  # the network's mixture, in scaled units
            return mixture(net(x), K)

        def loss(x, y):
            if point:
                return ((net(x) - y) ** 2).mean()  # MSE
            return -density(x).log_prob(y).mean()  # NLL

        def locate(d):
            """The network's answer (km): its point, or the mean of its mixture."""
            with torch.no_grad():
                x = scaling.inputs(d)
                return scaling.locations(net(x) if point else density(x).mean)

        def rms_km(d, m):
            return ((locate(d) - m) ** 2).sum(-1).mean().sqrt().item()

        n_train = len(m_train)
        batch_size = setup["batch_size"] or n_train
        recorder = Recorder(net, setup["epochs"])
        recorder.log(0, rms_km(d_train, m_train), rms_km(d_val, m_val))
        for epoch in range(1, setup["epochs"] + 1):
            for idx in torch.randperm(n_train).split(batch_size):
                opt.zero_grad()
                S = loss(x_train[idx], y_train[idx])
                S.backward()
                opt.step()
            recorder.log(epoch, rms_km(d_train, m_train), rms_km(d_val, m_val))
            if recorder.diverged:
                break
        return SimpleNamespace(
            net=net, scaling=scaling, recorder=recorder, config=setup,
            m_train=m_train, locate=locate, density=None if point else density,
        )  # fmt: skip

    return (train_density,)


@app.cell(hide_code=True)
def _(LocatorPlayground, line_stations, m_event, mo):
    locator_density = mo.ui.anywidget(
        LocatorPlayground(
            stations=line_stations,
            test_event=m_event.tolist(),
            train_prior="uniform",
            test_prior="uniform",
            n_train=1000,
            widths=[32, 32],
            lr=0.003,
            batch_size=64,
            epochs=100,
            output="mixture",
            n_components=1,
            show_output=True,
            auto_retrain=False,
            backgrounds=["network", "compare"],
        )
    )
    return (locator_density,)


@app.cell(hide_code=True)
def _(error_floor, locator_density, train_density):
    config_density = locator_density.train_config()
    run_density = train_density(config_density)  # cached per setting
    data_keys_density = "stations muted velocity field sigma train_prior"
    floor_density = error_floor(
        {k: config_density[k] for k in data_keys_density.split()}
    )
    return config_density, floor_density, run_density


@app.cell
def _(in_km, run_density):
    def posterior_of(d):
        """The network's posterior q(m | d~) in km for centred picks d (n, N)."""
        s = run_density.scaling
        return in_km(run_density.density(s.inputs(d)), s.m_mean, s.m_scale)

    return (posterior_of,)


@app.cell(hide_code=True)
def _(locator_density, mo):
    mo.vstack(
        [
            locator_density,
            mo.md(r"""
    ### Four things to look for

    Change one thing at a time and watch the maps, the curves and the errors.

    - **Output.** What does the network return when the data allow two answers?
    - **Components.** How many bumps does the answer need?
    - **Ambiguity.** What happens to the answer as the event moves towards the line of
      stations, and away from it?
    - **Shift.** Does the spread of the mixture tell you when the network is out of its
      depth?
        """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How the playground works

    The playground of 1b again, on the line of stations and the `uniform` prior, with
    one more control: **OUTPUT** *point* (MSE) or *mixture* with $K$ components. The
    page opens with 1000 events, two layers of 32 neurons, tanh, Adam, $\eta$ = 0.003,
    batches of 64, 100 epochs and a mixture of $K$ = 1. MAP A shows the
    network's mixture for the picks of the test event (68 % and 95 % regions, each
    component with its weight). MAP B compares it with the truth: the exact
    posterior is the green shading, the network's 68 % (solid) and 95 % (dashed)
    regions are the orange lines on top. Each setting trains once and is kept.
    """)
    return


@app.cell(hide_code=True)
def _(locator_density, run_density):
    locator_density.network_figure(run_density.net)
    return


@app.cell(hide_code=True)
def _(floor_density, locator_density, observe, posterior_of, run_density):
    locator_density.show(
        run_density, floor_density, run_density.locate, observe, posterior_of
    )
    return


@app.cell(hide_code=True)
def _(try_it):
    try_it(
        [
            (
                "Output",
                "What does each output return for the picks of the test event?",
                "OUTPUT *point*, then *mixture* with $K$ = 4.",
                "*point*: one dot on the line of stations, between the two lobes of "
                "the posterior. *mixture*: a bump in each lobe; compare the orange "
                "regions on MAP B with the green posterior. The error is the same "
                "for both (29 km, the dashed floor): it scores only the mean.",
            ),
            (
                "Components",
                "How many bumps does the answer need?",
                "$K$ = 1, 2, then 4.",
                "$K$ = 1 and 2: one ellipse on the line between the lobes, where the "
                "source is not. From $K$ = 4 a bump in each lobe.",
            ),
            (
                "Ambiguity",
                "What happens to the two lobes as the event moves?",
                "Drag the test event (star) towards the line of stations, then away.",
                "5 km from the line the lobes of the exact posterior are 8 km apart "
                "and look like one, 30 km from it 55 km apart. It always weighs "
                "them equally: a line of stations cannot tell north from south.",
            ),
            (
                "Shift",
                "Does the mixture know when the network is out of its depth?",
                "$K$ = 4, train prior *ridgecrest*, press train; then test prior "
                "*uniform*, maps on *compare*, drag the star into each corner.",
                "In the corners no training event fell, yet the orange regions stay "
                "as small as before and miss the star at all four.",
            ),
            (
                "Shift",
                "And when the data change in another way?",
                "$K$ = 4, train as it is; then mute one station at a time.",
                "Unless it is the middle one, the orange regions miss the star "
                "although the green posterior still holds it: the mixture's spread "
                "is the data's uncertainty, not the network's.",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A density can be combined

    A point is an answer. A density can take a **new prior**: what else we know about
    the source (a mapped fault, a felt report, a second array) as a density
    $\rho_M(\mathbf{m})$. The network was trained on a flat prior, so its answer
    $q$ is the **likelihood** of the picks, up to a constant. Bayes' rule puts the two
    together

    $$
    p(\mathbf{m} \mid \tilde{\mathbf{d}}) \;\propto\;
    q(\mathbf{m} \mid \tilde{\mathbf{d}})\; \rho_M(\mathbf{m})
    $$

    The new information is added after training, as in variational prior replacement
    (Zhao and Curtis, 2024): no new network, no retraining, one multiplication and a
    normalisation on a grid.

    A point cannot do this. The mean between two lobes stays between them, however
    clearly a fault says which lobe is right.
    """)
    return


@app.cell(hide_code=True)
def _(
    CombineFigure,
    config_density,
    locator_density,
    mo,
    observe,
    posterior_of,
    run_density,
    torch,
):
    mo.stop(
        run_density.density is None,
        mo.md("*Set OUTPUT to mixture in the playground and press train.*"),
    )
    _now = locator_density.current_scene(config_density)[0]
    mo.stop(
        _now.n_stations != len(config_density["stations"]),
        mo.md("*The stations changed: press train in the playground.*"),
    )
    # the playground's own test event and noise draw
    _m = torch.tensor(locator_density.test_event, dtype=torch.float64)
    torch.manual_seed(2)
    _d = observe(_now, _m[None])
    mo.vstack(
        [
            mo.ui.anywidget(
                CombineFigure.from_answers(
                    posterior_of,
                    _d,
                    _m,
                    _now.xi_x[_now.live],
                    _now.xi_y[_now.live],
                    config_density["train_prior"],
                )
            )
        ],
        justify="center",
    ).style(height="80vh")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why a network, when the grid is exact?

    The exact posterior of the playground is cheap for two unknowns: it is what we
    checked the network against. The price comes with more unknowns. A grid
    with $n$ nodes per axis has $n^d$ nodes for $d$ unknowns, while a mixture of $K$
    components has $K\,(1 + d + d(d+1)/2)$ numbers, $6K$ for $d = 2$.

    And the network never needs the likelihood, only a simulator: wherever we can
    simulate pairs $(\mathbf{m}, \tilde{\mathbf{d}})$ we can train one. This is
    **neural posterior estimation**.
    """)
    return


@app.cell(hide_code=True)
def _(cost_figure):
    cost_figure()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Further reading

    Simulate pairs, train a network to output a density, apply it to data:
    **neural posterior estimation** (simulation-based inference). Mixture density
    networks in geophysics: Meier, Curtis and Trampert (2007), crustal thickness;
    Käufl et al. (2014), earthquake sources. Background: Papamakarios and Murray
    (2016); Cranmer, Brehmer and Louppe (2020). Changing what a trained posterior
    assumes, without retraining: Zhao and Curtis (2024).
    """)
    return


if __name__ == "__main__":
    app.run()
