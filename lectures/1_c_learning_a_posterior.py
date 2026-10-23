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
# ml-seis = { path = "..", editable = true }
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
    layout_file="layouts/1_c_learning_a_posterior.slides.json",
    css_file="lecture_slides.css",
    html_head_file="slides_head.html",
)


@app.cell
def _():
    import time
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
    from ml_seis.density_plotting import (
        calibration_figure,
        density_fit_figure,
        ensemble_figure,
        mixture_components,
        summaries_figure,
    )
    from ml_seis.locatorwidget import LocatorPlayground, Recorder
    from ml_seis.physics import center, travel_times
    from ml_seis.posterior import (
        grid_posterior,
        grid_sources,
        posterior_from_log,
        posterior_means,
        posterior_rms,
        posterior_samples,
        posterior_scores,
    )
    from ml_seis.runtime import torch_threads
    from ml_seis.scene import make_scene, observe, scene_from
    from ml_seis.slides import artifact, show_source, try_it
    from ml_seis.velocity import SEED

    return (
        AffineTransform,
        artifact,
        cache,
        calibration_figure,
        Categorical,
        center,
        density_fit_figure,
        ensemble_figure,
        grid_posterior,
        grid_sources,
        LocatorPlayground,
        make_scene,
        mixture_components,
        MixtureSameFamily,
        mo,
        MultivariateNormal,
        nn,
        observe,
        partial,
        physics,
        posterior_from_log,
        posterior_means,
        posterior_rms,
        posterior_samples,
        posterior_scores,
        Recorder,
        sample_prior,
        scene_from,
        SEED,
        show_source,
        SimpleNamespace,
        summaries_figure,
        time,
        torch,
        torch_threads,
        TransformedDistribution,
        travel_times,
        try_it,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part 1c: from a point to a distribution

    ### Machine learning and inversion for seismologists

    In 1b the network returned one point, the posterior mean. When the data allow two
    answers, that point can be one that no event would produce. By the end you can

    - say what a loss function asks a network to return,
    - describe an answer as a Gaussian mixture and let a network output one,
    - check a learned distribution against the exact posterior with coverage and the
      NLL,
    - name where it fails.
    """)
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


@app.cell
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
        gen_train = torch.Generator().manual_seed(0)  # the same events every time
        # validation has its own stream, so it does not change with n_train
        gen_val = torch.Generator().manual_seed(1)
        m_train = sample_prior(config["train_prior"], config["n_train"], gen_train)
        m_val = sample_prior(config["train_prior"], N_VAL, gen_val)
        d_train = physics.observe(scene.F, m_train, scene.sigma, scene.live, gen_train)
        d_val = physics.observe(scene.F, m_val, scene.sigma, scene.live, gen_val)
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
        gen = torch.Generator().manual_seed(2)  # not the training or validation events
        m = sample_prior(config["train_prior"], 10_000, gen)
        d = physics.observe(scene.F, m, scene.sigma, scene.live, gen)
        return posterior_rms(
            scene.F, d, m, scene.sigma, config["train_prior"], scene.live
        )

    return (error_floor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What the network learned

    Each training pair is a source from the prior and its noisy picks, so the sources
    behind one $\tilde{\mathbf{d}}$ follow the posterior

    $$
    p(\mathbf{m} \mid \tilde{\mathbf{d}}) \propto
    p(\tilde{\mathbf{d}} \mid \mathbf{m})\,\rho_M(\mathbf{m})
    $$

    The squared error picks the point closest to them on average, the posterior mean
    $\hat{\mathbf{m}}_\text{MSE}(\tilde{\mathbf{d}}) = \mathbb{E}[\mathbf{m} \mid
    \tilde{\mathbf{d}}]$. Its average error is the dashed floor of the 1b playground.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The loss picks the summary

    Each loss asks for a different single point $\hat{\mathbf{m}}$ from the same
    posterior $p(\mathbf{m} \mid \tilde{\mathbf{d}})$:

    - **Mean** (squared error): minimises the expected squared distance to the
      sources behind $\tilde{\mathbf{d}}$

      $$
      \hat{\mathbf{m}}_\text{MSE} = \arg\min_{\hat{\mathbf{m}}}
      \mathbb{E}\left[\lVert \mathbf{m} - \hat{\mathbf{m}} \rVert^2 \mid
      \tilde{\mathbf{d}}\right] = \mathbb{E}[\mathbf{m} \mid \tilde{\mathbf{d}}]
      $$

    - **Median** (absolute error): minimises the expected absolute error in each
      coordinate, so it is the middle of the marginal posterior of $m_x$ and of $m_y$

      $$
      \hat{m}_{\alpha,\text{MAE}} = \arg\min_{\hat{m}_\alpha}
      \mathbb{E}\left[\lvert m_\alpha - \hat{m}_\alpha \rvert \mid
      \tilde{\mathbf{d}}\right]
      \quad\Rightarrow\quad
      \int_{-\infty}^{\hat{m}_\alpha} p(m_\alpha \mid \tilde{\mathbf{d}})\,
      \mathrm{d}m_\alpha = \tfrac{1}{2}
      $$

    - **Most likely point, MAP** (misfit plus prior): the peak of the posterior, found
      by minimising the misfit $S$ plus the negative log prior

      $$
      \hat{\mathbf{m}}_\text{MAP} = \arg\max_{\mathbf{m}}
      p(\mathbf{m} \mid \tilde{\mathbf{d}}) = \arg\min_{\mathbf{m}}
      \left[\frac{S(\mathbf{m})}{\sigma^2} - \log \rho_M(\mathbf{m})\right]
      $$

    The three agree when the posterior is a single symmetric blob. With the stations
    on a line, the posterior has two lobes: the mean lands between them, and the
    median and the MAP each pick one lobe. No single point describes both.

    From here on: five stations on an east–west line, $\sigma$ = 0.3 s, the
    `ridgecrest` prior and one event north of the line.
    """)
    return


@app.cell
def _(SEED, grid_posterior, make_scene, observe, torch):
    # one scene for the rest of the lecture: five stations on an east-west line
    line_stations = [[10.0 + 20.0 * i, 50.0] for i in range(5)]
    line = make_scene(line_stations, "homogeneous", SEED, sigma=0.3)
    m_event = torch.tensor([75.0, 65.0], dtype=torch.float64)  # the event (km)
    d_event = observe(line, m_event[None], torch.Generator().manual_seed(3))  # (1, N)
    post_event = grid_posterior(line.F, d_event[0], line.sigma, "ridgecrest")
    return line, line_stations, m_event, post_event


@app.cell
def _(line, m_event, post_event, summaries_figure):
    summaries_figure(post_event, m_event, line.xi_x, line.xi_y)
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
    lower-triangular $\mathbf{L}_k$. **6K numbers** in total. The loss is the
    **negative log-likelihood** of the truth, $S = -\frac{1}{n} \sum_j \log
    q(\mathbf{m}_j)$, which is small only where $q$ puts its mass.
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

    No network yet: fit $q$ to 400 samples of the posterior above by gradient
    descent on the NLL, and keep 400 samples apart. The held-out samples play the role
    of the validation events: the NLL on them is how well $q$ carries over to samples
    it was not fitted to.
    """)
    return


@app.cell
def _(post_event, posterior_samples, torch):
    # Fit a mixture to one posterior, no network yet: fit q to 400 samples of the
    # posterior by gradient descent on the NLL, and keep 400 samples apart.
    gen_samples = torch.Generator().manual_seed(1)
    m_fit, m_held = posterior_samples(post_event, 800, gen_samples).split(400)
    return m_fit, m_held


@app.cell
def _(cache, in_km, m_fit, mixture, torch, torch_threads):
    @cache  # each K is fitted once
    @torch_threads(1)
    def fit_mixture(K, steps=300):
        """Fit a K-component mixture to the samples m_fit by minimising the NLL."""
        torch.manual_seed(0)
        m_mean, m_scale = m_fit.mean(0), m_fit.std(0)
        x = ((m_fit - m_mean) / m_scale).float()  # scaled to about one
        start = x[torch.randperm(len(x))[:K]]  # the means start on K samples
        chol = torch.tensor([-1.0, 0.0, -1.0]).repeat(K)
        z = torch.cat([torch.zeros(K), start.flatten(), chol]).requires_grad_()
        opt = torch.optim.Adam([z], lr=0.03)
        for _ in range(steps):
            opt.zero_grad()
            S = -mixture(z, K).log_prob(x).mean()  # NLL
            S.backward()
            opt.step()
        return in_km(mixture(z.detach(), K), m_mean, m_scale)

    def nll(q, m):
        """Mean negative log-likelihood of locations m (km) under q."""
        with torch.no_grad():
            return -q.log_prob(m.float()).mean().item()

    return fit_mixture, nll


@app.cell
def _(mo):
    k_slider = mo.ui.slider(1, 8, value=2, label="components K", show_value=True)
    return (k_slider,)


@app.cell
def _(
    density_fit_figure,
    fit_mixture,
    grid_sources,
    k_slider,
    m_fit,
    m_held,
    mixture_components,
    mo,
    nll,
    post_event,
    torch,
):
    curves = {"fit": {}, "held-out": {}}  # every K once (cached), about 3 s
    for K in range(1, 9):
        curves["fit"][K] = nll(fit_mixture(K), m_fit)
        curves["held-out"][K] = nll(fit_mixture(K), m_held)
    q_shown = fit_mixture(k_slider.value)
    with torch.no_grad():
        log_q_grid = q_shown.log_prob(grid_sources().float())
    mo.vstack(
        [
            mo.vstack(
                [
                    k_slider,
                    density_fit_figure(
                        post_event,
                        m_fit,
                        log_q_grid,
                        mixture_components(q_shown),
                        curves,
                        k_slider.value,
                    ),
                ]
            ),
            mo.md(r"""
    ### Two things to look for

    - **One against two.** What does a single component do to a posterior with two
      lobes?
    - **The held-out curve.** How do the fit and the held-out NLL change as $K$ grows,
      and where do they part?
        """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(try_it):
    try_it(
        [
            (
                "One against two",
                "Where does a single Gaussian put its mass?",
                "Slider $K$ = 1, then 2.",
                "$K$ = 1: one wide bump between the lobes, where the posterior is "
                "small. $K$ = 2 finds both, and the NLL drops by about one.",
            ),
            (
                "The held-out curve",
                "Which $K$ would you choose, and why not the largest?",
                "Move the slider from 1 to 8 and read the NLL curves.",
                "Both curves fall from $K$ = 1 to 2. After that the fit NLL creeps "
                "down while the held-out NLL stays flat and then rises: more bumps "
                "chase the 400 samples.",
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

    which is smallest when $q$ is the posterior. For the error curves the
    network's answer is the mean of $q$.
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


@app.cell
def _(LocatorPlayground, line_stations, m_event, mo):
    locator_density = mo.ui.anywidget(
        LocatorPlayground(
            stations=line_stations,
            test_event=m_event.tolist(),
            train_prior="ridgecrest",
            test_prior="ridgecrest",
            n_train=3000,
            widths=[32, 32],
            lr=0.003,
            batch_size=64,
            epochs=100,
            output="mixture",
            n_components=2,
            show_output=True,
            auto_retrain=False,
            backgrounds=["network", "compare"],
        )
    )
    return (locator_density,)


@app.cell
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
    ### Five things to look for

    Change one thing at a time and watch the maps, the curves and the errors.

    - **Output.** What does the network return when the data allow two answers?
    - **Components.** How many bumps does the answer need?
    - **Ambiguity.** What happens to the answer as the event moves towards the line of
      stations, and away from it?
    - **Training.** When does the mixture turn too sure of itself?
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

    The playground of 1b again, on the line of stations and the `ridgecrest` prior, with
    one more control: **OUTPUT** *point* (MSE) or *mixture* with $K$ components. The
    page opens with 3000 events, two layers of 32 neurons, tanh, Adam, $\eta$ = 0.003,
    batches of 64, 100 epochs and a mixture of $K$ = 2. MAP A shows the
    network's mixture for the picks of the test event (68 % and 95 % regions, each
    component with its weight). MAP B compares it with the truth: the exact
    posterior is the green shading, the network's 68 % (solid) and 95 % (dashed)
    regions are the orange lines on top. Each setting trains once and is kept;
    the button below trains $K$ = 1 to 8 in one go.
    """)
    return


@app.cell
def _(locator_density, run_density):
    locator_density.network_figure(run_density.net)
    return


@app.cell
def _(floor_density, locator_density, observe, posterior_of, run_density):
    locator_density.show(
        run_density, floor_density, run_density.locate, observe, posterior_of
    )
    return


@app.cell
def _(mo):
    train_all = mo.ui.run_button(label="train K = 1 to 8 for this scene")
    train_all  # noqa: B018 (display it)
    return (train_all,)


@app.cell
def _(config_density, mo, train_all, train_density):
    mo.stop(not train_all.value)
    variants = [{"output": "point"}]
    variants += [{"output": "mixture", "n_components": k} for k in range(1, 9)]
    for variant in mo.status.progress_bar(
        variants, title="Training K = 1 to 8", remove_on_exit=True
    ):
        train_density(config_density | variant)
    mo.md("Trained: switching OUTPUT and $K$ is now instant.")
    return


@app.cell(hide_code=True)
def _(try_it):
    try_it(
        [
            (
                "Output",
                "What does each output return for the picks of the test event?",
                "OUTPUT *point*, then *mixture*.",
                "*point*: one dot between the lobes. *mixture*: two lobes with their "
                "weights; compare the orange regions on MAP B with the green "
                "posterior. The error is about the same for both, the dashed floor: "
                "it scores only the mean.",
            ),
            (
                "Components",
                "How many bumps does the answer need, and how many get a weight?",
                "$K$ = 1, 2, then 8 (the button below trains $K$ = 1 to 8 at once).",
                "$K$ = 1: one wide ellipse with its mass between the lobes, and a "
                "clearly worse NLL. $K$ = 8: only two components get a real weight; "
                "the others are close to zero.",
            ),
            (
                "Ambiguity",
                "What happens to the two lobes as the event moves?",
                "Drag the test event (star) towards the line of stations, then away.",
                "Near the line the two lobes sit close together (about 15 km apart); "
                "far from it they separate (about 60 km). The weights shift with the "
                "noise in the picks.",
            ),
            (
                "Training",
                "What goes wrong with a ten times larger step?",
                "$\\eta$ = 0.03 instead of 0.003.",
                "The errors jump about and never settle (around 50 km), and the NLL "
                "below is much worse.",
            ),
            (
                "Training",
                "How few events are too few?",
                "Training events 300, then 100, then 30.",
                "At 300 almost nothing changes. At 100 and 30 the mixture is too sure: "
                "17 % and 38 % of the true sources fall outside their 95 % region, "
                "where 5 % should (the coverage curve below).",
            ),
            (
                "Shift",
                "Does the mixture know when the network is out of its depth?",
                "RETRAIN *on button*; test prior *uniform*, maps on *compare*.",
                "In the corners no training event fell, yet the regions stay small "
                "and miss: about a third of the true sources lie outside their 95 % "
                "region.",
            ),
            (
                "Shift",
                "And when the data change in another way?",
                "RETRAIN *on button*; one at a time: mute a station, pick noise "
                "$\\sigma$ = 0.5, P speed *heterogeneous*.",
                "The spread stays that of the training data. With a muted station "
                "most true sources (70 to 90 %) fall outside their 95 % region; with "
                "more noise about a quarter.",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Is it honest? Coverage

    For each validation event, find the smallest high-density region of $q$ that
    still holds the true source; its probability is the event's **credible
    level**. If $q$ is honest, 68 % of the sources lie in their 68 % regions, and
    so on: the coverage curve follows the diagonal. The exact posterior's NLL is
    the floor. Coverage checks honesty; the NLL also rewards sharpness.
    """)
    return


@app.cell
def _(in_km, make_dataset, posterior_scores, torch, train_density):
    def network_scores(run, d, m, n_samples=500):
        """NLL and credible level of the true sources m (km) under the network's
        mixture: the level is the share of samples more probable than the truth."""
        s = run.scaling
        with torch.no_grad(), torch.random.fork_rng():
            torch.manual_seed(0)
            q = in_km(run.density(s.inputs(d)), s.m_mean, s.m_scale)
            log_q_true = q.log_prob(m.float())  # (E,)
            log_q_samples = q.log_prob(q.sample((n_samples,)))  # (samples, E)
        return -log_q_true, (log_q_samples > log_q_true).float().mean(0)

    scored = {}  # mo.cache cannot hash the helper functions above in `marimo run`

    def calibration(setup):
        """Credible levels and mean NLL on the validation events: exact posterior,
        the playground's mixture, and K = 1 for comparison."""
        key = repr(sorted(setup.items()))
        if key not in scored:
            scored[key] = score(setup)
        return scored[key]

    def score(setup):
        scene, _, _, m_val, d_val = make_dataset(setup)
        nll_exact, level_exact = posterior_scores(
            scene.F, d_val, m_val, scene.sigma, setup["train_prior"], scene.live
        )
        levels = {"exact posterior": level_exact}
        nlls = {"exact posterior": nll_exact.mean().item()}
        runs = {"mixture, K = 1": setup | {"output": "mixture", "n_components": 1}}
        if setup["output"] == "mixture" and setup["n_components"] > 1:
            runs = {f"mixture, K = {setup['n_components']}": setup} | runs
        for name, s in runs.items():
            nll_net, level_net = network_scores(train_density(s), d_val, m_val)
            levels[name], nlls[name] = level_net, nll_net.mean().item()
        return levels, nlls

    return (calibration,)


@app.cell
def _(calibration, calibration_figure, config_density):
    calibration_figure(*calibration(config_density))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where it breaks

    When the data differ from the training data (another prior, a muted station, more
    noise, a different P speed) the mixture stays as confident as before, and wrong.

    Its spread is the uncertainty in the data, not the network's own. Five seeds
    agree where training events were and disagree where there were none.
    """)
    return


@app.cell
def _(mo):
    ensemble_switch = mo.ui.switch(label="train five seeds (≈ 10 s)")
    return (ensemble_switch,)


@app.cell
def _(
    config_density,
    ensemble_switch,
    grid_posterior,
    grid_sources,
    in_km,
    mo,
    observe,
    posterior_from_log,
    scene_from,
    torch,
    train_density,
):
    mo.stop(not ensemble_switch.value, ensemble_switch)
    mo.stop(
        config_density["output"] == "point",
        mo.vstack([ensemble_switch, mo.md("*Set OUTPUT to mixture.*")]),
    )
    members = [
        train_density(config_density | {"seed": s})
        for s in mo.status.progress_bar(
            range(5), title="Training the ensemble", remove_on_exit=True
        )
    ]
    scene_trained = scene_from(config_density)
    grid = grid_sources()
    ensemble_cases = []
    for title, source in [
        ("inside the training events", [75.0, 65.0]),
        ("a corner with almost no training events", [90.0, 10.0]),
    ]:
        m_case = torch.tensor([source], dtype=torch.float64)
        d_case = observe(scene_trained, m_case, torch.Generator().manual_seed(4))
        exact = grid_posterior(
            scene_trained.F, d_case[0], scene_trained.sigma,
            config_density["train_prior"], scene_trained.live,
        )  # fmt: skip
        learned = []
        for member in members:
            s = member.scaling
            with torch.no_grad():
                q = in_km(member.density(s.inputs(d_case.expand(len(grid), -1))),
                          s.m_mean, s.m_scale)  # fmt: skip
                learned.append(posterior_from_log(q.log_prob(grid.float())))
        ensemble_cases.append((title, m_case[0], exact, learned))
    return ensemble_cases, members, scene_trained


@app.cell
def _(
    ensemble_cases,
    ensemble_figure,
    ensemble_switch,
    members,
    mo,
    scene_trained,
):
    mo.vstack([
        ensemble_switch,
        ensemble_figure(
            ensemble_cases, scene_trained.xi_x, scene_trained.xi_y,
            members[0].m_train[::3],
        ),
    ])  # fmt: skip
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Use the distribution

    A distribution answers questions a point cannot: *is the playground's test
    event north of the line?* And it is fast: the grid posterior evaluates every
    node for every event, the network evaluates once (**amortised** inference).
    """)
    return


@app.cell
def _(
    config_density,
    grid_posterior,
    locator_density,
    make_dataset,
    observe,
    posterior_means,
    posterior_of,
    run_density,
    time,
    torch,
):
    scene_now, scene_assumed = locator_density.current_scene(config_density)
    m_test = torch.tensor([locator_density.test_event], dtype=torch.float64)
    d_test = observe(scene_now, m_test, torch.Generator().manual_seed(3))
    post_test = grid_posterior(
        scene_assumed.F, d_test[0], scene_assumed.sigma,
        config_density["train_prior"], scene_assumed.live,
    )  # fmt: skip
    p_cells = post_test.density / post_test.density.sum()
    # rows above y = 50 km (1 km grid), plus half of the row on the line
    p_north_exact = (p_cells[51:].sum() + 0.5 * p_cells[50].sum()).item()
    p_north_net = None
    if run_density.density is not None:
        with torch.no_grad(), torch.random.fork_rng():
            torch.manual_seed(0)
            samples = posterior_of(d_test).sample((2000,))[:, 0]
        p_north_net = (samples[:, 1] > 50).double().mean().item()

    # time per event: the exact posterior mean against one pass of the network
    _, _, _, m_val, d_val = make_dataset(config_density)
    tic = time.perf_counter()
    posterior_means(
        scene_assumed.F, d_val, scene_assumed.sigma, config_density["train_prior"]
    )
    t_grid = (time.perf_counter() - tic) / len(d_val)
    tic = time.perf_counter()
    run_density.locate(d_val)
    t_net = (time.perf_counter() - tic) / len(d_val)
    return p_north_exact, p_north_net, t_grid, t_net


@app.cell
def _(mo, p_north_exact, p_north_net, t_grid, t_net):
    _net = "— (one point)" if p_north_net is None else f"{p_north_net:.2f}"
    mo.md(f"""
    | | P(north of the line) | time per event |
    |---|---|---|
    | exact posterior (grid) | {p_north_exact:.2f} | {1e3 * t_grid:.2f} ms |
    | network | {_net} | {1e6 * t_net:.1f} µs |

    The network's weights for the two lobes are only approximately right: the exact
    posterior is the reference, and the gap to it is the network's own error.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A name for what we built

    Simulate pairs, train a network to output a density, apply it to data:
    **neural posterior estimation** (simulation-based inference). Mixture density
    networks in geophysics: Meier, Curtis and Trampert (2007), crustal thickness;
    Käufl et al. (2014), earthquake sources. Background: Papamakarios and Murray
    (2016); Cranmer, Brehmer and Louppe (2020).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What we keep

    - Trained with MSE, a network returns the posterior mean, whatever the
      posterior looks like.
    - The output layer sets what can be answered; the loss sets what is learned.
      A mixture with the NLL can return several answers with their weights.
    - Choose $K$ with held-out data: the NLL stops improving after the lobes are
      found, while more bumps only chase the samples. Check the result with coverage
      and the NLL floor.
    - The mixture's spread is the data's uncertainty, not the network's: outside
      its training data it can be confidently wrong.
    """)
    return


if __name__ == "__main__":
    app.run()
