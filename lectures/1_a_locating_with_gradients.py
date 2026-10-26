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
    layout_file="layouts/1_a_locating_with_gradients.slides.json",
    css_file="lecture_slides.css",
    html_head_file="slides_head.html",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import torch

    from ml_seis import playground
    from ml_seis.physics import center, simulate_picks, travel_times
    from ml_seis.slides import artifact, show_source, try_it

    return (
        artifact,
        center,
        mo,
        playground,
        show_source,
        simulate_picks,
        torch,
        travel_times,
        try_it,
    )


@app.cell(hide_code=True)
def _(artifact, mo):
    mo.vstack(
        [
            mo.md(r"""
            # Backpropagation, Optimisation and Neural Networks

            ### Machine learning and inversion for seismologists

            #### Part 1a: locating an earthquake with gradients
            """),
            artifact("title_figure", width="65%"),
        ],
        gap=0,
    )
    return


@app.cell(hide_code=True)
def _(artifact, mo):
    mo.vstack(
        [
            mo.md(r"""
            ## 2D Seismic Source Location

            - **Given:** the station positions and the arrival time of the first
              P wave at each station.
            - **Unknown:** the epicentre $\mathbf{m} = (m_x, m_y)$ and the origin
              time $t_0$ of the earthquake.
            """),
            artifact("schematic_location", width="45%"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Arrival Time Picking

    For this example we don't use the full waveform of the seismograms to locate the
    events, only the
    arrival times of the P-wave. To get these arrival times, we need to pick them
    from the seismograms. This introduces a picking error, modelled as
    $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$, where $\sigma$ is the picking error
    standard deviation.
    """)
    return


@app.cell(hide_code=True)
def _(artifact):
    artifact("picking", width="100%")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("## Forward model: from source position to arrival time"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md(r"* source location $\mathbf{m} = (m_x, m_y)$ in km"),
                            mo.md(r"* stations $\boldsymbol{\xi}_i$"),
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md(r"* constant wave speed $v$"),
                            mo.md(r"* picking error $\varepsilon_i$"),
                        ]
                    ),
                ],
                justify="start",
                gap=4,
                align="start",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(artifact):
    artifact("forward_model", width="100%")
    return


@app.cell(hide_code=True)
def _(show_source, travel_times):
    show_source(travel_times)
    return


@app.cell
def _(torch):
    # Toy geometry: local frame in km
    xi_x = torch.tensor([10.0, 90.0, 40.0], dtype=torch.float64)  # station x (km)
    xi_y = torch.tensor([20.0, 30.0, 85.0], dtype=torch.float64)  # station y (km)
    v_ref = 6.0  # P speed (km/s)
    sigma = 0.1  # pick noise (s)
    return sigma, v_ref, xi_x, xi_y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Removing $t_0$

    In reality a station records the arrival time, which also contains the
    unknown origin time $t_0$ of the earthquake. In this example we are not
    interested in $t_0$. Station $i$ records

    $$
    d_i = t_0 + F_i(\mathbf{m}) + \varepsilon_i
    $$

    Following Tarantola and Valette (1982), subtract the mean over the $N$
    stations from each arrival time. Moving $t_0$ shifts all arrival times and
    their mean by the same amount, so the constant $t_0$ cancels,

    $$
    \tilde{d}_i = d_i - \langle d \rangle
    \qquad
    \langle d \rangle = \frac{1}{N}\sum_{j=1}^{N} d_j
    $$

    We do the same to the simulated data

    $$
    \tilde{F}_i(\mathbf{m})
    = F_i(\mathbf{m}) - \langle F \rangle(\mathbf{m})
    \qquad
    \langle F \rangle = \frac{1}{N}\sum_{j=1}^{N} F_j(\mathbf{m})
    $$

    This reduces the amount of information we have:
    with $N$ stations only $N-1$ centred times are independent.
    """)
    return


@app.cell(hide_code=True)
def _(artifact):
    artifact("remove_t0")
    return


@app.cell(hide_code=True)
def _(center, show_source, simulate_picks):
    show_source(simulate_picks, center)
    return


@app.cell
def _(center, sigma, simulate_picks, torch, v_ref, xi_x, xi_y):
    # One synthetic earthquake: true source, unknown origin time, noisy picks.
    m_x_true = torch.tensor(55.0, dtype=torch.float64)  # km
    m_y_true = torch.tensor(45.0, dtype=torch.float64)  # km
    t0 = 12.3  # s
    torch.manual_seed(0)  # for reproducibility
    d_obs = simulate_picks(m_x_true, m_y_true, xi_x, xi_y, v_ref, sigma, t0)
    d_obs_c = center(d_obs)  # centred noisy observations
    return (d_obs_c,)


@app.cell(hide_code=True)
def _(artifact, mo):
    mo.vstack(
        [
            mo.md(r"""
            ## A measure of misfit

            We can test how well a source location $\mathbf{m}$ agrees with the data by
            calculating its $\mathcal{L}_2$ misfit $S(\mathbf{m})$.
            """),
            artifact("misfit", width="80%"),
            mo.md(r"""
            The full version weights the residuals by their data uncertainty. We assume
            them to be constant and leave them out, since they would only rescale the
            loss. For Gaussian noise of equal size $\sigma$ the smallest $S$ is the
            maximum-likelihood source.
            """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why do we need gradients

    The misfit $S(\mathbf{m})$ tells us how good a guess is, not where to look next.

    With two unknowns we could evaluate $S$ on a grid, but the cost grows
    exponentially with the number of unknowns: $100$ values per parameter means
    $100^2$ evaluations for a location, $100^{10}$ for ten parameters. The networks
    later in the course have thousands to millions.

    The **gradient** $-\nabla S$ points towards lower misfits and we can follow it
    to the (local) minimum.

    $$
    \mathbf{m} \leftarrow \mathbf{m} - \eta \, \nabla S(\mathbf{m})
    $$

    with a (small) step size $\eta$. So we would like to calculate $\nabla S$ cheaply
    and for any forward model, not just this one.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The misfit as a graph

    While we could, for this simple forward model, calculate the gradient by hand, let's
    see how we can do it in a more general way.

    A **computational graph** records how a number is built from others:

    * every box is a list of numbers (a *tensor*)
    * every arrow is an operation $f_k$.

    A graph for $S$ reads left to right, from the unknowns $m_x, m_y$ to the single
    number $S$.

    We use only **elementary** operations to keep everything as general and also as
    simple as possible. We will see later how this simplifies taking gradients a lot.
    """)
    return


@app.cell(hide_code=True)
def _(artifact):
    artifact("computational_chain", mode="graph")
    return


@app.cell
def _(torch, v_ref, xi_x, xi_y):
    def misfit_forward(m_x, m_y, d_obs_c):
        """Misfit S(m) for centred data d_obs_c: one elementary operation per line."""
        delta_x = m_x - xi_x
        delta_y = m_y - xi_y
        delta_x2 = delta_x**2
        delta_y2 = delta_y**2
        Delta2 = delta_x2 + delta_y2
        Delta = torch.sqrt(Delta2)
        F = Delta / v_ref
        F_mean = F.mean()
        F_c = F - F_mean
        r = F_c - d_obs_c
        r2 = r**2
        r2_sum = r2.sum()
        S = 0.5 * r2_sum
        return S

    m_guess = (30.0, 70.0)  # trial source (m_x, m_y) in km, used in all figures
    return m_guess, misfit_forward


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The chain rule

    The graph is a composition of simple functions. Call the operations $f_1, \dots,
    f_n$ and the values between them $z_k$:

    $$
    z_0 = \mathbf{m}
    \qquad
    z_k = f_k(z_{k-1})
    \qquad
    S = z_n
    $$

    Every operation has a **local derivative** $f_k' = \partial z_k / \partial
    z_{k-1}$, taken at the value the forward pass produced. The chain rule multiplies
    them:

    $$
    \frac{\partial S}{\partial z_0}
    = \frac{\partial z_n}{\partial z_{n-1}} \cdots
    \frac{\partial z_2}{\partial z_1}\,\frac{\partial z_1}{\partial z_0}
    = f_n' \cdots f_2'\, f_1'
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We evaluate the product **backwards**, from $S$ to $\mathbf{m}$. $S$ is a single
    number, so one sweep gives the gradient with respect to every input, however
    many there are. Write $g_k \equiv \partial S / \partial z_k$ for the gradient at
    the value $z_k$. At the output it is 1, and every step back multiplies by one
    local derivative:

    $$
    g_n = 1
    \qquad
    g_{k-1} = g_k \, f_k'
    $$

    An operation never needs to know the rest of the graph. It takes the gradient that
    arrives from the right (the upstream gradient), multiplies it by its own local
    derivative and hands the result to the left. Non-linear operations need the values
    of the forward pass, so the forward pass keeps them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One operation, two inputs

    With letters instead of indices, call the input $u = z_{k-1}$ and the output
    $z = z_k$. An operation can have a second input $w$, and each input gets its own
    local derivative:

    $$
    g_u = g_z\,\frac{\partial z}{\partial u}
    \qquad
    g_w = g_z\,\frac{\partial z}{\partial w}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(artifact):
    artifact("local_derivatives", max_scale=4.0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Two bookkeeping rules

    **Fan-out.** If a value is used by several operations, the gradients flowing back
    from each of them **add up**. With several stations, the model $\mathbf{m}$ feeds
    the travel time of every station, so its gradient is the sum of the contributions
    from all of them.

    **Broadcast.** A single number used for all $N$ stations, such as $m_x$ in
    $m_x - \boldsymbol{\xi}_x$ or $\langle F \rangle$ in $\mathbf{F} - \langle F
    \rangle$, is really $N$ copies of it. Copying is a fan-out, so its backward step
    is a **sum over stations**:

    $$
    \frac{\partial S}{\partial m_x}
    = \sum_{i=1}^{N} \frac{\partial S}{\partial \delta_{x,i}}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(artifact):
    artifact("computational_chain", mode="backward")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Chain rule, multiplied out

    Multiplying the local derivatives from $S$ back to $\mathbf{m}$ gives

    $$
    \frac{\partial S}{\partial m_x}
    = \sum_i r_i \,\frac{1}{v}\,\frac{m_x - \xi_{i,x}}{\Delta_i}
    \qquad
    \frac{\partial S}{\partial m_y}
    = \sum_i r_i \,\frac{1}{v}\,\frac{m_y - \xi_{i,y}}{\Delta_i}
    $$

    In vector form $\nabla S = \mathbf{G}^\top \mathbf{r}$, where the columns of
    $\mathbf{G}$ are

    $$
    G_{i,x} = \frac{\partial F_i}{\partial m_x}
    = \frac{1}{v}\,\frac{m_x - \xi_{i,x}}{\Delta_i}
    $$

    and $G_{i,y}$ likewise with $y$.

    The backward pass computes $\mathbf{G}^\top \mathbf{r}$ **without ever forming**
    $\mathbf{G}$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Let torch record the graph, we supply the rules

    PyTorch builds the graph while the forward pass runs and walks it backwards when
    we call `S.backward()`. For every operation it needs just two functions, which we
    write ourselves with `torch.autograd.Function`:

    - `forward(ctx, ...)` computes $z$ and stores in `ctx` what `backward` will need,
    - `backward(ctx, grad_out)` receives $g_z = \partial S/\partial z$
      (`grad_out`) and returns the gradients of the inputs, $g_z$ times the local
      derivatives.

    The linear operations come first, they are the easier ones. The two nonlinear
    operations follow.
    """)
    return


@app.cell
def _(torch):
    def unbroadcast(grad, shape):
        """A scalar that was copied to N stations gets the sum of the N gradients."""
        return grad.sum() if shape == () else grad

    class Sub(torch.autograd.Function):
        @staticmethod
        def forward(ctx, u, w):
            ctx.shapes = u.shape, w.shape
            return u - w  # u or w may be a scalar broadcast over the stations

        @staticmethod
        def backward(ctx, grad_out):
            shape_u, shape_w = ctx.shapes
            return unbroadcast(grad_out, shape_u), -unbroadcast(grad_out, shape_w)

    class Add(torch.autograd.Function):
        @staticmethod
        def forward(ctx, u, w):
            return u + w

        @staticmethod
        def backward(ctx, grad_out):
            return grad_out, grad_out

    class Scale(torch.autograd.Function):
        @staticmethod
        def forward(ctx, u, c):
            ctx.c = c
            return c * u

        @staticmethod
        def backward(ctx, grad_out):
            return ctx.c * grad_out, None  # no gradient for the constant c

    class Mean(torch.autograd.Function):
        @staticmethod
        def forward(ctx, u):
            ctx.shape = u.shape
            return u.mean()

        @staticmethod
        def backward(ctx, grad_out):
            return grad_out.expand(ctx.shape) / ctx.shape.numel()

    class Sum(torch.autograd.Function):
        @staticmethod
        def forward(ctx, u):
            ctx.shape = u.shape
            return u.sum()

        @staticmethod
        def backward(ctx, grad_out):
            return grad_out.expand(ctx.shape)

    return Add, Mean, Scale, Sub, Sum


@app.cell
def _(torch):
    class Square(torch.autograd.Function):
        @staticmethod
        def forward(ctx, u):
            ctx.save_for_backward(u)  # the local derivative 2u needs the input
            return u**2

        @staticmethod
        def backward(ctx, grad_out):
            (u,) = ctx.saved_tensors
            return 2 * u * grad_out

    class Sqrt(torch.autograd.Function):
        @staticmethod
        def forward(ctx, u):
            z = torch.sqrt(u)
            ctx.save_for_backward(z)  # the local derivative 1/(2z) needs the output
            return z

        @staticmethod
        def backward(ctx, grad_out):
            (z,) = ctx.saved_tensors
            return grad_out / (2 * z)

    return Sqrt, Square


@app.cell
def _(Add, Mean, Scale, Sqrt, Square, Sub, Sum, v_ref, xi_x, xi_y):
    sub, add, square, sqrt = Sub.apply, Add.apply, Square.apply, Sqrt.apply
    scale, mean, total = Scale.apply, Mean.apply, Sum.apply

    def misfit_custom(m_x, m_y, d_obs_c):
        """Same lines as misfit_forward, but every operation is one of our own."""
        delta_x = sub(m_x, xi_x)
        delta_y = sub(m_y, xi_y)
        delta_x2 = square(delta_x)
        delta_y2 = square(delta_y)
        Delta2 = add(delta_x2, delta_y2)
        Delta = sqrt(Delta2)
        F = scale(Delta, 1 / v_ref)
        F_mean = mean(F)
        F_c = sub(F, F_mean)
        r = sub(F_c, d_obs_c)
        r2 = square(r)
        r2_sum = total(r2)
        return scale(r2_sum, 0.5)

    return (misfit_custom,)


@app.cell
def _(d_obs_c, m_guess, misfit_custom, torch):
    m_x_c = torch.tensor(m_guess[0], dtype=torch.float64, requires_grad=True)
    m_y_c = torch.tensor(m_guess[1], dtype=torch.float64, requires_grad=True)
    S_custom = misfit_custom(m_x_c, m_y_c, d_obs_c)  # torch records the graph here
    S_custom.backward()  # and calls our backward methods, last operation first
    grad_custom = torch.stack([m_x_c.grad, m_y_c.grad])

    print(f"S = {S_custom.item():.4f} s^2")
    print(f"grad S = ({grad_custom[0]:.4f}, {grad_custom[1]:.4f}) s^2/km")
    return (grad_custom,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Now let torch do all of it

    The first forward function, `misfit_forward`, used plain operators and plain
    `torch` functions. Give it leaf tensors with `requires_grad=True` and torch
    records the graph and knows every local derivative by itself, including the
    broadcasting and the fan-out. No custom code at all:
    """)
    return


@app.cell
def _(d_obs_c, m_guess, misfit_forward, torch):
    # Now let torch do all of it: give the plain forward function leaf tensors with
    # requires_grad=True and torch records the graph and knows every local
    # derivative itself, including the broadcasting and the fan-out.
    m_x_t = torch.tensor(m_guess[0], dtype=torch.float64, requires_grad=True)
    m_y_t = torch.tensor(m_guess[1], dtype=torch.float64, requires_grad=True)
    S_torch = misfit_forward(m_x_t, m_y_t, d_obs_c)  # plain torch operations
    S_torch.backward()
    grad_torch = torch.stack([m_x_t.grad, m_y_t.grad])
    return (grad_torch,)


@app.cell(hide_code=True)
def _(grad_custom, grad_torch, mo):
    mo.md(f"""
    $\\nabla S$ from torch alone = ({grad_torch[0]:.4f}, {grad_torch[1]:.4f}),
    from our hand-made backward pass = ({grad_custom[0]:.4f}, {grad_custom[1]:.4f}).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Walking downhill

    With $\nabla S$ we can walk downhill. From a trial source $\mathbf{m}_j$ take a step
    against the gradient,

    $$
    \mathbf{m}_{j+1} = \mathbf{m}_j - \eta\,\nabla S(\mathbf{m}_j)
    $$

    with a small step size $\eta$.
    """)
    return


@app.cell(hide_code=True)
def _(artifact):
    artifact("gradient_descent")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Optimisers: SGD, momentum, Adam

    We now have a rule that finds a (local) minimum. Is it good enough, especially if we
    want to use it for neural networks later? Plain gradient descent,
    $\mathbf{m}_{j+1} = \mathbf{m}_j - \eta\,\nabla S(\mathbf{m}_j)$, has two
    weaknesses:

    - In a narrow valley the gradient points mostly across it, so the path zigzags.
    - Noisy gradients make the path jittery.

    An **optimiser** is a rule that turns the gradient into the next step. Three common
    ones:

    - **SGD** (stochastic gradient descent) is the plain rule above. "Stochastic" means
      that the gradient is computed on a random mini-batch of the data, which is cheaper
      but noisier. Here one event has few stations, so the playground uses all of them.
    - **Momentum** keeps a running step $\mathbf{s}$ that averages recent gradients
      (momentum $\mu$, typically $0.9$):
      $\mathbf{s}_{j+1} = \mu\,\mathbf{s}_j + \nabla S(\mathbf{m}_j)$, then
      $\mathbf{m}_{j+1} = \mathbf{m}_j - \eta\,\mathbf{s}_{j+1}$. Zigzags across the
      valley cancel, the steady push along it builds up, and noise averages out.
      Setting $\mu = 0$ gives plain SGD back.
    - **Adam** divides every component of the step by a running estimate of its
      typical size, so all components move at a similar speed, whatever their units.
      It works very well for neural networks.
    """)
    return


@app.cell(hide_code=True)
def _(playground):
    editor = playground.make_editor()
    return (editor,)


@app.cell(hide_code=True)
def _(editor, playground):
    # The scene as dragged on the map: stations, picks d_obs, forward model, start.
    problem, m_start = playground.read_scene(editor)
    return m_start, problem


@app.cell
def _(center, editor, m_start, playground, problem, torch):
    method, lr, momentum, steps = playground.optimiser_settings(editor)  # header

    def misfit(m):
        """Misfit S (km^2) of model(s) m with shape (..., 2)."""
        F = problem.travel_times(m[..., 0], m[..., 1])
        r = problem.v_ref * (center(F) - center(problem.d_obs))
        return 0.5 * (r**2).mean(-1)

    m = m_start.clone().requires_grad_()
    if method == "SGD":
        optimiser = torch.optim.SGD([m], lr=lr, momentum=momentum)
    else:
        optimiser = torch.optim.Adam([m], lr=lr)

    path = [m.detach().clone()]  # the iterates, drawn on the maps
    for _ in range(steps):
        optimiser.zero_grad()
        S = misfit(m)
        S.backward()  # autograd: gradient of S
        optimiser.step()  # m <- m - lr * gradient (or momentum, Adam)
        path.append(m.detach().clone())
    return misfit, path


@app.cell(hide_code=True)
def _(editor, mo):
    mo.vstack(
        [
            editor,
            mo.md(r"""
    ### Four things to look for

    Change one thing at a time and watch the path and the misfit curve.

    - **Step size.** How large a step can plain gradient descent take?
    - **Momentum.** What does a running average of the gradient buy, and what does it
      cost?
    - **Adam.** Does the step size $\eta$ mean the same for Adam as for SGD?
    - **Start.** Does the answer depend on where the path starts?
        """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How the playground works

    **Drag** the source (star), the start (circle) and any station (triangle). The
    header sets the scene and the optimiser; ▶ plays the run, ⏭ steps, ↺ restarts, and
    any change re-runs it. The *heterogeneous* P-speed field is random and bends rays,
    and *random field* picks one of six.

    The page opens with 5 stations, SGD without momentum, $\eta$ = 3 and 10 steps.
    """)
    return


@app.cell(hide_code=True)
def _(try_it):
    try_it(
        [
            (
                "Step size",
                "How large a step can plain gradient descent take?",
                "SGD, momentum 0, 50 steps; $\\eta$ from 0.5 up to 5.",
                "A small $\\eta$ creeps. Around 1 to 3 the misfit falls and levels off "
                "at the noise. At 4 or 5 it overshoots and the misfit stays high.",
            ),
            (
                "Momentum",
                "What does momentum cost?",
                "SGD, $\\eta$ = 3, 25 steps; momentum 0, then 0.5, 0.7, 0.9.",
                "At 0.5 little changes. Beyond that the steps add up, the path "
                "overshoots the source and swings back; at 0.9 it flies off the map.",
            ),
            (
                "Momentum",
                "What does a narrow valley do to the path?",
                "Drag the five stations onto a horizontal line, SGD, $\\eta$ = 0.5, "
                "50 steps; momentum 0, then 0.5 and 0.7.",
                "The misfit is a long valley along the line. Without momentum the "
                "path crawls along it; with 0.5 to 0.7 it reaches a lower misfit in "
                "the same steps.",
            ),
            (
                "Adam",
                "Does $\\eta$ mean the same for Adam as for SGD?",
                "$\\eta$ = 5 with SGD, then with Adam.",
                "SGD at 5 overshoots. Adam moves each component by about $\\eta$ km "
                "per step whatever the size of the gradient, so its steps are bounded "
                "and the path settles near the source.",
            ),
            (
                "Start",
                "Does the answer depend on where the path starts?",
                "*Heterogeneous* P speed, one *random field* after another; drag the "
                "start (circle) to several places, 50 steps. Compare *homogeneous*.",
                "In a homogeneous medium every start ends at the source. In a random "
                "field some starts end in a local minimum: the misfit stops above the "
                "noise level, far from the source.",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(editor, misfit, path, playground, problem):
    # send the run to the maps above; shows a warning if it went wrong
    playground.show_run(editor, problem, path, misfit)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What we keep

    - The misfit $S(\mathbf{m})$ says how good a guess is. Its gradient says where to
      go next.
    - We write the **forward** computation; torch derives the **backward** one by
      multiplying local derivatives from $S$ back to $\mathbf{m}$.
    - Custom `autograd.Function`s remain useful where torch cannot differentiate, for
      example a numerical solver that supplies its own gradient.
    - From here on we set `requires_grad=True`, call `.backward()` and read `.grad`.
      An optimiser (SGD, momentum, Adam) turns that gradient into steps.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From one event to many: learn the inverse map

    So far we solved **one** earthquake at a time: start somewhere, walk downhill.
    A neural network does something different. It learns the whole inverse map
    $\tilde{\mathbf{d}} \mapsto \mathbf{m}$ from examples, so that a new event is
    located with one evaluation instead of one optimisation.
    """)
    return


if __name__ == "__main__":
    app.run()
