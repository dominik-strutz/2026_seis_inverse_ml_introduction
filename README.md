# AI for inverse problems in seismology

A short introduction to some key machine learning concepts as part of a course on inverse problems for geophysicists. Slides are [marimo](https://marimo.io) notebooks written mostly in plain [PyTorch](https://pytorch.org), functioning as the slide set, the lecture notes and code examples in one.

| Part | Notebook | Theme | Browser |
|---|---|---|---|
| 1a | `lectures/1_a_locating_with_gradients.py` | Location by physics: forward model, misfit, autodiff, gradient descent | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/dominik-strutz/2026_seis_inverse_ml_introduction/blob/main/lectures/1_a_locating_with_gradients.py) |
| 1b | `lectures/1_b_learning_the_inverse_map.py` | A network learns picks to location from simulated events | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/dominik-strutz/2026_seis_inverse_ml_introduction/blob/main/lectures/1_b_learning_the_inverse_map.py) |
| 1c | `lectures/1_c_learning_a_posterior.py` | From a point to a distribution: mixture density networks | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/dominik-strutz/2026_seis_inverse_ml_introduction/blob/main/lectures/1_c_learning_a_posterior.py) |

> [!NOTE]
> The slides rely heavily on interactive figures and playgrounds, and most of that
> was written with the help LLms: the anywidget
> playgrounds and the JavaScript that drives them, the standalone HTML figures
> (HTML, SVG and JavaScript), and the slide layouts and styling.
>
> I tested the outputs extensively, and wherever possible the underlying logic
> relies on transparent and concise PyTorch code. The large amount of HTML,
>  CSS and JavaScrip for interactive layers themselfs has not been reviewed 
> by me, I am neither qualified to review it, and it would take a lot of time.
>
> If anything seems off, please get in touch or
> [open an issue](https://github.com/dominik-strutz/2026_seis_inverse_ml_introduction/issues).

## Quick start

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Get the code and the environment (Python 3.13 and CPU-only PyTorch are handled by uv):

   ```bash
   git clone https://github.com/dominik-strutz/2026_seis_inverse_ml_introduction.git ml_seis
   cd ml_seis
   uv sync
   ```

3. Open a notebook (all commands from the repo root):

   ```bash
   uv run marimo edit lectures/1_a_locating_with_gradients.py   # edit / run
   ```

Every notebook also declares its own dependencies (PEP 723), so you can run a single notebook in an isolated environment from the repo root:

```bash
uvx marimo edit --sandbox lectures/1_a_locating_with_gradients.py
```

(needs uv >= 0.7.21). Parts 1a to 1c install `ml_seis` from GitHub in this mode, so it
needs a network connection. With a clone and `uv sync`, the `uv run marimo edit`
commands above work offline and use your local copy of `src/ml_seis`.

**GPU:** the project installs CPU-only PyTorch for a small, portable install. To use a GPU, remove the `[tool.uv.sources]` / `[[tool.uv.index]]` PyTorch entries in `pyproject.toml` and run `uv sync` again.

## Presenting as slides

The lecture notebooks are laid out as slides. Each notebook embeds its slide
layout in its own header (a base64 data URI in `layout_file`), so the deck —
slide order, fragments, hidden cells — also works on molab and for anyone who
downloads a single notebook; `lectures/layouts/*.slides.json` holds the same
layouts as the editable source, and `make layouts` re-embeds them. The slide
styling (`lecture_slides.css`, `slides_head.html` in `lectures/`) is read from
disk next to the notebook and therefore applies to local slides only.

For the three notebooks linked above, the repository also contains marimo session
snapshots. They give molab its ready-made preview outputs instead of making a reader
wait for every cell to run. After changing a published notebook or a helper it uses,
run `make snapshots`: it embeds layouts first, rebuilds those previews, and records
the exact input hash. `make hooks` installs pre-commit checks that reject stale
previews or layouts.

```bash
# Lecturing: read-only slides, code hidden (add --include-code to show it)
uv run marimo run lectures/1_a_locating_with_gradients.py

# Preparing: editor with the slide layout; pick "Slides" in the layout menu
# (or open the app view) to preview, edit cells and re-run live
uv run marimo edit lectures/1_a_locating_with_gradients.py

# Editing in your own editor (VS Code, ...) while watching the slides in the browser:
# the slides reload each time you save the file
uv run marimo run --no-sandbox --watch lectures/1_a_locating_with_gradients.py
```

## Contents

```
lectures/   slide notebooks, one per lecture (layouts/ for slides)
homework/   one homework notebook per lecture (none written yet)
src/ml_seis shared non-ML helpers (physics, priors, widgets, data, plotting, artifacts/)
scripts/    instructor-side data and figure builds
data/       downloaded and generated data (not in git)
```

## References

**General**

- Tarantola, A. (2005). *Inverse Problem Theory and Methods for Model Parameter Estimation*. SIAM. doi:10.1137/1.9780898717921 (also the notation of the course)
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press. Free online: <https://probml.github.io/pml-book/book1.html>

**Inverse theory** (origin-time removal; Part 1a)

- Tarantola, A., and Valette, B. (1982). Inverse problems = quest for information. *Journal of Geophysics*, 50, 159–170.

**Gradients and optimisation** (Parts 1a, 1b)

- Rumelhart, D. E., Hinton, G. E., and Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533–536. doi:10.1038/323533a0
- Polyak, B. T. (1964). Some methods of speeding up the convergence of iteration methods. *USSR Computational Mathematics and Mathematical Physics*, 4(5), 1–17. doi:10.1016/0041-5553(64)90137-5
- Kingma, D. P., and Ba, J. (2015). Adam: a method for stochastic optimization. *ICLR*. arXiv:1412.6980

**Posteriors from networks** (Part 1c)

- Bishop, C. M. (1994). Mixture density networks. Technical report NCRG/94/004, Aston University.
- Meier, U., Curtis, A., and Trampert, J. (2007). Global crustal thickness from neural network inversion of surface wave data. *Geophysical Journal International*, 169, 706–722. doi:10.1111/j.1365-246X.2007.03373.x
- Käufl, P., Valentine, A. P., O'Toole, T. B., and Trampert, J. (2014). A framework for fast probabilistic centroid-moment-tensor determination: inversion of regional static displacement measurements. *Geophysical Journal International*, 196(3), 1676–1693. doi:10.1093/gji/ggt473
- Zhao, X., and Curtis, A. (2024). Variational prior replacement in Bayesian inference and inversion. *Geophysical Journal International*, 239(2), 1236–1256. doi:10.1093/gji/ggae334 (combining a trained density with new information)

**Software and data**

- Paszke, A., et al. (2019). PyTorch: an imperative style, high-performance deep learning library. *NeurIPS 32*. arXiv:1912.01703
- Sethian, J. A. (1996). A fast marching level set method for monotonically advancing fronts. *PNAS*, 93(4), 1591–1595. doi:10.1073/pnas.93.4.1591 (travel times in the heterogeneous model, via [scikit-fmm](https://github.com/scikit-fmm/scikit-fmm))
- [marimo](https://marimo.io) notebooks and [anywidget](https://anywidget.dev) widgets.
- U.S. Geological Survey, ANSS Comprehensive Earthquake Catalog (ComCat), 2019 Ridgecrest sequence, used for the `ridgecrest` prior: <https://earthquake.usgs.gov/fdsnws/event/1/>

## License

[MIT](LICENSE)
