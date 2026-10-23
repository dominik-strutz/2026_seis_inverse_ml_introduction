# AI for inverse problems in seismology

A two-lecture course in which machine learning is introduced as an inverse problem. Lectures and exercises are [marimo](https://marimo.io) notebooks written
mostly in plain [PyTorch](https://pytorch.org).

| Part | Notebook | Theme |
|---|---|---|
| 1a | `lectures/1_a_locating_with_gradients.py` | Location by physics: forward model, misfit, autodiff, gradient descent |
| 1b | `lectures/1_b_learning_the_inverse_map.py` | A network learns picks to location from simulated events |
| 1c | `lectures/1_c_learning_a_posterior.py` | From a point to a distribution: mixture density networks, coverage |
| 2a | `lectures/2_a_filters_on_waveforms.py` | Filters, feature maps, pooling, the receptive field |
| 2b | `lectures/2_b_classifying_waveforms.py` | Training and judging a waveform classifier |
| 2c | `lectures/2_c_learning_without_labels.py` | An autoencoder, its code, clusters named with a few labels |
| 2d | `lectures/2_d_picking_phases.py` | A U-Net that picks P and S, and what its error means for location |

The running problem is earthquake epicentre location in horizontal map coordinates
(Southern California). The source parameters are `(m_x, m_y)`; origin time is removed
by centring arrivals over stations. Part 1 uses P arrivals, and Part 2 learns
P/S picks from real waveforms.

> Status: Parts 1 (1a to 1c) and 2 (2a to 2d) are built. Homework is not written yet.

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
   uv run marimo edit lectures/2_a_filters_on_waveforms.py
   ```

Every notebook also declares its own dependencies (PEP 723), so you can run a single notebook in an isolated environment from the repo root:

```bash
uvx marimo edit --sandbox lectures/1_a_locating_with_gradients.py
```

(needs uv >= 0.7.21). Parts 1a and 1b install `ml_seis` from GitHub in this mode, so it
needs a network connection. With a clone and `uv sync`, the `uv run marimo edit`
commands above work offline and use your local copy of `src/ml_seis`.

**GPU:** the project installs CPU-only PyTorch for a small, portable install. To use a GPU, remove the `[tool.uv.sources]` / `[[tool.uv.index]]` PyTorch entries in `pyproject.toml` and run `uv sync` again.

## Presenting as slides

The lecture notebooks are laid out as slides (`layouts/*.slides.json` and
`lecture_slides.css` in `lectures/`).

```bash
# Lecturing: read-only slides, code hidden (add --include-code to show it)
uv run marimo run lectures/1_a_locating_with_gradients.py
uv run marimo run lectures/2_a_filters_on_waveforms.py

# Preparing: editor with the slide layout; pick "Slides" in the layout menu
# (or open the app view) to preview, edit cells and re-run live
uv run marimo edit lectures/1_a_locating_with_gradients.py

# Editing in your own editor (VS Code, ...) while watching the slides in the browser:
# the slides reload each time you save the file
uv run marimo run --no-sandbox --watch lectures/1_a_locating_with_gradients.py
```

Useful flags: `--headless` (do not open a browser), `--port 2718`, `--watch` (reload
when the file changes). Slides advance with the arrow keys; `f` toggles full screen.
The first run of Part 2 downloads its datasets (hundreds of MB) into `data/`.

## Browser (no install)

[molab](https://molab.marimo.io) runs a notebook from this repository in the browser,
with nothing to install. A link opens a preview of the notebook with its saved
outputs; run it on molab's server to change the controls and train the networks.

| Part | Open in molab |
|---|---|
| 1a Location by physics | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/dominik-strutz/2026_seis_inverse_ml_introduction/blob/main/lectures/1_a_locating_with_gradients.py) |
| 1b Learning the inverse map | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/dominik-strutz/2026_seis_inverse_ml_introduction/blob/main/lectures/1_b_learning_the_inverse_map.py) |

Parts 1c and 2 are not on molab yet. The notebooks above run on their own: the
`# /// script` header at the top of each one installs `ml_seis` from this repository,
so no other file is needed. The slide layout (`layouts/`, `lecture_slides.css`) only
applies in a clone. molab has PyTorch preinstalled and an optional GPU, but only about
1 GB of persistent storage, so datasets are downloaded at runtime.

**Offline.** From a clone, everything in Parts 1a and 1b works without a network
connection. The network is needed only for the first `uv sync` and, in Part 2, for the
dataset downloads, which are cached in `data/` afterwards.

## Contents

```
lectures/   slide notebooks, one per lecture (layouts/ for slides)
homework/   one homework notebook per lecture (none written yet)
src/ml_seis shared non-ML helpers (physics, priors, widgets, data, plotting, artifacts/)
scripts/    instructor-side data and figure builds
data/       downloaded and generated data (not in git)
```

## References

**Inverse theory** (notation and removing the origin time)

- Tarantola, A. (2005). *Inverse Problem Theory and Methods for Model Parameter Estimation*. SIAM.
- Tarantola, A., and Valette, B. (1982). Inverse problems = quest for information. *Journal of Geophysics*, 50, 159–170.

**Machine learning**

- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press. Free online draft: <https://probml.github.io/pml-book/book1.html>
- Kingma, D. P., and Ba, J. (2015). Adam: a method for stochastic optimization. *ICLR*. arXiv:1412.6980
- Ioffe, S., and Szegedy, C. (2015). Batch normalization: accelerating deep network training by reducing internal covariate shift. *ICML*. arXiv:1502.03167
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. *JMLR*, 15, 1929–1958.
- Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: convolutional networks for biomedical image segmentation. *MICCAI*. arXiv:1505.04597
- Paszke, A., et al. (2019). PyTorch: an imperative style, high-performance deep learning library. *NeurIPS 32*. arXiv:1912.01703

**Posteriors from networks** (Part 1c)

- Bishop, C. M. (1994). Mixture density networks. Technical report NCRG/94/004, Aston University.
- Meier, U., Curtis, A., and Trampert, J. (2007). Global crustal thickness from neural network inversion of surface wave data. *Geophysical Journal International*, 169, 706–722. doi:10.1111/j.1365-246X.2007.03373.x
- Käufl, P., Valentine, A. P., O'Toole, T. B., and Trampert, J. (2014). A framework for fast probabilistic centroid-moment-tensor determination: inversion of regional static displacement measurements. *Geophysical Journal International*, 196.
- Papamakarios, G., and Murray, I. (2016). Fast ε-free inference of simulation models with Bayesian conditional density estimation. *NeurIPS 29*. arXiv:1605.06376
- Cranmer, K., Brehmer, J., and Louppe, G. (2020). The frontier of simulation-based inference. *PNAS*, 117(48), 30055–30062.

**Data**

- Canário, J. P., Mello, R. F., Curilem, M., Huenupan, F., and Rios, R. A. (2020). Llaima volcano dataset: in-depth comparison of deep artificial neural network architectures on seismic events classification. *Data in Brief*, 30, 105627. Data: doi:10.17632/dv8nwdd36k.1 (CC BY-NC 3.0)
- Zhu, W., et al. (2025). California earthquake dataset for machine learning and cloud computing (CEED). arXiv:2502.11500. Records from SCSN/SCEDC.
- USGS ComCat earthquake catalogue (2019 Ridgecrest sequence), used for the `ridgecrest` prior: <https://earthquake.usgs.gov/fdsnws/event/1/>

## For contributors

Formatting and linting use ruff. Before sending changes:

```bash
uv run marimo check lectures homework solutions
uv run ruff check . && uv run ruff format --check .
```

## License

[MIT](LICENSE)
