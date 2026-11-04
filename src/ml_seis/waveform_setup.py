"""Small, side-effect-free setup helpers for Part 2."""

import tempfile
from pathlib import Path


def _lines(*parts: str) -> str:
    """Join markdown lines; keeps multi-line inserts flush-left for ``mo.md``."""
    return "\n".join(parts)


def default_cache_dir(notebook_dir) -> Path:
    """Where the downloaded waveforms go.

    In a clone of the repository: ``data/waveforms`` next to ``lectures/``. A notebook
    run on its own (molab, a downloaded file) has no repository around it, so it uses
    ``~/.cache/ml_seis/waveforms``, or a temporary folder if that cannot be written.
    """
    if notebook_dir is not None:
        root = Path(notebook_dir).parent
        if (root / "pyproject.toml").is_file() and (root / "src" / "ml_seis").is_dir():
            return root / "data" / "waveforms"
    for base in (Path.home() / ".cache", Path(tempfile.gettempdir())):
        folder = base / "ml_seis" / "waveforms"
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        return folder
    raise OSError("no writable folder for the downloaded waveforms")


def cache_ready(cache_path) -> bool:
    """True when a finished cache file is on disk (writes are atomic)."""
    return Path(cache_path).is_file()


def llaima_card(mo, data, cache_path, classes_names):
    """Source line and records per class and split of the cached Llaima data."""
    size = Path(cache_path).stat().st_size / 1e6
    doi = data.provenance["source"][4:]
    per_split = [data.counts(split) for split in ("train", "validation", "test")]
    rows = [
        f"| `{c}` | {classes_names[c]} | {int(per_split[0][k])} "
        f"| {int(per_split[1][k])} | {int(per_split[2][k])} "
        f"| {int(data.counts()[k])} |"
        for k, c in enumerate(data.classes)
    ]
    total = [int(n.sum()) for n in per_split]
    return mo.md(
        _lines(
            f"**Llaima volcano, Chile** — station LAV, vertical component, "
            f"{len(data.label)} records of {data.u.shape[-1]} samples at "
            f"{data.sampling_rate:.0f} Hz. Canario et al. (2020), doi:{doi}, "
            f"CC BY-NC 3.0. Cached locally ({size:.0f} MB).",
            "",
            "| label | meaning | train | validation | test | total |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
            *rows,
            f"| | | **{total[0]}** | **{total[1]}** | **{total[2]}** "
            f"| **{sum(total)}** |",
        )
    )


def ceed_card(mo, data, cache_path):
    """Source, selection and size of the cached CEED picking subset."""
    size = Path(cache_path).stat().st_size / 1e6
    meta = data.provenance
    events = len(set(data.event_id))
    return mo.md(
        _lines(
            f"**CEED, Southern California {meta['year']}** — three-component records "
            f"with analyst P and S picks (SCSN/SCEDC data, packaged by Zhu et al., "
            f"2025). Pinned revision `{meta['revision'][:8]}` of "
            f"`{meta['source'].split('datasets/')[-1]}`; the "
            f"{meta['download_mb']:.0f} MB year file was downloaded once and reduced "
            f"to a {size:.0f} MB cache: `{cache_path}`.",
            "",
            f"{len(data.p)} traces from {events} earthquakes kept (three real "
            f"components, both picks present, S minus P below 23 s, non-zero analyst "
            f"weights).",
        )
    )


def shape_card(mo, rows, receptive_samples, sampling_rate):
    """Tensor shapes through a network and the receptive field in seconds."""
    seconds = receptive_samples / sampling_rate
    return mo.md(
        _lines(
            "| after layer | shape (batch, channel, sample) |",
            "| --- | --- |",
            *[f"| {name} | `{shape}` |" for name, shape in rows],
            "",
            f"Each output value of the last convolution depends on "
            f"**{receptive_samples} input samples = {seconds:.1f} s** of the record "
            f"(its receptive field).",
        )
    )


def settings_card(mo, settings: dict):
    """Resolved training configuration as one compact line."""
    text = ", ".join(f"{k} = `{v}`" for k, v in settings.items())
    return mo.md(f"**Configuration:** {text}")


def pick_card(mo, summary_p: dict, summary_s: dict):
    """Pick-error summary for P and S at the chosen threshold."""

    def row(name, s):
        return (
            f"| {name} | {s['median_abs']:.2f} | {s['bias']:+.2f} | {s['rms']:.2f} "
            f"| {100 * s['within_0.1']:.0f}% | {100 * s['within_0.25']:.0f}% "
            f"| {100 * s['within_0.5']:.0f}% | {100 * s['miss_rate']:.0f}% |"
        )

    header = "| phase | median abs (s) | bias (s) | rms (s) |"
    header += " <0.1 s | <0.25 s | <0.5 s | missed |"
    rule = "| --- |" + " ---: |" * 7
    return mo.md("\n".join([header, rule, row("P", summary_p), row("S", summary_s)]))
