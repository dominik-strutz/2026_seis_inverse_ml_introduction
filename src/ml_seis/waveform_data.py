"""Real waveform data for Part 2: download, cache, parse, split, crop.

Two datasets, both fetched at run time into ``data/waveforms`` (git-ignored):

* **Llaima** volcano-seismic classification set (Canario et al., 2020, CC BY-NC 3.0),
  3,592 vertical-component records of station LAV, four analyst classes.
* **CEED / Southern California** phase-picking subset: three-component records with
  analyst P and S picks (Zhu et al., 2025; original data SCSN/SCEDC).

Everything here is non-ML plumbing. Networks, losses and training loops live in the
notebook.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LLAIMA_CLASSES = ("LP", "TR", "VT", "TC")
LLAIMA_NAMES = {
    "LP": "long-period",
    "TR": "tremor",
    "VT": "volcano-tectonic",
    "TC": "regional tectonic",
}
LLAIMA_COUNTS = {"LP": 1310, "TR": 490, "VT": 304, "TC": 1488}
LLAIMA_ZIP_URL = "https://data.mendeley.com/public-api/zip/dv8nwdd36k/download/1"
LLAIMA_DOI = "10.17632/dv8nwdd36k.1"
LLAIMA_LENGTH = 3008  # 60 s at 50 Hz after decimation by 2, padded to 16 x 188
LLAIMA_RATE = 50.0

# The old HDF5-era release keeps one file per year; 2007 is the largest one that is
# still a single laptop-sized download (361 MB, 2,416 three-component traces).
CEED_REPOSITORY = "AI4EPS/quakeflow_sc"
CEED_REVISION = "e4d532fcdb1a07149fcee51798bd4980d283637c"
CEED_URL = (
    "https://huggingface.co/datasets/{repo}/resolve/{revision}/waveform_h5/{year}.h5"
)
CEED_RATE = 100.0
CEED_WINDOW = 6000  # stored samples per trace: 15 s before P ... 45 s after
CEED_P_INDEX = 1500  # P sits at this sample of every stored window
CEED_MAX_S_MINUS_P = 2300  # samples; keeps both picks inside a 30 s crop
CEED_CHANNELS = ("E", "N", "Z")
SPLITS = ("train", "validation", "test")

_USER_AGENT = {"User-Agent": "ml-seis-course/0.1"}


# --------------------------------------------------------------------------- files


def _download(url: str, destination: Path, progress=None, expected: int = 0) -> Path:
    """Stream ``url`` to ``destination`` atomically. ``progress(fraction)`` is called
    after every megabyte; the total comes from the server, else from ``expected``
    (bytes), and the fraction stays below 1 until the file is complete."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers=_USER_AGENT)
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or expected)
        done = 0
        with open(partial, "wb") as file:
            while chunk := response.read(1 << 20):
                file.write(chunk)
                done += len(chunk)
                if progress and total:
                    progress(min(done / total, 0.99))
    partial.replace(destination)
    return destination


def _save_cache(cache: Path, **arrays) -> None:
    """Write ``cache`` atomically, so an interrupted run never leaves a broken file
    that ``cache.exists()`` would accept."""
    partial = cache.with_name(cache.stem + ".partial.npz")
    np.savez(partial, **arrays)
    partial.replace(cache)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _megabytes(path: Path) -> float:
    return path.stat().st_size / 1e6


def _splits(groups: np.ndarray, seed: int = 0, fractions=(0.7, 0.15, 0.15)):
    """Assign whole groups to train, validation, test: one split id per element."""
    unique = np.unique(groups)
    order = np.random.default_rng(seed).permutation(len(unique))
    n_train = int(round(fractions[0] * len(unique)))
    n_validation = int(round(fractions[1] * len(unique)))
    split_of = np.empty(len(unique), dtype=np.int64)
    split_of[order[:n_train]] = 0
    split_of[order[n_train : n_train + n_validation]] = 1
    split_of[order[n_train + n_validation :]] = 2
    return split_of[np.searchsorted(unique, groups)]


# --------------------------------------------------------------------------- Llaima


@dataclass(frozen=True)
class LlaimaData:
    """Llaima records as ``u`` with shape ``(N, 1, LLAIMA_LENGTH)``."""

    u: object
    label: object
    split: object
    classes: tuple
    sampling_rate: float
    provenance: dict

    def indices(self, split: str):
        return (self.split == SPLITS.index(split)).nonzero().squeeze(1)

    def pick(self, split: str, class_name: str, k: int = 0) -> int:
        """Global index of the ``k``-th record of ``class_name`` in ``split``."""
        idx = self.indices(split)
        members = idx[self.label[idx] == self.classes.index(class_name)]
        return int(members[k])

    def counts(self, split: str | None = None):
        import torch

        label = self.label if split is None else self.label[self.indices(split)]
        return torch.bincount(label, minlength=len(self.classes))


def llaima_cache_path(cache_dir) -> Path:
    return Path(cache_dir) / "llaima.npz"


def prepare_llaima(cache_dir, progress=None) -> Path:
    """Download the Llaima archive once and store a compact decimated copy.
    ``progress(fraction)`` reports the download."""
    import h5py

    cache = llaima_cache_path(cache_dir)
    if cache.exists():
        return cache
    work = Path(cache_dir) / "llaima_download"
    archive = work / "llaima.zip"
    if not archive.exists():  # a finished download survives an interrupted prepare
        _download(LLAIMA_ZIP_URL, archive, progress, expected=130_000_000)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(work / "raw")

    blocks, labels = [], []
    for k, name in enumerate(LLAIMA_CLASSES):
        (path,) = (work / "raw").rglob(f"{name.lower()}.hdf5")
        with h5py.File(path) as file:
            a = file[name][:, :, 0]
        assert len(a) == LLAIMA_COUNTS[name], (name, len(a))
        a = 2.0 * (a - 1.5)  # the source stores each record rescaled to [1, 2]
        a = a.reshape(len(a), -1, 2).mean(axis=2)  # 100 Hz -> 50 Hz, band is 1-10 Hz
        pad = LLAIMA_LENGTH - a.shape[1]
        a = np.pad(a, ((0, 0), (pad // 2, pad - pad // 2)), mode="reflect")
        blocks.append(a.astype(np.float16))
        labels.append(np.full(len(a), k, dtype=np.int64))
    x, label = np.concatenate(blocks), np.concatenate(labels)

    # Fixed stratified split: the source gives no event identity or time.
    split = np.empty(len(label), dtype=np.int64)
    for k in range(len(LLAIMA_CLASSES)):
        members = np.flatnonzero(label == k)
        split[members] = _splits(np.arange(len(members)), seed=k)
    provenance = {
        "source": f"doi:{LLAIMA_DOI}",
        "url": LLAIMA_ZIP_URL,
        "archive_mb": round(_megabytes(archive), 1),
        "archive_sha256": _sha256(archive),
        "prepared": time.strftime("%Y-%m-%d"),
    }
    _save_cache(cache, x=x, label=label, split=split, provenance=json.dumps(provenance))
    shutil.rmtree(work)
    return cache


def load_llaima(cache_dir) -> LlaimaData:
    import torch

    with np.load(llaima_cache_path(cache_dir)) as file:
        u = torch.from_numpy(file["x"].astype(np.float32)).unsqueeze(1)
        label = torch.from_numpy(file["label"])
        split = torch.from_numpy(file["split"])
        provenance = json.loads(str(file["provenance"]))
    return LlaimaData(u, label, split, LLAIMA_CLASSES, LLAIMA_RATE, provenance)


# ---------------------------------------------------------------------------- CEED


@dataclass(frozen=True)
class PickerData:
    """Three-component records with analyst P and S sample indices.

    ``u`` has shape ``(N, 3, CEED_WINDOW)``; ``p`` and ``s`` are sample indices into
    that stored window. ``position_eval`` fixes one crop position per trace for
    evaluation.
    """

    u: object
    p: object
    s: object
    split: object
    position_eval: object
    event_id: np.ndarray
    station: np.ndarray
    distance_km: np.ndarray
    snr: np.ndarray
    magnitude: np.ndarray
    event_lonlat: np.ndarray
    station_lonlat: np.ndarray
    provenance: dict

    def indices(self, split: str):
        return (self.split == SPLITS.index(split)).nonzero().squeeze(1)

    def examples(self, split: str = "train", quantiles=(0.1, 0.5, 0.9)):
        """Traces at the given quantiles of epicentral distance (near ... far)."""
        idx = self.indices(split)
        order = idx[np.argsort(self.distance_km[idx.numpy()])]
        return order[[int(q * (len(order) - 1)) for q in quantiles]]

    def crop(self, idx, length: int, position, margin: int = 200):
        """Cut one ``length``-sample window per trace, containing both picks.

        ``position`` in [0, 1) chooses the start inside the feasible range, so training
        can draw it at random and evaluation can reuse ``position_eval``.
        Returns ``(u, p, s)`` with picks relative to the crop.
        """
        import torch

        p, s = self.p[idx], self.s[idx]
        lowest = (s + margin - length).clamp(min=0)
        highest = torch.minimum(p - margin, torch.tensor(CEED_WINDOW - length))
        position = torch.as_tensor(position, dtype=torch.float64)
        start = (lowest + (position * (highest - lowest + 1)).long()).clamp(max=highest)
        steps = torch.arange(length)
        columns = start[:, None] + steps[None, :]
        u = torch.gather(self.u[idx], 2, columns[:, None, :].expand(-1, 3, -1))
        return u, p - start, s - start


def ceed_cache_path(cache_dir, year: int = 2007) -> Path:
    return Path(cache_dir) / f"ceed_sc_{year}.npz"


def _highpass(x: np.ndarray, rate: float, low: float = 1.0) -> np.ndarray:
    """Zero-phase FFT high-pass with a raised-cosine ramp between low/2 and low."""
    spectrum = np.fft.rfft(x, axis=-1)
    freq = np.fft.rfftfreq(x.shape[-1], 1.0 / rate)
    ramp = np.clip((freq - low / 2) / (low / 2), 0.0, 1.0)
    taper = 0.5 - 0.5 * np.cos(np.pi * ramp)
    return np.fft.irfft(spectrum * taper, n=x.shape[-1], axis=-1)


def prepare_ceed(cache_dir, year: int = 2007, progress=None) -> Path:
    """Download one CEED year file, keep clean three-component picks, cache them.
    ``progress(fraction)`` reports the download."""
    import h5py

    cache = ceed_cache_path(cache_dir, year)
    if cache.exists():
        return cache
    work = Path(cache_dir) / "ceed_download"
    url = CEED_URL.format(repo=CEED_REPOSITORY, revision=CEED_REVISION, year=year)
    source = work / f"{year}.h5"
    if not source.exists():  # a finished download survives an interrupted prepare
        _download(url, source, progress)

    rows = []
    with h5py.File(source) as file:
        for event_id, event in file.items():
            for name, trace in event.items():
                a = trace.attrs
                p, s = int(a["p_phase_index"]), int(a["s_phase_index"])
                good = (
                    str(a["component"]) == "ENZ"
                    and 0 < s - p <= CEED_MAX_S_MINUS_P
                    and float(a["p_phase_score"]) > 0
                    and float(a["s_phase_score"]) > 0
                    and p >= CEED_P_INDEX
                    and p + CEED_WINDOW - CEED_P_INDEX <= trace.shape[1]
                )
                if not good:
                    continue
                full = trace[:].astype(np.float64)
                full = _highpass(full - full.mean(axis=1, keepdims=True), CEED_RATE)
                window = full[:, p - CEED_P_INDEX : p - CEED_P_INDEX + CEED_WINDOW]
                window = window / (window.std() + 1e-12)
                rows.append(
                    (
                        event_id,
                        name.split(".")[1],
                        window.astype(np.float16),
                        CEED_P_INDEX,
                        CEED_P_INDEX + s - p,
                        float(a["distance_km"]),
                        float(np.mean(a["snr"])),
                        float(event.attrs["magnitude"]),
                        (
                            float(event.attrs["longitude"]),
                            float(event.attrs["latitude"]),
                        ),
                        (float(a["longitude"]), float(a["latitude"])),
                    )
                )
    events = np.array([r[0] for r in rows])
    n = len(rows)
    provenance = {
        "source": f"huggingface.co/datasets/{CEED_REPOSITORY}",
        "revision": CEED_REVISION,
        "year": year,
        "download_mb": round(_megabytes(source), 1),
        "kept_traces": n,
        "prepared": time.strftime("%Y-%m-%d"),
    }
    _save_cache(
        cache,
        x=np.stack([r[2] for r in rows]),
        p=np.array([r[3] for r in rows]),
        s=np.array([r[4] for r in rows]),
        split=_splits(events, seed=0),
        u_eval=np.random.default_rng(1).random(n),
        event_id=events,
        station=np.array([r[1] for r in rows]),
        distance_km=np.array([r[5] for r in rows]),
        snr=np.array([r[6] for r in rows]),
        magnitude=np.array([r[7] for r in rows]),
        event_lonlat=np.array([r[8] for r in rows]),
        station_lonlat=np.array([r[9] for r in rows]),
        provenance=json.dumps(provenance),
    )
    shutil.rmtree(work)
    return cache


def load_ceed(cache_dir, year: int = 2007) -> PickerData:
    import torch

    with np.load(ceed_cache_path(cache_dir, year)) as file:
        return PickerData(
            u=torch.from_numpy(file["x"].astype(np.float32)),
            p=torch.from_numpy(file["p"]),
            s=torch.from_numpy(file["s"]),
            split=torch.from_numpy(file["split"]),
            position_eval=torch.from_numpy(file["u_eval"]),
            event_id=file["event_id"],
            station=file["station"],
            distance_km=file["distance_km"],
            snr=file["snr"],
            magnitude=file["magnitude"],
            event_lonlat=file["event_lonlat"],
            station_lonlat=file["station_lonlat"],
            provenance=json.loads(str(file["provenance"])),
        )


def make_pick_targets(p, s, length: int, width: float):
    """Soft per-sample targets ``(B, 3, length)`` for the classes P, S and neither.

    P and S are Gaussians of standard deviation ``width`` samples centred on the
    analyst pick; "neither" takes the remaining probability, so the three channels
    sum to one at every sample.
    """
    import torch

    t = torch.arange(length, dtype=torch.float32)

    def bump(centre):
        return torch.exp(-0.5 * ((t[None, :] - centre[:, None]) / width) ** 2)

    target_p, target_s = bump(p.float()), bump(s.float())
    neither = (1.0 - target_p - target_s).clamp(min=0.0)
    target = torch.stack([target_p, target_s, neither], dim=1)
    return target / target.sum(dim=1, keepdim=True)
