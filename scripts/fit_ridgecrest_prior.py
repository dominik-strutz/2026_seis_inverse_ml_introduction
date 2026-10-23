# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Build the ``ridgecrest`` prior of ``ml_seis.priors`` from the 2019 sequence.

Downloads the USGS ComCat catalogue (M >= 1.5, 4 Jul - 4 Aug 2019), projects it to the
local map frame (km, x east, y north, sequence centred in the 100 km box) and counts
the events on a coarse grid over the usable box. The prior is a Gaussian-kernel
mixture: one kernel of sd ``BANDWIDTH`` per cell, weighted by the cell's share of the
events, plus a weak uniform background. Saves the cell weights and both constants to
``src/ml_seis/ridgecrest_prior.npz``.

    uv run scripts/fit_ridgecrest_prior.py
"""

import http.client
import io
import urllib.request
from pathlib import Path

import numpy as np

URL = (
    "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv"
    "&starttime=2019-07-04&endtime=2019-08-04&minmagnitude=1.5"
    "&minlatitude=35.2&maxlatitude=36.3&minlongitude=-118.2&maxlongitude=-116.9"
)
OUT = Path(__file__).parents[1] / "src" / "ml_seis" / "ridgecrest_prior.npz"
BOX, MARGIN = 100.0, 5.0  # as in ml_seis.geometry / ml_seis.priors
LAT0, LON0 = 35.80, -117.64  # map centre (50, 50) km
CELL = 2.0  # km, grid cell: one kernel per cell
BANDWIDTH = 2.0  # km, sd of each Gaussian kernel (location error, fault-zone width)
BACKGROUND = 0.05  # probability mass of the uniform component

for attempt in range(5):  # the USGS endpoint sometimes drops long responses
    try:
        raw = urllib.request.urlopen(URL, timeout=120).read().decode()
        break
    except (OSError, http.client.IncompleteRead):
        if attempt == 4:
            raise
table = np.genfromtxt(
    io.StringIO(raw), delimiter=",", names=True, usecols=(1, 2), dtype=float
)
x = 50 + (table["longitude"] - LON0) * 111.32 * np.cos(np.radians(LAT0))
y = 50 + (table["latitude"] - LAT0) * 110.57

lo, hi = MARGIN, BOX - MARGIN
edges = np.arange(lo, hi + CELL / 2, CELL)
counts, _, _ = np.histogram2d(y, x, bins=[edges, edges])  # [iy, ix]
weights = counts / counts.sum()
np.savez(
    OUT,
    weights=weights.astype(np.float32),
    bandwidth=BANDWIDTH,
    background=BACKGROUND,
)
print(f"{int(counts.sum())} events in the box, grid {weights.shape}, saved to {OUT}")
