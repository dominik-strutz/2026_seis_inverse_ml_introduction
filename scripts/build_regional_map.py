# /// script
# requires-python = ">=3.11"
# dependencies = ["pyshp>=2.3"]
# ///
"""Build the coastline and fault lines under the Part 2d dataset map.

Writes ``src/ml_seis/artifacts/regional_map.json``, a few tens of kB that ship with the
package (the map must work offline). Run it once; it needs the network.

* coastline: Natural Earth 10 m coastline (public domain, naturalearthdata.com), read
  from cartopy's cache if it is there, otherwise downloaded;
* faults: USGS Quaternary Fault and Fold Database, California layer (U.S. Geological
  Survey, public domain), only the faults with a slip rate above 1 mm/yr, so that the
  map shows the plate boundary and not every small fault.

Both are clipped to the region of the CEED subset and simplified to about 300 m.

    uv run scripts/build_regional_map.py
"""

import io
import json
import subprocess
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import shapefile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "ml_seis" / "artifacts" / "regional_map.json"
BBOX = (-123.3, 30.4, -114.0, 39.2)  # lon min, lat min, lon max, lat max
TOLERANCE = 0.003  # degrees, Douglas-Peucker
COAST_URL = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip"
CARTOPY = (
    Path.home()
    / ".local/share/cartopy/shapefiles/natural_earth/physical/ne_10m_coastline.shp"
)
FAULTS_URL = (
    "https://earthquake.usgs.gov/arcgis/rest/services/haz/Qfaults/MapServer/4/query"
)
# slip-rate classes of the database, by the start of their text, and the rate in mm/yr
RATES = {"Greater than 5.0": 5, "Between 1.0": 1}


def inside(p):
    return BBOX[0] <= p[0] <= BBOX[2] and BBOX[1] <= p[1] <= BBOX[3]


def clip(points):
    """The runs of consecutive points that lie inside the region."""
    runs, run = [], []
    for p in points:
        if inside(p):
            run.append(p)
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    return [r for r in runs if len(r) > 1]


def simplify(points, tolerance=TOLERANCE):
    """Douglas-Peucker."""
    if len(points) < 3:
        return points
    if points[0] == points[-1]:
        # a closed ring (an island) has no chord to measure against, and would shrink
        # to a point: split it at the point farthest from the start instead
        x0, y0 = points[0]
        far = max(
            range(1, len(points) - 1),
            key=lambda i: (points[i][0] - x0) ** 2 + (points[i][1] - y0) ** 2,
        )
        return simplify(points[: far + 1], tolerance)[:-1] + simplify(
            points[far:], tolerance
        )
    (x1, y1), (x2, y2) = points[0], points[-1]
    dx, dy = x2 - x1, y2 - y1
    norm = (dx * dx + dy * dy) ** 0.5 or 1e-12
    far, index = 0.0, 0
    for i, (x, y) in enumerate(points[1:-1], 1):
        d = abs(dy * (x - x1) - dx * (y - y1)) / norm
        if d > far:
            far, index = d, i
    if far <= tolerance:
        return [points[0], points[-1]]
    return simplify(points[: index + 1], tolerance)[:-1] + simplify(
        points[index:], tolerance
    )


def encode(points):
    """[x0, y0, dx1, dy1, ...] in units of 0.001 degree: the first point, then steps."""
    ints = [(round(x * 1000), round(y * 1000)) for x, y in points]
    flat = list(ints[0])
    for (x0, y0), (x1, y1) in zip(ints[:-1], ints[1:], strict=True):
        flat += [x1 - x0, y1 - y0]
    return flat


def coastline():
    if CARTOPY.exists():
        reader = shapefile.Reader(str(CARTOPY))
    else:
        request = urllib.request.Request(
            COAST_URL, headers={"User-Agent": "ml-seis-map-build"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        reader = shapefile.Reader(
            shp=io.BytesIO(archive.read("ne_10m_coastline.shp")),
            shx=io.BytesIO(archive.read("ne_10m_coastline.shx")),
            dbf=io.BytesIO(archive.read("ne_10m_coastline.dbf")),
        )
    lines = []
    for shape in reader.shapes():
        bounds = shape.bbox
        if bounds[2] < BBOX[0] or bounds[0] > BBOX[2]:
            continue
        if bounds[3] < BBOX[1] or bounds[1] > BBOX[3]:
            continue
        starts = list(shape.parts) + [len(shape.points)]
        for a, b in zip(starts[:-1], starts[1:], strict=True):
            for run in clip(shape.points[a:b]):
                lines.append(encode(simplify(run)))
    return lines


def fetch_faults(prefix):
    """The faults whose slip-rate class starts with ``prefix``, page by page. (The
    server refuses a WHERE clause with IN (...) or with the word "and".)"""
    features, offset = [], 0
    while True:
        query = {
            "f": "geojson",
            "where": f"slip_rate LIKE '{prefix}%'",
            "geometry": ",".join(str(v) for v in BBOX),
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "fault_name,slip_rate",
            "outSR": "4326",
            "maxAllowableOffset": "0.002",
            "geometryPrecision": "4",
            "returnGeometry": "true",
            "resultOffset": str(offset),
            "resultRecordCount": "1000",
        }
        url = (
            FAULTS_URL
            + "?"
            + urllib.parse.urlencode(query, quote_via=urllib.parse.quote)
        )
        # curl, not urllib: the USGS server answers curl and refuses Python's urllib
        done = subprocess.run(
            ["curl", "-fsS", "--max-time", "120", "-A", "ml-seis-map-build", url],
            capture_output=True,
            check=True,
        )
        page = json.loads(done.stdout)
        features += page["features"]
        if not page.get("exceededTransferLimit"):
            return features
        offset += len(page["features"])


def chain(lines):
    """Join segments that share an end point into longer lines (the database cuts a
    fault into thousands of short sections)."""
    ends, out = {}, []
    for line in lines:
        line = [tuple(p) for p in line]
        while True:
            hit = None
            for key in (line[0], line[-1]):
                j = ends.get(key)
                if j is not None and out[j] is not None:
                    hit = (key, j)
                    break
            if hit is None:
                break
            key, j = hit
            other, out[j] = out[j], None
            for end in (other[0], other[-1]):
                ends.pop(end, None)
            if key == line[0]:
                line = (other if other[-1] == key else other[::-1]) + line[1:]
            else:
                line = line + (other if other[0] == key else other[::-1])[1:]
        out.append(line)
        ends[line[0]] = ends[line[-1]] = len(out) - 1
    return [list(line) for line in out if line is not None]


def faults():
    out = {rate: [] for rate in RATES.values()}
    for prefix, rate in RATES.items():
        pieces = []
        for feature in fetch_faults(prefix):
            geometry = feature["geometry"]
            parts = (
                geometry["coordinates"]
                if geometry["type"] == "MultiLineString"
                else [geometry["coordinates"]]
            )
            for part in parts:
                for run in clip([(round(x, 3), round(y, 3)) for x, y, *_ in part]):
                    pieces.append(run)
        for line in chain(pieces):
            line = simplify(line, 0.006)
            if len(line) > 1:
                out[rate].append(encode(line))
    return out


if __name__ == "__main__":
    data = {
        "source": (
            "Coastline: Natural Earth 10 m (public domain). Faults: USGS Quaternary "
            "Fault and Fold Database, California, slip rate above 1 mm/yr "
            "(public domain)."
        ),
        "bbox": list(BBOX),
        "encoding": "each line is [x0, y0, dx1, dy1, ...] in units of 0.001 degree",
        "coast": coastline(),
        "faults": {str(rate): lines for rate, lines in faults().items()},
    }
    OUT.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(
        f"{OUT.relative_to(ROOT)}: {len(data['coast'])} coast lines, "
        f"{sum(len(v) for v in data['faults'].values())} fault lines, "
        f"{OUT.stat().st_size / 1e3:.0f} kB"
    )
