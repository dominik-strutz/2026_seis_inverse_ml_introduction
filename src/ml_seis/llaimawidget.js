// The Part 2 Llaima explorer (see llaimawidget.py); controls.js is put in front of
// this module (tokens C, FONT and the ui* controls).
//
// One row per class, three records each: the class balance bars on the left line up
// with the rows, and each page turn shows the next three records of every class (a
// class that has run out shows empty frames). The actual samples of the records on
// this page and the next two are fetched from Python as they are needed, and a new
// page is shown only once all its records have arrived (the old one stays meanwhile). Below, one panel for the open
// record: its amplitude spectrum against the class spectra, its spectrogram, or its
// waveform at full resolution (drag to zoom, click to reset). Arrow keys turn
// the page.

const W = 1200, H = 672;
const SPLITS = ["train", "validation", "test"];
const SHADE = [1.0, 0.6, 0.32];  // opacity of the train, validation, test part of a bar
const GRID = { x: 252, y: 44, cols: 3, w: 309, h: 78, gap: 10, strip: 14 };
const PITCH = GRID.h + GRID.gap;  // one row per class
const BAR = { x: 64, w: 120, h: 22 };
const LEFT = GRID.x, RIGHT = GRID.x + GRID.cols * GRID.w + (GRID.cols - 1) * GRID.gap;
const PANEL = { t: 438, b: 618 };  // the one panel below the grid
const FACTS = { head: 418 };
const VIEWS = [["spectrum", "spectrum"], ["spectrogram", "spectrogram"],
  ["waveform", "waveform"]];
const LOCAL_CSS = `
.llx-pill { font: 12px ${FONT}; padding: 3px 12px; border: 1px solid ${C.grid};
  border-radius: 999px; background: ${C.bg}; color: ${C.ink}; cursor: pointer; }
.llx-pill:hover { border-color: ${C.ink2}; }
.llx-row { display: flex; align-items: center; gap: 8px; }
.llx-page { font-size: 15px; min-width: 6.5ch; text-align: center;
  font-variant-numeric: tabular-nums; }
`;

// a pseudo-random but reproducible number in [0, 1) for record i and a seed
function hash01(i, seed) {
  let h = (Math.imul(i + 1, 0x9e3779b1) ^ Math.imul(seed + 7, 0x85ebca6b)) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d) >>> 0;
  h = Math.imul(h ^ (h >>> 12), 0x297a2d39) >>> 0;
  return ((h ^ (h >>> 15)) >>> 0) / 4294967296;
}

function render({ model, el }) {
  const st = makeStage(el, W);
  const local = elt("style");
  local.textContent = LOCAL_CSS;
  st.stage.append(local);

  // ---- the data set ------------------------------------------------------------
  const classes = model.get("classes"), names = model.get("names");
  const colours = model.get("colours"), label = model.get("label");
  const split = model.get("split"), features = model.get("features");
  const duration = model.get("duration");
  const spectra = model.get("class_spectra");
  const N = label.length, K = classes.length;
  const total = Array.from({ length: K }, () => [0, 0, 0]);
  for (let i = 0; i < N; i++) total[label[i]][split[i]] += 1;
  const maxCount = Math.max(...total.map((t) => t[0] + t[1] + t[2]));
  const FMAX = 20;  // Hz; the spectra and the spectrogram stop here

  // ---- state -------------------------------------------------------------------
  const splitOn = [true, true, true];
  let view = "spectrum";  // what the panel below shows
  let seed = 1, page = 0;
  let order = [];  // per class, the records that pass the split filter
  let zoom = [0, duration];
  let hover = null, brush = null;
  let hits = [];

  const nPages = () => Math.max(1, Math.ceil(
    Math.max(...order.map((o) => o.length)) / GRID.cols));
  const selected = () => model.get("selected");
  const shownOnPage = () => order.flatMap((o) => o.slice(page * GRID.cols,
    (page + 1) * GRID.cols));

  function computeOrder() {
    order = classes.map(() => []);
    for (let i = 0; i < N; i++) if (splitOn[split[i]]) order[label[i]].push(i);
    for (const o of order) o.sort((a, b) => hash01(a, seed) - hash01(b, seed) || a - b);
    page = Math.min(page, nPages() - 1);
  }

  // ---- header bar --------------------------------------------------------------
  const bar = elt("div", "mls-bar");
  const shuffle = elt("button", "llx-pill", "shuffle");
  shuffle.onclick = () => { seed += 1; page = 0; computeOrder(); draw(); };
  const prev = elt("button", "mls-pm", "‹"), next = elt("button", "mls-pm", "›");
  prev.title = "previous page"; next.title = "next page";
  prev.onclick = () => gotoPage(page - 1);
  next.onclick = () => gotoPage(page + 1);
  const pageText = elt("div", "llx-page");
  const pager = elt("div", "llx-row");
  pager.append(prev, pageText, next);
  const viewSeg = uiSegmented(VIEWS, view, (v) => { view = v; draw(); });
  const status = elt("div", "mls-status");
  bar.append(
    uiField(uiLabel("PAGE"), pager), uiField(uiLabel("ORDER"), shuffle),
    elt("div", "mls-rule"), uiField(uiLabel("SHOW BELOW"), viewSeg), status);
  st.stage.append(bar);

  const fig = stageCanvas(st, W, H);
  fig.canvas.tabIndex = 0;
  fig.canvas.style.outline = "none";
  st.stage.append(fig.canvas);
  const ctx = fig.ctx;

  function gotoPage(p) {
    page = Math.min(Math.max(p, 0), nPages() - 1);
    draw();
  }
  function syncHeader() {
    pageText.textContent = `${page + 1} / ${nPages()}`;
    prev.disabled = page <= 0;
    next.disabled = page >= nPages() - 1;
    const n = order.reduce((sum, o) => sum + o.length, 0);
    status.textContent = `${n} of ${N} records`;
  }

  // ---- drawing helpers ---------------------------------------------------------
  const font = (px) => { ctx.font = `${px}px ${FONT}`; };
  const text = (s, x, y, colour = C.ink, align = "left", px = 11) => {
    font(px); ctx.fillStyle = colour; ctx.textAlign = align; ctx.textBaseline = "alphabetic";
    ctx.fillText(s, x, y);
  };
  const alpha = (hex, a) => {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${n >> 16}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
  };
  const niceTicks = (a, b) => {
    const raw = (b - a) / 6;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw);
    const out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(v);
    return out;
  };
  function openAxes(l, t, r, b, { x0, x1, yticks = [], ymap, xlabel }) {
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(l, t); ctx.lineTo(l, b); ctx.lineTo(r, b); ctx.stroke();
    for (const v of niceTicks(x0, x1)) {
      const x = l + (v - x0) / (x1 - x0) * (r - l);
      ctx.beginPath(); ctx.moveTo(x, b); ctx.lineTo(x, b - 4); ctx.stroke();
      // the last label would be cut off at the edge of the figure
      text(String(+v.toFixed(2)), x, b + 13, C.ink2, x > r - 8 ? "right" : "center", 10);
    }
    for (const v of yticks) {
      const y = ymap(v);
      ctx.beginPath(); ctx.moveTo(l, y); ctx.lineTo(l + 4, y); ctx.stroke();
      text(String(v), l - 6, y + 3.5, C.ink2, "right", 10);
    }
    text(xlabel, (l + r) / 2, b + 28, C.ink2, "center", 10.5);
  }

  // a long description on two short lines: "volcano-" / "tectonic"
  const twoLines = (name) => {
    const cut = name.search(/[- ]/);
    return cut < 0 ? [name] : [name.slice(0, cut + 1).trim(), name.slice(cut + 1)];
  };

  // ---- the class balance: one bar per row of the grid ----------------------------
  function drawBalance() {
    text("CLASS BALANCE", 0, 12, C.ink2, "left", 10.5);
    SPLITS.forEach((name, s) => {  // split chips, also the legend of the shading
      const x = [0, 62, 150][s], y = 22, on = splitOn[s];
      ctx.fillStyle = alpha(C.ink, on ? SHADE[s] : 0.07);
      ctx.fillRect(x, y + 3, 12, 12);
      text(name, x + 18, y + 13, on ? C.ink : C.ink2, "left", 11);
      hits.push({ x, y, w: s === 1 ? 84 : 56, h: 18, type: "split", i: s });
    });
    classes.forEach((c, k) => {
      const y = GRID.y + k * PITCH, mid = y + GRID.h / 2;
      text(c, 0, mid - 8, colours[k], "left", 13);
      twoLines(names[k]).forEach((line, j) => text(line, 0, mid + 6 + 12 * j, C.ink2,
        "left", 10));
      let x = BAR.x, shown = 0;
      for (let s = 0; s < 3; s++) {
        if (!splitOn[s]) continue;
        const w = BAR.w * total[k][s] / maxCount;
        ctx.fillStyle = alpha(colours[k], SHADE[s]);
        ctx.fillRect(x, mid - BAR.h / 2, w, BAR.h);
        x += w; shown += total[k][s];
      }
      text(String(shown), x + 6, mid + 4, C.ink, "left", 11);
    });
  }

  // ---- the grid of records -------------------------------------------------------
  const tileXY = (k, c) => [GRID.x + c * (GRID.w + GRID.gap), GRID.y + k * PITCH];
  // full-resolution records for the grid, asked from Python a page ahead
  const records = new Map();  // record -> Int16Array, within [-32767, 32767]
  const asked = new Set();
  let nonce = 0;
  function wantRecords() {
    const ahead = [];
    for (const p of [page, page + 1, page + 2]) {
      for (const o of order) ahead.push(...o.slice(p * GRID.cols, (p + 1) * GRID.cols));
    }
    const missing = ahead.filter((i) => !records.has(i) && !asked.has(i));
    if (!missing.length) return;
    missing.forEach((i) => asked.add(i));
    // a changing negative entry, ignored by Python, so that a repeated request still counts as a change
    model.set("want", [...missing, -(++nonce)]);
    model.save_changes();
  }
  function takeChunk() {
    const { indices, data } = model.get("chunk");
    if (!indices) return;
    const raw = atob(data), n = raw.length / 2 / indices.length;
    const bytes = new Uint8Array(raw.length);
    for (let j = 0; j < raw.length; j++) bytes[j] = raw.charCodeAt(j);
    const all = new Int16Array(bytes.buffer);
    indices.forEach((i, j) => records.set(i, all.subarray(j * n, (j + 1) * n)));
    draw();
  }

  function drawWiggles(i, x, y, w, h, colour) {
    const mid = y + h / 2;
    ctx.strokeStyle = colour; ctx.lineWidth = 0.8; ctx.beginPath();
    const v = records.get(i);
    if (!v) {  // not here yet: a flat line until the samples arrive
      ctx.strokeStyle = C.grid; ctx.moveTo(x, mid); ctx.lineTo(x + w, mid); ctx.stroke();
      return;
    }
    const sc = h / 2 * 0.92 / 32767;
    for (let j = 0; j < v.length; j++) {
      const px = x + j / (v.length - 1) * w, py = mid - v[j] * sc;
      j ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    }
    ctx.stroke();
  }
  let shown = null;  // per class, the records on screen: the last page that was complete
  function drawGrid() {
    wantRecords();
    const target = order.map((o) => o.slice(page * GRID.cols, (page + 1) * GRID.cols));
    // the first page is drawn at once (flat lines until its samples arrive); later pages
    // replace the old one only when complete
    if (!shown || target.every((t) => t.every((i) => records.has(i)))) shown = target;
    classes.forEach((name, k) => {
      for (let c = 0; c < GRID.cols; c++) {
        const [x, y] = tileXY(k, c), i = shown ? shown[k][c] : undefined;
        if (i === undefined) {  // this class has run out: an empty frame
          ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
          ctx.strokeRect(x + 0.5, y + 0.5, GRID.w - 1, GRID.h - 1);
          continue;
        }
        const on = i === selected();
        ctx.strokeStyle = on ? C.ink : hover === i ? C.ink2 : C.grid;
        ctx.lineWidth = on ? 2 : 1;
        ctx.strokeRect(x + 0.5, y + 0.5, GRID.w - 1, GRID.h - 1);
        text(name, x + 6, y + 11, colours[k], "left", 11);
        text(`#${i}`, x + 30, y + 11, C.ink2, "left", 10);
        text(`${Math.round(100 * features.flat[i])}% flat, ${SPLITS[split[i]]}`,
          x + GRID.w - 6, y + 11, C.ink2, "right", 10);
        const [bx, by, bw, bh] = [x + 2, y + GRID.strip, GRID.w - 4, GRID.h - GRID.strip - 2];
        drawWiggles(i, bx, by, bw, bh, colours[k]);
        hits.push({ x, y, w: GRID.w, h: GRID.h, type: "record", i });
      }
    });
  }

  // ---- the panel below -------------------------------------------------------------
  const specImage = new Image();
  let specSrc = "";
  specImage.onload = () => draw();
  const timeToX = (t) => LEFT + (t - zoom[0]) / (zoom[1] - zoom[0]) * (RIGHT - LEFT);
  const xToTime = (px) => zoom[0] + (px - LEFT) / (RIGHT - LEFT) * (zoom[1] - zoom[0]);
  const ready = (detail, k) => detail && k >= 0 && detail.index === k;

  function title(s) { text(s, LEFT, PANEL.t - 14, C.ink2, "left", 11); }
  function zoomHint() {
    const zoomed = zoom[0] > 0 || zoom[1] < duration;
    text(zoomed ? "click to reset the zoom" : "drag to zoom", RIGHT,
      PANEL.t - 14, C.ink2, "right", 10);
  }
  function needRecord(k) {
    if (k >= 0) return false;
    text("click a record above to open it", (LEFT + RIGHT) / 2, (PANEL.t + PANEL.b) / 2,
      C.ink2, "center", 12);
    return true;
  }

  function drawSpectrum(k, detail) {
    const { t, b } = PANEL, l = LEFT, r = RIGHT;
    const focus = k >= 0 ? label[k] : -1;  // the class of the open record
    const shown = (f) => f <= FMAX;
    // the scale fits what is drawn, so a low class mean is not squeezed under a tall axis
    const peaks = spectra.mean.map((m) => Math.max(...m.filter((_, j) => shown(spectra.freq[j]))));
    if (focus >= 0) {
      peaks.push(Math.max(...spectra.high[focus].filter((_, j) => shown(spectra.freq[j]))));
    }
    if (ready(detail, k)) {
      peaks.push(Math.max(...detail.spectrum.filter((_, j) => shown(detail.freq[j]))));
    }
    const top = 1.08 * Math.max(...peaks);
    const ymap = (v) => b - v * (b - t) / top;
    title(k >= 0 ? "amplitude spectrum, scaled to its maximum"
      : "class mean amplitude spectrum, each record scaled to its maximum");
    openAxes(l, t, r, b, { x0: 0, x1: FMAX, xlabel: "frequency (Hz)" });
    const px = (f) => l + f / FMAX * (r - l);
    ctx.save();
    ctx.beginPath(); ctx.rect(l, t, r - l, b - t); ctx.clip();
    const line = (xs, ys, colour, lw) => {
      ctx.beginPath();
      xs.forEach((f, j) => (j ? ctx.lineTo(px(f), ymap(ys[j])) : ctx.moveTo(px(f), ymap(ys[j]))));
      ctx.strokeStyle = colour; ctx.lineWidth = lw; ctx.stroke();
    };
    if (focus >= 0) {  // the band holding the middle 80% of the record's class
      ctx.beginPath();
      spectra.freq.forEach((f, j) => {
        const y = ymap(spectra.high[focus][j]);
        j ? ctx.lineTo(px(f), y) : ctx.moveTo(px(f), y);
      });
      for (let j = spectra.freq.length - 1; j >= 0; j--) {
        ctx.lineTo(px(spectra.freq[j]), ymap(spectra.low[focus][j]));
      }
      ctx.closePath(); ctx.fillStyle = alpha(colours[focus], 0.18); ctx.fill();
    }
    classes.forEach((c, j) => {  // the other classes stay as a faint reference
      const mine = focus < 0 || focus === j;
      line(spectra.freq, spectra.mean[j], alpha(colours[j], mine ? 1 : 0.35),
        mine ? 1.6 : 1.0);
    });
    if (ready(detail, k)) line(detail.freq, detail.spectrum, C.ink, 0.8);
    ctx.restore();
    let y = t + 4;
    classes.forEach((c, j) => {
      if (focus >= 0 && focus !== j) return;
      text(`${c} mean`, r - 4, y + 8, colours[j], "right", 10.5);
      y += 14;
    });
    if (k >= 0) text(`#${k}`, r - 4, y + 8, C.ink, "right", 10.5);
  }

  function drawSpectrogram(k, detail) {
    const { t, b } = PANEL, l = LEFT, r = RIGHT;
    const ymap = (v) => b - v / FMAX * (b - t);
    title("spectrogram, dB below its maximum");
    zoomHint();
    openAxes(l, t, r, b, { x0: zoom[0], x1: zoom[1], yticks: niceTicks(0, FMAX), ymap,
      xlabel: "time (s)" });
    text("Hz", l - 6, t - 14, C.ink2, "right", 10);
    if (needRecord(k)) return;
    if (!ready(detail, k)) { text("loading...", r, t + 12, C.ink2, "right", 10); return; }
    const sp = detail.spectrogram;
    if (specSrc !== sp.png) { specSrc = sp.png; specImage.src = sp.png; return; }
    if (!specImage.complete) return;
    ctx.save();
    ctx.beginPath(); ctx.rect(l, t, r - l, b - t); ctx.clip();
    const x0 = timeToX(sp.t0), x1 = timeToX(sp.t1);
    const y1 = ymap(sp.f1), y0 = ymap(sp.f0);
    ctx.imageSmoothingEnabled = false;  // one block per time-frequency cell
    ctx.drawImage(specImage, x0, y1, x1 - x0, y0 - y1);
    ctx.restore();
  }

  function drawWaveform(k, detail) {
    const { t, b } = PANEL, l = LEFT, r = RIGHT;
    title("waveform");
    zoomHint();
    // the amplitude scale fits the part of the record in view
    let peak = 0;
    if (k >= 0 && ready(detail, k)) {
      const n = detail.x.length, dt = duration / n;
      const i0 = Math.max(Math.floor(zoom[0] / dt), 0), i1 = Math.min(Math.ceil(zoom[1] / dt), n - 1);
      for (let i = i0; i <= i1; i++) peak = Math.max(peak, Math.abs(detail.x[i]));
    }
    const unit = 10 ** Math.floor(Math.log10(Math.max(peak, 0.01)));
    peak = +(Math.ceil(Math.max(peak, 0.01) / unit) * unit).toPrecision(2);  // 1 digit up
    const ymap = (v) => (t + b) / 2 - v * (b - t) / 2 / (1.1 * peak);
    openAxes(l, t, r, b, { x0: zoom[0], x1: zoom[1], yticks: [-peak, 0, peak], ymap,
      xlabel: "time (s)" });
    if (needRecord(k)) return;
    const colour = colours[label[k]];
    ctx.save();
    ctx.beginPath(); ctx.rect(l, t, r - l, b - t); ctx.clip();
    if (ready(detail, k)) {
      const x = detail.x, dt = duration / x.length;
      const i0 = Math.max(Math.floor(zoom[0] / dt) - 1, 0);
      const i1 = Math.min(Math.ceil(zoom[1] / dt) + 1, x.length - 1);
      ctx.beginPath();
      for (let i = i0; i <= i1; i++) {
        const px = timeToX((i + 0.5) * dt), py = ymap(x[i]);
        i === i0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.strokeStyle = colour; ctx.lineWidth = 0.9; ctx.stroke();
    } else text("loading the record...", r - 4, t + 12, C.ink2, "right", 10);
    ctx.restore();
  }

  function drawFacts(k) {
    text("RECORD", 0, FACTS.head, C.ink2, "left", 10.5);
    if (k < 0) return;
    const rows = [
      [classes[label[k]], colours[label[k]], 14],
      [names[label[k]], C.ink2, 11],
      [`#${k}, ${SPLITS[split[k]]} set`, C.ink, 11],
      [`flat share  ${(100 * features.flat[k]).toFixed(0)}%`, C.ink, 11],
      [`rms  ${features.rms[k].toFixed(3)}`, C.ink, 11],
      [`dominant  ${features.f_dom[k].toFixed(1)} Hz`, C.ink, 11],
      [`mean  ${features.f_mean[k].toFixed(1)} Hz`, C.ink, 11],
    ];
    rows.forEach(([s, colour, px], j) =>
      text(s, 0, FACTS.head + 22 + j * 19, colour, "left", px));
  }

  // ---- draw ----------------------------------------------------------------------
  function draw() {
    syncHeader();
    ctx.clearRect(0, 0, W, H);
    hits = [];
    const k = selected(), detail = model.get("detail");
    drawBalance();
    drawGrid();
    drawFacts(k);
    text("left and right arrow keys turn the page", 0, PANEL.b + 28, C.ink2, "left", 10);
    if (view === "spectrum") drawSpectrum(k, detail);
    else if (view === "spectrogram") drawSpectrogram(k, detail);
    else drawWaveform(k, detail);
    if (brush) {
      const [a, c] = [Math.min(brush[0], brush[1]), Math.max(brush[0], brush[1])];
      ctx.fillStyle = alpha(C.ink, 0.1);
      ctx.fillRect(a, PANEL.t, c - a, PANEL.b - PANEL.t);
    }
  }

  // ---- pointer and keys -----------------------------------------------------------
  const hitAt = ([x, y]) => hits.find((h) => x >= h.x && x < h.x + h.w &&
    y >= h.y && y < h.y + h.h);
  const inPanel = ([x, y]) => view !== "spectrum" && x >= LEFT && x <= RIGHT &&
    y >= PANEL.t && y <= PANEL.b;
  function setSelected(i) {
    model.set("selected", i);
    model.save_changes();
    draw();
  }

  fig.canvas.addEventListener("click", (ev) => {
    const h = hitAt(fig.xy(ev));
    if (!h) return;
    if (h.type === "record") setSelected(h.i === selected() ? -1 : h.i);
    else if (h.type === "split") {
      splitOn[h.i] = !splitOn[h.i];
      if (!splitOn.some(Boolean)) splitOn.fill(true);  // never leave nothing to show
      page = 0; computeOrder(); draw();
    }
  });
  fig.canvas.addEventListener("dblclick", (ev) => {
    if (inPanel(fig.xy(ev))) { zoom = [0, duration]; draw(); }
  });
  fig.canvas.addEventListener("pointerdown", (ev) => {
    fig.canvas.focus();
    const p = fig.xy(ev);
    if (!inPanel(p) || selected() < 0) return;
    fig.canvas.setPointerCapture(ev.pointerId);
    brush = [p[0], p[0]];
  });
  fig.canvas.addEventListener("pointermove", (ev) => {
    const p = fig.xy(ev);
    if (brush) {
      brush[1] = Math.min(Math.max(p[0], LEFT), RIGHT);
      draw();
      return;
    }
    const h = hitAt(p);
    const over = h && h.type === "record" ? h.i : null;
    fig.canvas.style.cursor = h ? "pointer" : inPanel(p) ? "crosshair" : "default";
    if (over !== hover) { hover = over; draw(); }
  });
  const endBrush = () => {
    if (!brush) return;
    const [a, c] = [Math.min(brush[0], brush[1]), Math.max(brush[0], brush[1])];
    if (c - a > 4) zoom = [Math.max(xToTime(a), 0), Math.min(xToTime(c), duration)];
    else zoom = [0, duration];  // a click without a drag zooms out
    brush = null;
    draw();
  };
  fig.canvas.addEventListener("pointerup", endBrush);
  fig.canvas.addEventListener("pointercancel", endBrush);
  fig.canvas.addEventListener("keydown", (ev) => {
    const move = { ArrowLeft: () => gotoPage(page - 1), ArrowRight: () => gotoPage(page + 1),
      PageUp: () => gotoPage(page - 1), PageDown: () => gotoPage(page + 1),
      Escape: () => setSelected(-1) }[ev.key];
    if (!move) return;
    ev.preventDefault();
    move();
  });

  // ---- from Python ---------------------------------------------------------------
  model.on("change:chunk", takeChunk);
  model.on("change:detail", draw);
  model.on("change:selected", draw);

  computeOrder();
  draw();
  // the first request can be sent before Python listens: ask again until the page is here
  let tries = 0;
  const retry = setInterval(() => {
    const need = shownOnPage().filter((i) => !records.has(i));
    if (!need.length || ++tries > 20) { clearInterval(retry); return; }
    need.forEach((i) => asked.delete(i));
    wantRecords();
  }, 500);
}

export default { render };
