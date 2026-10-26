// The Lecture 1 MLP playground (see mlpwidget.py); controls.js is put in front
// of this module (tokens C, FONT, SYM and the ui* controls). The figure reads
// left to right like TensorFlow Playground: DATA (examples), the network (one
// tile per neuron showing its output over x, edges coloured by weight, +/- for
// depth and width), OUTPUT (train and validation loss and the fit, drawn on by the
// student). DATA holds two sets of points: train (filled blue; the network is
// fitted to it) and validation (hollow vermillion; only its loss is computed), as in
// the prior maps later in the lecture; the network's output is drawn in ink. A toggle picks the set a drag draws or
// erases; an example has as many points of each as the two number fields say,
// and once drawn on, the fields show the counts and are off. Under the graph: the same network as weight matrices, each block
// under the band of edges it holds; the parameter count; the equation of the
// selected neuron. The header plays the run back (a longer run plays a little
// longer, at most PLAY_MAX_MS): weights, tiles, fit and loss all move together.

// Layout (logical px). Neurons keep one size: a wide layer makes the graph,
// and with it the figure, taller (everything from the brackets down moves by
// grow()); the heights below are for a graph of up to MIN_SPAN.
const W = 1200, H = 646;
const ERASE = 18;  // eraser radius
const MAXW = 16, MAXL = 3;  // neurons per hidden layer, hidden layers
const IO = 34;  // input and output tiles
const TILE = 26, PITCH = 34;  // hidden tiles and the distance between their centres
const NET = { xIn: 205, xOut: 680, top: 112, minSpan: 300, bracket: 446 };
const OUTX = 730;  // left edge of the output column
const SPARK = { l: 985, t: 30, w: 170, h: 56 };
const PLOT = { l: 770, t: 124, w: 410, h: 296 };
const ROW2 = 478;  // second row: matrices, parameters, the selected neuron
const MAT = { top: 520, maxH: 96, maxCell: 12 };
const N_TRAIN = { min: 3, max: 200 }, N_VAL = { min: 0, max: 200 };  // per example

const PRESETS = {
  sine: (x) => 0.8 * Math.sin(3.2 * x),
  wiggle: (x) => 0.65 * Math.sin(9 * x),
  kink: (x) => 1.7 * Math.abs(x) - 0.85,
  step: (x) => (x < 0.1 ? -0.6 : 0.7),
};
const ACTS = {
  ReLU: (z) => Math.max(0, z),
  tanh: Math.tanh,
  sigmoid: (z) => 1 / (1 + Math.exp(-z)),
  linear: (z) => z,
};
// An example: n points of the curve f with a little fixed noise. ``which`` is 0
// for the train set and 1 for the validation set (other x, other noise). x runs on a
// golden-ratio sequence, so more points only add to the ones already there.
// mlpwidget.py (example_data) holds the same recipe for the opening drawing.
// ``rms`` is the noise level (the pattern sin * cos has rms 1/2, so its amplitude is
// 2 rms); ``seed`` > 0 shifts the pattern: a new draw of the noise.
const GOLDEN = 0.6180339887498949;
const NOISE_MAX = 0.5;
function sample(f, n, which, [x0, x1], rms = 0.1, seed = 0) {
  const pts = [];
  const shift = 1000 * seed;
  for (let i = 0; i < n; i++) {
    const x = x0 + (x1 - x0) * ((GOLDEN * i + 0.5 * which + 0.1) % 1);
    const noise = 2 * rms * Math.sin(97 * i + 13 * which + shift)
      * Math.cos(31 * i + 7 * which + shift);
    pts.push([+x.toFixed(4), +(f(x) + noise).toFixed(4)]);
  }
  return pts;
}
// A noise level typed in freely, clamped to [0, max] when committed.
function uiDecimal(value, max, onChange) {
  const e = elt("input", "mls-number");
  e.type = "text"; e.inputMode = "decimal"; e.value = String(value); e.size = 5;
  let last = value;
  const commit = () => {
    const v = Number(e.value);
    const ok = Number.isFinite(v) ? Math.round(Math.min(Math.max(v, 0), max) * 100) / 100 : last;
    e.value = String(ok);
    if (ok !== last) { last = ok; onChange(ok); }
  };
  e.addEventListener("change", commit);
  e.addEventListener("keydown", (ev) => { if (ev.key === "Enter") e.blur(); });
  return e;
}
const TRAIN_KEYS = ["points", "val_points", "widths", "activation", "lr", "epochs",
  "seed"];
const AUTO_MAX = 4;  // seconds: automatic retraining is skipped for slower runs
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const LRS = [0.001, 0.003, 0.01, 0.03, 0.1];
const EPOCHS = { min: 10, max: 5000 };  // typed in freely within these
// playback duration (ms) of a run of n epochs: 3 s + 2 ms per epoch, at most 10 s
const playMs = (n) => Math.min(3000 + 2 * n, 10000);

const LOCAL_CSS = `
.mlp-preset { width: 62px; height: 44px; padding: 0; border: 1px solid ${C.grid};
  border-radius: 4px; background: ${C.bg}; cursor: pointer; }
.mlp-preset:hover { border-color: ${C.ink}; }
.mlp-preset svg { width: 100%; height: 100%; display: block; }
.mlp-pill { font: 12px ${FONT}; padding: 3px 12px; border: 1px solid ${C.grid};
  border-radius: 999px; background: ${C.bg}; color: ${C.ink}; cursor: pointer; }
.mlp-pill:hover { border-color: ${C.ink2}; }
.mlp-note { font-size: 12px; line-height: 1.4; color: ${C.ink2}; width: 136px; }
.mlp-col { display: flex; flex-direction: column; align-items: center; gap: 6px; }
.mlp-count { font-size: 12px; color: ${C.ink2}; white-space: nowrap; }
.mlp-depth { display: flex; align-items: center; gap: 10px; }
.mlp-loss .mls-value { font-size: 22px; }
.mlp-train .mls-value { color: ${C.blue}; }
.mlp-val .mls-value { color: ${C.orange}; }
.mlp-train-btn { font: 600 13px ${FONT}; letter-spacing: 0.06em; height: 40px;
  padding: 0 18px; border-radius: 999px; border: none; background: ${C.orange};
  color: #fff; cursor: pointer; position: relative; white-space: nowrap; }
.mlp-train-btn:hover { background: #b84f00; }
.mlp-train-btn:disabled { background: ${C.grid}; cursor: default; }
.mlp-train-btn.mlp-dirty::after { content: ""; position: absolute; top: 4px; right: 6px;
  width: 8px; height: 8px; border-radius: 50%; background: #fff; }
.mlp-counts { display: flex; gap: 14px; }
.mlp-counts .mls-number { width: 4.2em; }
`;

const fmtTick = (v) => {
  const e = Math.floor(Math.log10(v) + 1e-9);
  return e >= -2 && e <= 2 ? String(+v.toPrecision(1)) : `10${sup(e)}`;
};

// One snapshot's payload [[W flat, b], ...] -> [{W, b, nIn, nOut}, ...], or
// null if malformed. W is row-major: W[j][i] multiplies input i into neuron j.
export function parseLayers(snapshot) {
  if (!Array.isArray(snapshot) || !snapshot.length) return null;
  const layers = [];
  for (const [wf, b] of snapshot) {
    if (!Array.isArray(wf) || !Array.isArray(b) || b.length < 1) return null;
    const nOut = b.length, nIn = wf.length / nOut;
    if (!Number.isInteger(nIn) || nIn < 1) return null;
    const Wm = [];
    for (let j = 0; j < nOut; j++) Wm.push(wf.slice(j * nIn, (j + 1) * nIn));
    layers.push({ W: Wm, b, nIn, nOut });
  }
  return layers;
}

// The hidden activations of layers 1..upto on a grid of m points: h[0] is the
// input (one curve, so layer 1 must have a single input), h[l] one curve per neuron.
export function forwardPass(layers, upto, x0, x1, actName, m = 24) {
  const f = ACTS[actName] || ACTS.ReLU;
  const grid = [];
  for (let i = 0; i < m; i++) grid.push(x0 + (x1 - x0) * i / (m - 1));
  const out = [[grid]];
  for (let l = 0; l < upto; l++) {
    const { W: Wm, b, nIn, nOut } = layers[l];
    const cur = [];
    for (let j = 0; j < nOut; j++) {
      const c = new Array(m);
      for (let g = 0; g < m; g++) {
        let z = b[j];
        for (let i = 0; i < nIn; i++) z += Wm[j][i] * out[l][i][g];
        c[g] = f(z);
      }
      cur.push(c);
    }
    out.push(cur);
  }
  return out;
}

function render({ model, el }) {
  const st = makeStage(el, W);
  const local = elt("style");
  local.textContent = LOCAL_CSS;
  st.stage.append(local);

  // ---- state ------------------------------------------------------------------
  let epoch = 0;          // replay position in epochs (a float while playing)
  let playing = false;    // the playback loop is running
  let autoplay = true;    // every new run plays (cleared by pause, step or scrub)
  let pending = !model.get("run_id");  // inputs changed, the new run not yet here
  let progress = 0;       // share of the epochs trained so far (sent by Python)
  let auto = model.get("auto_retrain");  // retrain as soon as a setting changes
  let autoTimer = null, queued = false;  // a change waiting for the run in progress
  let lastT = 0;
  let live = null;        // the points of the chosen set while drawing
  let drawn = [];         // the same points as they were when the drag began
  let drawSet = "train";  // the set a drag draws or erases
  const setName = (set) => (set === "val" ? "validation" : set);
  let erasing = false;
  let hover = null, pinned = null, mouse = [0, 0];
  let cur = null;         // weights of the replayed epoch (null: shape mismatch)
  let matGeom = [];

  // ---- header bar -------------------------------------------------------------
  const bar = elt("div", "mls-bar");
  bar.style.flexWrap = "nowrap";  // one line: the credit never wraps below the bar
  const trainBtn = elt("button", "mlp-train-btn", "TRAIN");
  trainBtn.onclick = train;
  const tp = uiTransport({
    onReset: reset, onPlay: togglePlay, onStep: stepOnce,
    resetTitle: "re-initialise the weights and train again",
  });
  const epochStat = uiStat("EPOCH", "mls-counter");
  const lrSel = uiSelect(LRS.map((v) => [v, String(v)]), model.get("lr"),
    (v) => setInput("lr", v));
  const actSel = uiSelect(Object.keys(ACTS).map((k) => [k, k]),
    model.get("activation"), (v) => { setInput("activation", v); drawAct(v); });
  const actPlot = elt("div");
  actPlot.style.cssText = "margin-left:-8px;";
  // the chosen activation as a small plot: phi(z) for z in [-2, 2]
  function drawAct(name) {
    const f = ACTS[name] || ACTS.ReLU, w = 76, hgt = 44;
    const X = (z) => 4 + (z + 2) / 4 * (w - 8);
    const Y = (v) => 4 + (2 - v) / 3.2 * (hgt - 8);  // v from -1.2 to 2
    const pts = [];
    for (let k = 0; k <= 60; k++) {
      const z = -2 + 4 * k / 60;
      pts.push(`${X(z).toFixed(1)},${Y(f(z)).toFixed(1)}`);
    }
    actPlot.innerHTML = `<svg width="${w}" height="${hgt}" viewBox="0 0 ${w} ${hgt}" ` +
      `style="display:block"><title>${name}: φ(z) for z from −2 to 2</title>` +
      `<line x1="2" x2="${w - 2}" y1="${Y(0)}" y2="${Y(0)}" stroke="${C.grid}"/>` +
      `<line x1="${X(0)}" x2="${X(0)}" y1="2" y2="${hgt - 2}" stroke="${C.grid}"/>` +
      `<clipPath id="mls-act-clip"><rect x="0" y="2" width="${w}" ` +
      `height="${hgt - 4}"/></clipPath>` +
      `<polyline points="${pts.join(" ")}" fill="none" stroke="${C.ink}" ` +
      `clip-path="url(#mls-act-clip)" ` +
      `stroke-width="1.8" stroke-linejoin="round"/>` +
      `<text x="${w - 3}" y="${Y(0) + 11}" text-anchor="end" font-size="11" ` +
      `font-style="italic" font-family="${SYM.replace(/"/g, "'")}" ` +
      `fill="${C.ink2}">z</text></svg>`;
  }
  drawAct(model.get("activation"));
  const epInput = uiNumber(model.get("epochs"), EPOCHS, (v) => setInput("epochs", v));
  const credit = elt("div", "mlp-note");  // credit, right end of the bar
  credit.style.cssText = "width:auto;align-self:center;white-space:nowrap;flex:none;";
  credit.innerHTML = `inspired by <a href="https://playground.tensorflow.org" ` +
    `target="_blank" rel="noopener" style="color:${C.blue}">TensorFlow Playground</a>`;
  const retrainSeg = uiSegmented([[true, "on change"], [false, "on button"]], auto, (v) => {
    auto = v;
    model.set("auto_retrain", v);
    model.save_changes();
    if (auto) scheduleAuto();
    updateStatus();
  });
  retrainSeg.title = "retrain at every change, or only when TRAIN is pressed";
  const status = elt("div", "mls-status");
  const statusText = elt("span"), trainBar = uiProgress();
  trainBar.el.style.marginLeft = "8px";
  status.append(statusText, trainBar.el);
  const rule = () => elt("div", "mls-rule");
  bar.append(trainBtn, uiField(uiLabel("RETRAIN"), retrainSeg), rule(),  // train
    uiField(uiLabel("STEP SIZE", "η"), lrSel),                          // settings
    uiField(uiLabel("ACTIVATION", "φ"), actSel), actPlot,
    uiField(uiLabel("EPOCHS"), epInput), rule(),
    tp.el, epochStat.el,                                                // playback
    status, credit);
  const scrub = uiScrub((f) => { pause(); setEpoch(f * lastEpoch()); });

  const body = elt("div");
  body.style.cssText = `position:relative;width:${W}px;height:${H}px;margin-top:8px;`;
  const cv = stageCanvas(st, W, H);
  const { canvas, ctx } = cv;
  canvas.style.cursor = "crosshair";
  body.append(canvas);
  st.stage.append(bar, scrub.el, body);
  st.onScale(() => { cv.resize(); draw(); });

  // ---- column heads and the DOM controls on the figure ------------------------
  const put = (e, x, y, align) => body.appendChild(uiPlace(e, x, y, align));
  put(elt("div", "mls-head", "DATA"), 0, 0);
  put(elt("div", "mls-head", "OUTPUT"), OUTX, 0);
  const row2 = [];  // [element, y]: moved down with grow()
  const putRow2 = (e, x, dy) => { row2.push([put(e, x, ROW2 + dy), dy]); return e; };
  putRow2(elt("div", "mls-head", "WEIGHT MATRICES"), 0, 0);
  putRow2(elt("div", "mlp-note", "one cell per edge, one block per band of edges"),
    0, 20);

  const setSeg = uiSegmented([["train", "● train"], ["val", "○ validation"]], drawSet,
    (v) => { drawSet = v; clearBtn.textContent = `clear ${setName(v)}`; });
  put(setSeg, 0, 22);
  const presets = elt("div");
  presets.style.cssText = "display:grid;grid-template-columns:62px 62px;gap:10px;";
  for (const [name, f] of Object.entries(PRESETS)) {
    const b = elt("button", "mlp-preset");
    b.title = name;
    const pts = [];
    for (let i = 0; i <= 40; i++) {
      const x = -1 + i / 20;
      pts.push(`${(6 + 50 * i / 40).toFixed(1)},${(22 - 17 * f(x)).toFixed(1)}`);
    }
    b.innerHTML = `<svg viewBox="0 0 62 44"><line x1="4" x2="58" y1="22" y2="22" ` +
      `stroke="${C.grid}"/><polyline points="${pts.join(" ")}" fill="none" ` +
      `stroke="${C.blue}" stroke-width="2" stroke-linejoin="round"/></svg>`;
    b.onclick = () => load(name);
    presets.append(b);
  }
  put(presets, 0, 62);
  const nTrainIn = uiNumber(model.get("n_train"), N_TRAIN, (v) => setCount("train", v));
  const nValIn = uiNumber(model.get("n_val"), N_VAL, (v) => setCount("val", v));
  const countsBox = elt("div", "mlp-counts");
  const trainField = uiField(uiLabel("TRAIN"), nTrainIn);
  const valField = uiField(uiLabel("VALIDATION"), nValIn);
  countsBox.append(trainField, valField);
  put(countsBox, 0, 172);
  const noiseTrainIn = uiDecimal(model.get("noise_train"), NOISE_MAX,
    (v) => setNoise("train", v));
  const noiseValIn = uiDecimal(model.get("noise_val"), NOISE_MAX,
    (v) => setNoise("val", v));
  const noiseBox = elt("div", "mlp-counts");
  const noiseTrainField = uiField(uiLabel("NOISE"), noiseTrainIn);
  const noiseValField = uiField(uiLabel("NOISE"), noiseValIn);
  noiseBox.append(noiseTrainField, noiseValField);
  put(noiseBox, 0, 224);
  const newNoiseBtn = elt("button", "mlp-pill", "new noise");
  newNoiseBtn.title = "draw the noise of the example again";
  newNoiseBtn.onclick = newNoise;
  put(newNoiseBtn, 0, 276);
  const clearBtn = elt("button", "mlp-pill", `clear ${setName(drawSet)}`);
  clearBtn.onclick = () => commit(drawSet, []);
  put(clearBtn, 0, 314);
  put(elt("div", "mlp-note",
    "drag on the output plot to draw the chosen set, shift-drag to erase"), 0, 358);

  const depth = elt("div", "mlp-depth");
  const depthPM = uiPlusMinus(() => setDepth(+1), () => setDepth(-1));
  const depthText = elt("div", "mls-head");
  depth.append(depthPM, depthText);
  put(depth, (NET.xIn + NET.xOut) / 2, -4, "center");
  const colBox = elt("div");
  body.append(colBox);

  const lossStat = uiStat("TRAIN LOSS (MSE)", "mlp-loss mlp-train");
  put(lossStat.el, OUTX, 26);
  const valStat = uiStat("VALIDATION LOSS", "mlp-loss mlp-val");
  put(valStat.el, OUTX + 125, 26);
  const paramStat = uiStat("PARAMETERS");
  putRow2(paramStat.el, OUTX, 0);
  putRow2(elt("div", "mls-head", "WEIGHT"), OUTX + 170, 0);
  putRow2(elt("div", "mls-head", "NEURON"), OUTX, 76);

  // ---- the network's shape and the replayed weights ---------------------------
  const trainPts = () => (live && drawSet === "train" ? live : model.get("points"));
  const valPts = () => (live && drawSet === "val" ? live : model.get("val_points"));
  const nTrain = () => trainPts().length;
  const xr = () => model.get("x_range"), yr = () => model.get("y_range");
  const widths = () => {
    const w = model.get("widths");
    if (!Array.isArray(w)) return [4];  // an empty list is a net without hidden layers
    return w.slice(0, MAXL).map((v) =>
      Math.min(Math.max(Math.round(Number(v)) || 1, 1), MAXW));
  };
  const cols = () => [1, ...widths(), 1];
  const nCols = () => widths().length + 2;
  const colX = (c) => NET.xIn + (NET.xOut - NET.xIn) * c / (nCols() - 1);
  const span = () => Math.max(NET.minSpan, (Math.max(...widths()) - 1) * PITCH);
  const grow = () => span() - NET.minSpan;  // extra height for a wide layer
  const cy = () => NET.top + span() / 2;
  const nodeY = (c, i) => cy() + (i - (cols()[c] - 1) / 2) * PITCH;
  const tileOf = (c) => (c === 0 || c === nCols() - 1 ? IO : TILE);

  // fewer than three points: no training set, so the last run is hidden (until
  // the next run arrives, also if points are drawn again meanwhile)
  let cleared = false;
  const noData = () => cleared || nTrain() < 3;
  const E = () => (noData() ? [] : model.get("frame_epochs"));
  const lastEpoch = () => (E().length ? E().at(-1) : 0);
  const framePos = () => {  // the snapshots either side of the replayed epoch
    const e = E(), n = e.length;
    if (!n) return [0, 0, 0];
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (e[mid] <= epoch) lo = mid; else hi = mid;
    }
    if (epoch >= e[hi]) lo = hi;
    const k1 = Math.min(lo + 1, n - 1);
    const t = k1 === lo ? 0 : (epoch - e[lo]) / (e[k1] - e[lo]);
    return [lo, k1, Math.min(Math.max(t, 0), 1)];
  };
  const frameCurve = () => {  // the network output, interpolated
    const fr = model.get("frames");
    if (!fr.length || noData()) return [];
    const [k0, k1, t] = framePos();
    const a = fr[k0], b = fr[k1];
    return t === 0 || a.length !== b.length ? a : a.map((v, i) => v + (b[i] - v) * t);
  };
  const frameLayers = () => {  // the weights, interpolated
    const ws = model.get("weights");
    if (!ws.length || noData()) return null;
    const [k0, k1, t] = framePos();
    const a = parseLayers(ws[k0]);
    if (!a || t === 0) return a;
    const b = parseLayers(ws[k1]);
    if (!b || b.length !== a.length || b.some((l, i) =>
      l.nIn !== a[i].nIn || l.nOut !== a[i].nOut)) return a;
    return a.map((lay, l) => ({
      ...lay,
      W: lay.W.map((row, j) => row.map((w, i) => w + (b[l].W[j][i] - w) * t)),
      b: lay.b.map((v, j) => v + (b[l].b[j] - v) * t),
    }));
  };
  const matchShape = (ly) => {  // null while the new shape is still training
    const cs = cols();
    if (!ly || ly.length !== cs.length - 1) return null;
    return ly.every((l, i) => l.nIn === cs[i] && l.nOut === cs[i + 1]) ? ly : null;
  };
  const lossAt = (e, key = "loss") => {  // loss[k]: the loss after k epochs
    const loss = model.get(key);
    if (!loss.length || noData()) return NaN;
    const x = Math.min(Math.max(e, 0), loss.length - 1), k = Math.floor(x);
    const k1 = Math.min(k + 1, loss.length - 1);
    return loss[k] + (loss[k1] - loss[k]) * (x - k);
  };
  const actName = () => model.get("fit_activation");
  const selected = () =>
    (hover?.kind === "neuron" ? hover : null) || pinned;

  // ---- playback -----------------------------------------------------------------
  const syncPlay = () => tp.setPlaying(playing || (pending && autoplay));
  function setEpoch(e) {
    epoch = Math.min(Math.max(e, 0), lastEpoch());
    draw();
  }
  function tick(now) {
    if (!playing) return;
    const dt = Math.min(now - lastT, 100);
    lastT = now;
    epoch = Math.min(epoch + lastEpoch() * dt / playMs(lastEpoch()), lastEpoch());
    if (epoch >= lastEpoch()) { playing = false; syncPlay(); }
    draw();
    if (playing) requestAnimationFrame(tick);
  }
  function play() {
    autoplay = true;
    if (!pending && E().length > 1) {
      if (epoch >= lastEpoch()) epoch = 0;
      if (!playing) {
        playing = true;
        lastT = performance.now();
        requestAnimationFrame(tick);
      }
    }
    syncPlay();
  }
  function pause() { playing = false; autoplay = false; syncPlay(); }
  function togglePlay() { if (playing || (pending && autoplay)) pause(); else play(); }
  function stepOnce() {
    pause();
    const next = E().find((e) => e > epoch + 1e-6);
    setEpoch(next === undefined ? lastEpoch() : next);
  }
  const settings = () => {
    const out = {};
    for (const k of TRAIN_KEYS) out[k] = JSON.parse(JSON.stringify(model.get(k)));
    return out;
  };
  const dirty = () => !same(settings(), model.get("trained"));
  function train() {  // capture the settings: Python trains and sends the run
    if (pending || nTrain() < 3) return;
    if (!dirty()) { epoch = 0; play(); return; }  // nothing new: play it again
    model.set("trained", settings());
    model.set("train_id", model.get("train_id") + 1);
    model.save_changes();
    pending = true;
    progress = 0;
    playing = false;
    syncPlay();
    updateStatus();
    draw();
  }
  // automatic mode: train once a burst of changes is over, after the current run
  function scheduleAuto() {
    clearTimeout(autoTimer);
    autoTimer = setTimeout(() => {
      if (!auto || nTrain() < 3 || !dirty() || estSeconds() > AUTO_MAX) return;
      if (pending) queued = true; else train();
    }, 150);
  }
  function estSeconds() {  // rough: a run on a laptop CPU, from the epochs and depth
    return 0.3 + 0.0007 * model.get("epochs") * (1 + 0.3 * widths().length);
  }
  const waitsForButton = () => !auto || estSeconds() > AUTO_MAX;
  function reset() {  // new initial weights: a new seed, trained again
    model.set("seed", model.get("seed") + 1);
    model.save_changes();
    train();
  }
  function newRun() {
    pending = false;
    if (queued) { queued = false; scheduleAuto(); }
    cleared = false;
    playing = false;
    epoch = 0;
    if (autoplay) play(); else syncPlay();
    updateStatus();
    draw();
  }

  // ---- inputs (a new run on change, or when train is pressed) ------------------
  function setInputs(values) {
    for (const [name, value] of Object.entries(values)) model.set(name, value);
    model.save_changes();
    if (Object.keys(values).some((k) => TRAIN_KEYS.includes(k))) scheduleAuto();
    syncData();
    updateStatus();
    draw();
  }
  const setInput = (name, value) => setInputs({ [name]: value });
  function updateStatus() {
    const few = nTrain() < 3, changed = !pending && dirty();
    trainBtn.disabled = few || pending;
    trainBtn.textContent = pending ? "TRAINING…" : "TRAIN";
    trainBtn.classList.toggle("mlp-dirty", changed);
    trainBtn.title = changed ? "settings changed since the last training" : "train again";
    statusText.textContent = few ? "draw at least three train points"
      : pending ? "training…" : changed
        ? (waitsForButton() ? "changed: press TRAIN" : "") : "";
    trainBar.el.style.display = !few && pending ? "" : "none";
    trainBar.set(progress);
  }
  // The number fields: the example's counts, or (off) the counts of the drawing.
  function syncData() {
    const byExample = !!model.get("example");
    nTrainIn.value = String(byExample ? model.get("n_train") : model.get("points").length);
    nValIn.value = String(byExample ? model.get("n_val") : model.get("val_points").length);
    trainField.setOff(!byExample);
    valField.setOff(!byExample);
    noiseTrainIn.value = String(model.get("noise_train"));
    noiseValIn.value = String(model.get("noise_val"));
    noiseTrainField.setOff(!byExample);
    noiseValField.setOff(!byExample);
    newNoiseBtn.disabled = !byExample;
  }
  // The example's points with the noise settings of the fields.
  const exampleSet = (f, which) => sample(f, model.get(which ? "n_val" : "n_train"),
    which, xr(), model.get(which ? "noise_val" : "noise_train"), model.get("noise_seed"));
  function setNoise(which, v) {  // rms noise of one set of the example
    const f = PRESETS[model.get("example")];
    if (!f) return;
    const train = which === "train";
    model.set(train ? "noise_train" : "noise_val", v);
    setInputs({ [train ? "points" : "val_points"]: exampleSet(f, train ? 0 : 1) });
  }
  function newNoise() {  // the same curve and noise levels, a new draw of the noise
    const f = PRESETS[model.get("example")];
    if (!f) return;
    model.set("noise_seed", model.get("noise_seed") + 1);
    setInputs({ points: exampleSet(f, 0), val_points: exampleSet(f, 1) });
  }
  function commit(which, pts) {  // a hand-drawn set: the data are no longer an example
    setInputs({ [which === "train" ? "points" : "val_points"]: pts, example: "" });
  }
  function load(name) {  // an example, with the counts of the number fields
    const f = PRESETS[name];
    setInputs({
      example: name,
      points: exampleSet(f, 0),
      val_points: exampleSet(f, 1),
    });
  }
  function setCount(which, n) {  // more or fewer points of the example
    const f = PRESETS[model.get("example")];
    if (!f) return;
    const train = which === "train";
    model.set(train ? "n_train" : "n_val", n);
    setInputs({ [train ? "points" : "val_points"]: exampleSet(f, train ? 0 : 1) });
  }
  function setDepth(d) {
    const w = widths(), L = Math.min(MAXL, Math.max(0, w.length + d));
    if (L === w.length) return;
    const nw = w.slice(0, L);
    while (nw.length < L) nw.push(4);
    setInput("widths", nw);
  }
  function setWidth(l, d) {
    const nw = [...widths()];
    nw[l] = Math.min(MAXW, Math.max(1, nw[l] + d));
    if (nw[l] !== widths()[l]) setInput("widths", nw);
  }
  function buildColumns() {  // +/- and the neuron count above each hidden column
    colBox.replaceChildren();
    const w = widths();
    w.forEach((n, l) => {
      const col = elt("div", "mlp-col");
      const pm = uiPlusMinus(() => setWidth(l, +1), () => setWidth(l, -1));
      pm.enable(n < MAXW, n > 1);
      col.append(pm, elt("div", "mlp-count", `${n} neuron${n > 1 ? "s" : ""}`));
      colBox.append(uiPlace(col, colX(l + 1), 30, "center"));
    });
    depthPM.enable(w.length < MAXL, w.length > 0);
    depthText.textContent = `${w.length} HIDDEN LAYER${w.length === 1 ? "" : "S"}`;
    if (pinned && (pinned.col >= nCols() || pinned.idx >= cols()[pinned.col]))
      pinned = null;
    const cs = cols();
    let total = 0;
    for (let l = 0; l < cs.length - 1; l++) total += cs[l] * cs[l + 1] + cs[l + 1];
    paramStat.set(String(total));
    for (const [e, dy] of row2) e.style.top = `${ROW2 + grow() + dy}px`;
    body.style.height = `${H + grow()}px`;
    cv.setHeight(H + grow());
  }

  // ---- drawing helpers ------------------------------------------------------------
  function text(s, x, y, align = "left", colour = C.ink, size = 11, font = FONT,
    style = "") {
    ctx.font = `${style ? style + " " : ""}${size}px ${font}`;
    ctx.fillStyle = colour; ctx.textAlign = align;
    ctx.fillText(s, x, y);
  }
  const toPx = ([x, y]) => [
    PLOT.l + (x - xr()[0]) / (xr()[1] - xr()[0]) * PLOT.w,
    PLOT.t + PLOT.h - (y - yr()[0]) / (yr()[1] - yr()[0]) * PLOT.h,
  ];
  const dataOf = (px, py) => [
    xr()[0] + (px - PLOT.l) / PLOT.w * (xr()[1] - xr()[0]),
    yr()[0] + (PLOT.t + PLOT.h - py) / PLOT.h * (yr()[1] - yr()[0]),
  ];
  const inPlot = (px, py) => px >= PLOT.l && px <= PLOT.l + PLOT.w &&
    py >= PLOT.t && py <= PLOT.t + PLOT.h;
  function curve(ys, colour, width, dash = []) {
    const n = ys.length;
    ctx.strokeStyle = colour; ctx.lineWidth = width; ctx.lineJoin = "round";
    ctx.setLineDash(dash); ctx.beginPath();
    ys.forEach((y, i) => {
      const [px, py] = toPx([xr()[0] + (xr()[1] - xr()[0]) * i / (n - 1), y]);
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    });
    ctx.stroke(); ctx.setLineDash([]);
  }
  function tag(s, x, y) {  // a value label on a white chip
    ctx.font = `12px ${FONT}`;
    const w = ctx.measureText(s).width;
    x = Math.min(Math.max(x, w / 2 + 8), W - w / 2 - 8);
    ctx.fillStyle = C.bg; ctx.strokeStyle = C.ink2; ctx.lineWidth = 0.8;
    ctx.beginPath(); ctx.roundRect(x - w / 2 - 6, y - 11, w + 12, 21, 4);
    ctx.fill(); ctx.stroke();
    text(s, x, y + 4, "center", C.ink, 12);
  }
  // A neuron tile: its output over x, filled blue above zero and orange below.
  // ``scale`` is the value drawn at the tile edge (one per layer, so the tiles
  // of a layer compare).
  function drawTile(x, y, s, ys, { line = C.ink, border = C.grid, lw = 1, scale = 1 } = {}) {
    const r = 3, x0 = x - s / 2, y0 = y - s / 2;
    ctx.fillStyle = C.bg; ctx.beginPath(); ctx.roundRect(x0, y0, s, s, r); ctx.fill();
    if (ys && ys.length > 1) {
      ctx.save(); ctx.beginPath(); ctx.roundRect(x0, y0, s, s, r); ctx.clip();
      const m = ys.length;
      const amp = s / 2 - 2;
      const P = ys.map((v, g) => [x0 + 2 + (s - 4) * g / (m - 1), y - v / scale * amp]);
      const area = () => {
        ctx.beginPath(); ctx.moveTo(P[0][0], y);
        for (const [px, py] of P) ctx.lineTo(px, py);
        ctx.lineTo(P[m - 1][0], y); ctx.closePath();
      };
      for (const [colour, top] of [[C.blue, true], [C.orange, false]]) {
        ctx.save(); ctx.beginPath();
        ctx.rect(x0, top ? y0 : y, s, s / 2); ctx.clip();
        ctx.globalAlpha = 0.4; ctx.fillStyle = colour; area(); ctx.fill();
        ctx.restore();
      }
      ctx.globalAlpha = 1;
      ctx.strokeStyle = line; ctx.lineWidth = Math.max(1, s / 22); ctx.beginPath();
      P.forEach(([px, py], g) => (g ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
      ctx.stroke();
      ctx.restore();
    }
    ctx.strokeStyle = border; ctx.lineWidth = lw;
    ctx.beginPath(); ctx.roundRect(x0, y0, s, s, r); ctx.stroke();
  }

  // ---- the network ---------------------------------------------------------------
  function edgeOn(c, i, j) {
    const hv = hover, s = selected();
    return (hv?.kind === "edge" && hv.c === c && hv.i === i && hv.j === j)
      || (hv?.kind === "cell" && hv.layer === c && hv.row === j && hv.col === i)
      || (hv?.kind === "bcell" && hv.layer === c && hv.row === j)
      || (s && (s.col === c && s.idx === i || s.col === c + 1 && s.idx === j));
  }
  function drawNet() {
    const cs = cols(), L = cs.length - 2, ly = cur;
    const focus = hover || pinned;
    const maxAbs = ly ? ly.map((l) => Math.max(...l.W.flat().map(Math.abs), 1e-9)) : [];
    for (let c = 0; c < cs.length - 1; c++) {
      for (let i = 0; i < cs[c]; i++) for (let j = 0; j < cs[c + 1]; j++) {
        const x1 = colX(c), y1 = nodeY(c, i), x2 = colX(c + 1), y2 = nodeY(c + 1, j);
        ctx.lineCap = "round";
        if (!ly) {  // the new shape is still training
          ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
          continue;
        }
        const w = ly[c].W[j][i], on = edgeOn(c, i, j);
        ctx.globalAlpha = on ? 1 : focus ? 0.12 : 0.85;
        ctx.strokeStyle = w >= 0 ? C.blue : C.orange;
        ctx.lineWidth = (0.6 + 3.6 * Math.abs(w) / maxAbs[c]) * (on ? 1.5 : 1);
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }
    // tiles: input, hidden (their outputs over x), output (the fit)
    const [x0, x1] = xr();
    const fp = ly ? forwardPass(ly, L, x0, x1, actName(), 24) : null;
    const grid = [...Array(24).keys()].map((g) => x0 + (x1 - x0) * g / 23);
    drawTile(colX(0), cy(), IO, grid, { border: C.ink2 });
    text("x", colX(0) - IO / 2 - 10, cy() + 6, "right", C.ink, 19, SYM, "italic");
    const s = selected();
    for (let c = 1; c <= L; c++) {
      const scale = fp ? Math.max(...fp[c].flat().map(Math.abs), 1e-6) : 1;
      for (let j = 0; j < cs[c]; j++) {
        const isPin = pinned?.col === c && pinned.idx === j;
        const isSel = s?.col === c && s.idx === j;
        drawTile(colX(c), nodeY(c, j), tileOf(c), fp ? fp[c][j] : null, {
          border: isPin ? C.green : isSel ? C.ink : C.ink2, lw: isSel || isPin ? 2 : 1,
          scale });
      }
    }
    const fit = frameCurve();
    const fitSmall = fit.length ? [...Array(24).keys()].map((i) =>
      fit[Math.round(i * (fit.length - 1) / 23)]) : null;
    const oc = L + 1;
    drawTile(colX(oc), cy(), IO, fitSmall, {
      scale: Math.max(1, ...(fitSmall || []).map(Math.abs)),
      border: pinned?.col === oc ? C.green : s?.col === oc ? C.ink : C.ink2,
      lw: s?.col === oc ? 2 : 1 });
    text("y", colX(oc) + IO / 2 + 10, cy() + 6, "left", C.ink, 19, SYM, "italic");
    if (hover?.kind === "edge" && ly) {
      const { c, i, j } = hover;
      tag(`W${sup(`(${c + 1})`)}${sub(j + 1)}${sub(i + 1)} = ${ly[c].W[j][i].toFixed(2)}`,
        (colX(c) + colX(c + 1)) / 2, (nodeY(c, i) + nodeY(c + 1, j)) / 2 - 14);
    }
    // brackets: the band of edges that each matrix block below holds
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1.2;
    for (let c = 0; c < cs.length - 1; c++) {
      const a = colX(c) + 8, b = colX(c + 1) - 8, y = NET.bracket + grow();
      ctx.beginPath(); ctx.moveTo(a, y - 5); ctx.lineTo(a, y); ctx.lineTo(b, y);
      ctx.lineTo(b, y - 5); ctx.stroke();
    }
  }

  // ---- the output: train and validation loss and the fit -------------------------------
  function drawLoss() {
    const hasData = !noData();
    const loss = hasData ? model.get("loss") : [], n = loss.length;
    const valLoss = hasData && n ? model.get("val_loss") : [];
    const hasVal = valLoss.length === n && n > 0;
    lossStat.set(n ? fmtStat(lossAt(epoch)) : "—");
    valStat.set(hasVal ? fmtStat(lossAt(epoch, "val_loss")) : "—");
    if (n < 2) return;
    const { l, t, w, h } = SPARK;
    const ok = [...loss, ...(hasVal ? valLoss : [])].filter((v) => v > 0 && isFinite(v));
    if (!ok.length) return;
    const lo = Math.log10(Math.min(...ok)) - 0.05, hi = Math.log10(Math.max(...ok)) + 0.05;
    const X = (k) => l + k / (n - 1) * w;
    const Y = (v) => t + h - (Math.log10(Math.max(v, 1e-12)) - lo) / (hi - lo) * h;
    for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) {
      const y = Y(10 ** e);
      ctx.strokeStyle = C.grid; ctx.lineWidth = 0.8;
      ctx.beginPath(); ctx.moveTo(l, y); ctx.lineTo(l + w, y); ctx.stroke();
      text(fmtTick(10 ** e), l + w + 6, y + 4, "left", C.ink2, 10);
    }
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(l, t + h); ctx.lineTo(l + w, t + h); ctx.stroke();
    const upto = Math.min(Math.floor(epoch), n - 1);
    const curves = [[loss, "loss", C.blue]];
    if (hasVal) curves.unshift([valLoss, "val_loss", C.orange]);
    for (const [values, key, colour] of curves) {
      ctx.strokeStyle = colour; ctx.lineWidth = 1.8; ctx.lineJoin = "round";
      ctx.beginPath();
      for (let k = 0; k <= upto; k++)
        k ? ctx.lineTo(X(k), Y(values[k])) : ctx.moveTo(X(k), Y(values[k]));
      ctx.lineTo(X(Math.min(epoch, n - 1)), Y(lossAt(epoch, key)));
      ctx.stroke();
      ctx.fillStyle = colour; ctx.beginPath();
      ctx.arc(X(Math.min(epoch, n - 1)), Y(lossAt(epoch, key)), 3.5, 0, 7); ctx.fill();
    }
  }

  function drawPlot() {
    const { l, t, w, h } = PLOT;
    for (const v of [-1, -0.5, 0, 0.5, 1]) {
      const [x] = toPx([v, 0]), [, y] = toPx([0, v]);
      ctx.strokeStyle = C.grid; ctx.lineWidth = 0.8;
      ctx.beginPath(); ctx.moveTo(x, t); ctx.lineTo(x, t + h);
      ctx.moveTo(l, y); ctx.lineTo(l + w, y); ctx.stroke();
      text(String(v), x, t + h + 17, "center", C.ink, 11);
      text(String(v), l - 8, y + 4, "right", C.ink, 11);
    }
    text("x", l + w / 2, t + h + 36, "center", C.ink, 16, SYM, "italic");
    text("y", l - 38, t + h / 2 + 5, "center", C.ink, 16, SYM, "italic");

    const drawing = live !== null;
    const s = selected(), L = widths().length;
    ctx.save(); ctx.beginPath(); ctx.rect(l - 4, t - 4, w + 8, h + 8); ctx.clip();
    if (!drawing && cur && s && s.col <= L) {  // the selected neuron's output h(x)
      const fp = forwardPass(cur, s.col, xr()[0], xr()[1], actName(), 90);
      curve(fp[s.col][s.idx], C.green, 2.2, [6, 4]);
    }
    if (!drawing && frameCurve().length) curve(frameCurve(), C.ink, 2.8);
    ctx.strokeStyle = C.orange; ctx.lineWidth = 1.6;  // validation: hollow
    for (const p of valPts()) {
      const [x, y] = toPx(p);
      ctx.beginPath(); ctx.arc(x, y, 3.4, 0, 7); ctx.stroke();
    }
    ctx.fillStyle = C.blue;  // train: filled
    for (const p of trainPts()) {
      const [x, y] = toPx(p);
      ctx.beginPath(); ctx.arc(x, y, 3.2, 0, 7); ctx.fill();
    }
    ctx.restore();
    if (!trainPts().length && !valPts().length)
      text("draw a curve here", l + w / 2, t + h / 2, "center", C.ink2, 14);

    // key, one line above the plot
    let kx = l;
    const ky = t - 14;
    const item = (draw, s2, colour = C.ink) => {
      draw(kx); kx += 24;
      text(s2, kx, ky + 4, "left", colour, 12);
      ctx.font = `12px ${FONT}`; kx += ctx.measureText(s2).width + 18;
    };
    const seg = (colour, width, dash = []) => (x) => {
      ctx.strokeStyle = colour; ctx.lineWidth = width; ctx.setLineDash(dash);
      ctx.beginPath(); ctx.moveTo(x, ky); ctx.lineTo(x + 18, ky); ctx.stroke();
      ctx.setLineDash([]);
    };
    item((x) => { ctx.fillStyle = C.blue; ctx.beginPath(); ctx.arc(x + 9, ky, 3.2, 0, 7); ctx.fill(); },
      "train");
    item((x) => {
      ctx.strokeStyle = C.orange; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.arc(x + 9, ky, 3.4, 0, 7); ctx.stroke();
    }, "validation");
    item(seg(C.ink, 2.8), "network output");
    if (s && s.col <= L && cur) item(seg(C.green, 2.2, [6, 4]), neuronName(s.col, s.idx));
  }

  // ---- the same network as matrices --------------------------------------------------
  function fillCell(x, y, s, w, max) {
    ctx.fillStyle = w >= 0 ? C.blue : C.orange;
    ctx.globalAlpha = 0.1 + 0.9 * Math.abs(w) / max;
    ctx.fillRect(x, y, s, s);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = C.bg; ctx.lineWidth = 0.6;
    ctx.strokeRect(x, y, s, s);
  }
  function drawMatrices() {
    matGeom = [];
    const cs = cols();
    for (let l = 0; l < cs.length - 1; l++) {
      const nIn = cs[l], nOut = cs[l + 1];
      const gap = colX(l + 1) - colX(l) - 16;
      const cell = Math.min(MAT.maxCell, MAT.maxH / nOut, (gap - 8) / (nIn + 1.6));
      const bw = nIn * cell + 6 + cell, cx = (colX(l) + colX(l + 1)) / 2;
      const x0 = cx - bw / 2, y0 = MAT.top + grow(), bx = x0 + nIn * cell + 6;
      ctx.font = `bold italic 15px ${SYM}`;
      const name = `W${sup(`(${l + 1})`)}  b${sup(`(${l + 1})`)}`;
      text(name, cx, y0 - 8, "center", C.ink, 15, SYM, "bold italic");
      text(`${nOut}×${nIn} + ${nOut}`, cx, y0 + nOut * cell + 16, "center",
        C.ink2, 11);
      if (!cur) {  // the new shape is still training: empty cells
        ctx.strokeStyle = C.grid; ctx.lineWidth = 0.8;
        ctx.strokeRect(x0, y0, nIn * cell, nOut * cell);
        ctx.strokeRect(bx, y0, cell, nOut * cell);
        continue;
      }
      const lay = cur[l];
      const maxW = Math.max(...lay.W.flat().map(Math.abs), 1e-9);
      const maxB = Math.max(...lay.b.map(Math.abs), 1e-9);
      for (let j = 0; j < nOut; j++) {
        for (let i = 0; i < nIn; i++)
          fillCell(x0 + i * cell, y0 + j * cell, cell, lay.W[j][i], maxW);
        fillCell(bx, y0 + j * cell, cell, lay.b[j], maxB);
      }
      matGeom.push({ layer: l, x0, y0, cell, rows: nOut, colsN: nIn, bx });
    }
    if (!cur) return;
    const ring = (rc, colour = C.ink, lw = 1.6) => {
      if (!rc) return;
      ctx.strokeStyle = colour; ctx.lineWidth = lw;
      ctx.strokeRect(rc[0] - 1, rc[1] - 1, rc[2] + 2, rc[2] + 2);
    };
    const geom = (layer) => matGeom.find((m) => m.layer === layer);
    const cellRect = (layer, row, col) => {
      const g = geom(layer);
      return g && [g.x0 + col * g.cell, g.y0 + row * g.cell, g.cell];
    };
    const bRect = (layer, row) => {
      const g = geom(layer);
      return g && [g.bx, g.y0 + row * g.cell, g.cell];
    };
    const ringRow = (layer, row, colour) => {
      const g = geom(layer);
      if (!g) return;
      ctx.strokeStyle = colour; ctx.lineWidth = 1.6;
      ctx.strokeRect(g.x0 - 1, g.y0 + row * g.cell - 1, g.colsN * g.cell + 2, g.cell + 2);
      ring(bRect(layer, row), colour);
    };
    if (hover?.kind === "cell") {
      ring(cellRect(hover.layer, hover.row, hover.col));
      tag(`W${sup(`(${hover.layer + 1})`)}${sub(hover.row + 1)}${sub(hover.col + 1)}` +
        ` = ${cur[hover.layer].W[hover.row][hover.col].toFixed(2)}`,
        mouse[0], mouse[1] - 22);
    }
    if (hover?.kind === "bcell") {
      ring(bRect(hover.layer, hover.row));
      tag(`b${sup(`(${hover.layer + 1})`)}${sub(hover.row + 1)}` +
        ` = ${cur[hover.layer].b[hover.row].toFixed(2)}`, mouse[0], mouse[1] - 22);
    }
    if (hover?.kind === "edge") ring(cellRect(hover.c, hover.j, hover.i));
    if (hover?.kind === "neuron") ringRow(hover.col - 1, hover.idx, C.ink);
    if (pinned) ringRow(pinned.col - 1, pinned.idx, C.green);
  }

  // ---- weight key and the selected neuron -------------------------------------------
  function drawKey() {
    const x = OUTX + 170, y = ROW2 + grow() + 24, w = 120, hgt = 9;
    const g = ctx.createLinearGradient(x, 0, x + w, 0);
    g.addColorStop(0, C.orange); g.addColorStop(0.5, "#ffffff"); g.addColorStop(1, C.blue);
    ctx.fillStyle = g; ctx.fillRect(x, y, w, hgt);
    ctx.strokeStyle = C.grid; ctx.lineWidth = 0.8; ctx.strokeRect(x, y, w, hgt);
    text("−", x, y + hgt + 15, "center", C.ink, 12);
    text("0", x + w / 2, y + hgt + 15, "center", C.ink, 12);
    text("+", x + w, y + hgt + 15, "center", C.ink, 12);
  }
  function neuronName(col, idx) {
    return col === nCols() - 1 ? "y" : `h${sub(idx + 1)}${sup(`(${col})`)}`;
  }
  function drawNeuron() {
    const s = selected(), y = ROW2 + grow() + 116;
    if (!s || !cur) {
      text(cur ? "hover a neuron, click to pin it" : "", OUTX, y - 4, "left", C.ink2, 12);
      return;
    }
    const lay = cur[s.col - 1], isOut = s.col === nCols() - 1;
    const nIn = lay.nIn, shown = Math.min(nIn, 3);
    const inName = (i) => (s.col === 1 ? "x" : neuronName(s.col - 1, i));
    const num = (v, first) => `${v < 0 ? "−" : first ? "" : "+"}` +
      `${first ? "" : " "}${Math.abs(v).toFixed(2)}`;
    let args = "";
    for (let i = 0; i < shown; i++) {
      const v = lay.W[s.idx][i];
      args += `${i ? " " : ""}${num(v, i === 0)} ${inName(i)}`;
    }
    if (nIn > shown) args += " + ⋯";
    args += ` ${num(lay.b[s.idx], false)}`;
    const lhs = neuronName(s.col, s.idx);
    text(`${lhs} = ${isOut ? args : `φ(${args})`}`, OUTX, y, "left", C.ink, 17, SYM);
    const where = isOut ? "the output neuron"
      : `hidden layer ${s.col}, neuron ${s.idx + 1}, φ = ${actName()}`;
    text(`${where} · epoch ${Math.round(epoch)}` +
      (pinned ? " · pinned" : ""), OUTX, y + 22, "left", C.ink2, 12);
  }

  function draw() {
    cur = matchShape(frameLayers());
    ctx.clearRect(0, 0, W, H + grow());
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H + grow());
    drawNet(); drawLoss(); drawPlot(); drawMatrices(); drawKey(); drawNeuron();
    const last = lastEpoch();
    epochStat.set(String(Math.round(epoch)), `/ ${last || model.get("epochs")}`);
    scrub.set(last ? epoch / last : 0);
  }

  // ---- interaction --------------------------------------------------------------------
  const erase = (px, py) => {
    live = live.filter((p) => {
      const [x, y] = toPx(p);
      return Math.hypot(x - px, y - py) > ERASE;
    });
  };
  const distToSeg = (px, py, x1, y1, x2, y2) => {
    const dx = x2 - x1, dy = y2 - y1;
    const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)));
    return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
  };
  function netHit(px, py) {
    const cs = cols();
    for (let c = 1; c < cs.length; c++) {
      const half = tileOf(c) / 2 + 3;
      for (let i = 0; i < cs[c]; i++)
        if (Math.abs(px - colX(c)) <= half && Math.abs(py - nodeY(c, i)) <= half)
          return { kind: "neuron", col: c, idx: i };
    }
    if (px < colX(0) || px > colX(cs.length - 1) || py > NET.bracket + grow() - 10)
      return null;
    for (let c = 0; c < cs.length - 1; c++)
      for (let i = 0; i < cs[c]; i++) for (let j = 0; j < cs[c + 1]; j++)
        if (distToSeg(px, py, colX(c), nodeY(c, i), colX(c + 1), nodeY(c + 1, j)) < 4)
          return { kind: "edge", c, i, j };
    return null;
  }
  function matHit(px, py) {
    for (const g of matGeom) {
      if (py < g.y0 || py > g.y0 + g.rows * g.cell) continue;
      const row = Math.floor((py - g.y0) / g.cell);
      if (px >= g.bx && px <= g.bx + g.cell) return { kind: "bcell", layer: g.layer, row };
      if (px >= g.x0 && px < g.x0 + g.colsN * g.cell)
        return { kind: "cell", layer: g.layer, row, col: Math.floor((px - g.x0) / g.cell) };
    }
    return null;
  }
  const hitTest = (px, py) => (!cur ? null : py < MAT.top + grow() - 20 ? netHit(px, py)
    : matHit(px, py));
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

  canvas.addEventListener("pointerdown", (ev) => {
    const [px, py] = cv.xy(ev);
    const hit = hitTest(px, py);
    if (hit?.kind === "neuron") {  // click pins a neuron, click again releases it
      pinned = pinned && pinned.col === hit.col && pinned.idx === hit.idx
        ? null : { col: hit.col, idx: hit.idx };
      draw();
      return;
    }
    if (!inPlot(px, py)) return;
    hover = null;
    canvas.setPointerCapture(ev.pointerId);
    drawn = model.get(drawSet === "train" ? "points" : "val_points");
    live = drawn.map((p) => [...p]);
    erasing = ev.shiftKey;
    if (erasing) erase(px, py); else live.push(dataOf(px, py));
    draw();
  });
  canvas.addEventListener("pointermove", (ev) => {
    const [px, py] = cv.xy(ev);
    mouse = [px, py];
    if (live) {
      if (erasing) erase(px, py);
      else {
        const [x, y] = dataOf(Math.min(Math.max(px, PLOT.l), PLOT.l + PLOT.w),
          Math.min(Math.max(py, PLOT.t), PLOT.t + PLOT.h));
        const last = live[live.length - 1];
        if (!last || Math.hypot(...toPx([x, y]).map((v, i) => v - toPx(last)[i])) > 4)
          live.push([x, y]);
      }
      draw();
      return;
    }
    const hit = hitTest(px, py);
    if (!same(hit, hover)) { hover = hit; draw(); }
    else if (hit?.kind === "cell" || hit?.kind === "bcell") draw();  // tag follows
    canvas.style.cursor = inPlot(px, py) ? "crosshair" : hit ? "pointer" : "default";
  });
  const release = () => {
    if (!live) return;
    const pts = live.map(([x, y]) => [+x.toFixed(4), +y.toFixed(4)]);
    const untouched = erasing && pts.length === drawn.length;  // nothing erased
    const which = drawSet;
    live = null; erasing = false;
    if (untouched) draw(); else commit(which, pts);
  };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
  canvas.addEventListener("pointerleave", () => {
    if (!live && hover) { hover = null; draw(); }
  });

  model.on("change:widths", () => { buildColumns(); updateStatus(); draw(); });
  model.on("change:points", () => {
    if (nTrain() < 3) { cleared = true; playing = false; epoch = 0; syncPlay(); }
    syncData(); updateStatus(); draw();
  });
  model.on("change:val_points", () => { syncData(); draw(); });
  model.on("change:example", syncData);
  model.on("change:run_id", newRun);
  model.on("change:progress", () => {
    if (pending) { progress = model.get("progress"); updateStatus(); }
  });
  buildColumns();
  syncData();
  updateStatus();
  if (!pending && E().length > 1) newRun(); else { syncPlay(); draw(); }
}
export default { render };
