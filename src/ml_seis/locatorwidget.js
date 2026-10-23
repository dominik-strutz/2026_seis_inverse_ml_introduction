// The Lecture 1 locator playground (see locatorwidget.py); controls.js is put in
// front of this module (tokens C, FONT, SYM and the ui* controls). Reading order:
// a bar with the train button and the playback; four panels (SCENE, DATA,
// NETWORK, TRAINING); two maps that share one scene (stations, test event), each
// with its own background (P speed with rays, prior, error, posterior,
// likelihood); on the right the learning curves and predicted against true
// locations; under everything one line of errors. The network itself is drawn
// by mlp_structure_figure in the notebook, not here. With show_output the NETWORK
// panel also chooses the OUTPUT (one point or a Gaussian mixture of K components)
// and a map can show the network's posterior ("network") or compare it with the
// exact one ("compare": exact posterior shaded, the network's regions on top).
const HIT = 11;  // pick radius (logical px)

// Layout (logical px)
const W = 1200, H = 540;
const FS = 0.9;  // type scale for everything drawn on the canvas
const P = 330;                                     // side of one map
const MAPS = [{ l: 44, t: 44 }, { l: 480, t: 44 }];
const LOSS = { l: 934, t: 50, w: 250, h: 130 };    // learning curves
const SCAT = { t: 262, s: 108, ls: [934, 1078] };  // predicted vs true, m_x and m_y
const TABLE = { x: 44, y: 528 };                   // one line of errors

const BACKGROUNDS = ["P speed", "prior", "error", "posterior", "likelihood"];
const OUTPUTS = [["point", "point"], ["mixture", "mixture"]];
const MAXK = 8;  // mixture components
const COST = { point: 1, mixture: 2 };  // training time relative to point
const NET_INK = "#8a3c00";  // contours of the network's posterior
const EXACT_INK = "#00573f";  // contours of the exact posterior
const VELOCITIES = [["homogeneous", "homogeneous"], ["heterogeneous", "heterogeneous"]];
const FIELDS = [1, 2, 3, 4, 5, 6];
const SIGMAS = [0.1, 0.2, 0.3, 0.5];
const PRIOR_NAMES = [["ridgecrest", "ridgecrest + background"],
  ["ridgecrest_faults", "ridgecrest faults only"], ["uniform", "uniform"]];
const N_TRAINS = [10, 30, 100, 300, 1000, 3000, 10000, 30000];
const ACTS = ["ReLU", "tanh", "sigmoid", "linear"];
const LRS = [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1];
const BATCHES = [[0, "all"], [16, "16"], [32, "32"], [64, "64"], [128, "128"], [256, "256"]];
const EPOCHS = [30, 60, 100, 150, 300, 1000, 2000];
const DURATIONS = [1, 2, 4, 8, 16];  // seconds for one playback
const AUTO_MAX = 4;  // seconds: automatic retraining is skipped for slower runs
const WIDTHS = [1, 2, 4, 8, 16, 32, 64, 128];  // choices for a hidden layer
const MAXL = 4;  // hidden layers
const ERR_MAX = 10;  // km at the top of the error colour bar
const ERR_STOPS = [[255, 247, 236], [253, 212, 158], [252, 141, 89], [215, 48, 31],
  [127, 0, 0]];

// Every setting the training depends on (as TRAIN_KEYS in Python); the scene
// settings change the data the trained network is shown, not the network.
const TRAIN_KEYS = ["stations", "muted", "velocity", "field", "sigma", "train_prior",
  "n_train", "widths", "activation", "optimiser", "lr", "batch_size", "epochs",
  "normalise", "output", "n_components"];
const SCENE_KEYS = ["stations", "muted", "velocity", "field", "sigma", "test_prior"];

const LOCAL_CSS = `
.loc-train { font: 600 13px ${FONT}; letter-spacing: 0.06em; height: 46px;
  padding: 0 18px; border-radius: 999px; border: none; background: ${C.orange};
  color: #fff; cursor: pointer; position: relative; white-space: nowrap; }
.loc-train:hover { background: #b84f00; }
.loc-train small { display: block; font: 11px ${FONT}; letter-spacing: 0; opacity: 0.85; }
.loc-train.loc-dirty::after { content: ""; position: absolute; top: 4px; right: 6px;
  width: 8px; height: 8px; border-radius: 50%; background: ${C.ink}; }
.loc-toggle { font: 12px ${FONT}; padding: 3px 10px; border: 1px solid ${C.grid};
  border-radius: 999px; background: ${C.bg}; color: ${C.ink2}; cursor: pointer;
  white-space: nowrap; }
.loc-toggle.mls-on { background: ${C.ink}; color: ${C.bg}; border-color: ${C.ink}; }
.loc-panels { display: flex; border-top: 1px solid ${C.grid};
  border-bottom: 1px solid ${C.grid}; margin: 2px 0 6px; }
.loc-panel { padding: 6px 18px 10px 0; margin-right: 18px; display: flex;
  flex-direction: column; gap: 7px; border-right: 1px solid ${C.grid}; }
.loc-panel:last-child { border-right: none; margin-right: 0; }
.loc-row { display: flex; align-items: flex-end; gap: 14px; min-height: 36px; }
.loc-num { font: 15px ${FONT}; min-width: 1.6em; text-align: center;
  font-variant-numeric: tabular-nums; }
.loc-note { font-size: 11px; color: ${C.ink2}; }
.loc-spin { display: inline-block; width: 12px; height: 12px; border-radius: 50%;
  border: 2px solid rgba(255,255,255,0.4); border-top-color: #fff;
  animation: loc-rot 0.7s linear infinite; vertical-align: -2px; margin-right: 7px; }
.loc-spin.loc-dark { border-color: rgba(92,99,112,0.3); border-top-color: ${C.orange}; }
@keyframes loc-rot { to { transform: rotate(360deg); } }
@keyframes loc-sweep { from { left: -30%; } to { left: 100%; } }
.loc-train.loc-working { background: ${C.ink2}; pointer-events: none; }
.loc-sweep { position: absolute; top: 6px; height: 5px; width: 30%; border-radius: 3px;
  background: ${C.orange}; opacity: 0; transition: opacity 0.12s 0.1s; pointer-events: none;
  animation: loc-sweep 0.9s ease-in-out infinite; }
.loc-sweep.on { opacity: 1; }
.loc-veil { position: absolute; left: 0; top: 0; right: 0; bottom: 0; z-index: 5;
  display: flex; align-items: center; justify-content: center; pointer-events: none;
  background: rgba(255,255,255,0.6); opacity: 0; transition: opacity 0.15s 0.15s; }
.loc-veil.on { opacity: 1; }
.loc-veil div { background: ${C.ink}; color: #fff; border-radius: 999px; padding: 10px 24px;
  font: 600 13px ${FONT}; letter-spacing: 0.1em; box-shadow: 0 2px 10px rgba(0,0,0,0.25);
  min-width: 120px; text-align: center; font-variant-numeric: tabular-nums; }
.loc-veil .mls-progress { display: block; width: auto; margin-top: 7px;
  background: rgba(255,255,255,0.25); }
.loc { font-size: 12px; }
.loc .mls-label { font-size: 9px; }
.loc .mls-sym { font-size: 12px; }
.loc .mls-select, .loc .mls-number { font-size: 12px; }
.loc .mls-seg button { font-size: 11px; padding: 2px 9px; }
.loc .mls-head { font-size: 9.5px; }
.loc .mls-value { font-size: 21px; }
.loc .mls-value small { font-size: 10.5px; }
.loc .mls-status { font-size: 11px; }
.loc .loc-toggle { font-size: 11px; padding: 2px 9px; }
.loc .loc-num { font-size: 13px; }
.loc .loc-train { font-size: 12px; height: 40px; padding: 0 16px; }
.loc .loc-train small { font-size: 10px; }
.loc .mls-btn.mls-play { width: 40px; height: 40px; }
`;

const opts = (vals) => vals.map((v) => [v, String(v)]);
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const lerp = (a, b, t) => a + (b - a) * t;
const fin = (v) => typeof v === "number" && Number.isFinite(v);
const fmtKm = (v) => (fin(v) ? (v < 10 ? v.toFixed(2) : v.toFixed(1)) : "—");

function errColour(f) {  // 0..1 -> [r, g, b], white to dark red
  if (!Number.isFinite(f)) f = 1;
  const x = Math.min(Math.max(f, 0), 1) * (ERR_STOPS.length - 1);
  const i = Math.min(Math.floor(x), ERR_STOPS.length - 2), t = x - i;
  return ERR_STOPS[i].map((c, k) => Math.round(lerp(c, ERR_STOPS[i + 1][k], t)));
}

function render({ model, el }) {
  const st = makeStage(el, W);
  const local = elt("style");
  local.textContent = LOCAL_CSS;
  st.stage.append(local);
  st.stage.classList.add("loc");

  // ---- state ------------------------------------------------------------------
  let epoch = 0;          // replay position in epochs (a float while playing)
  let playing = false;
  let autoplay = true;    // play each new run (cleared by pause, step or scrub)
  let lastT = 0;
  let duration = 2;       // seconds per playback
  let auto = model.get("auto_retrain");  // retrain as soon as a setting changes
  let progress = 0;       // share of the epochs trained so far (sent by Python)
  let autoTimer = null, queued = false;  // a change waiting for the run in progress
  let drag = null, live = null, hover = null, scrubbing = false;
  let waitScene = false, waitEvent = false;  // edits sent, Python not back yet
  const showOutput = model.get("show_output");
  const backgrounds = showOutput ? [...BACKGROUNDS, "network", "compare"] : BACKGROUNDS;
  const maps = model.get("backgrounds").slice(0, 2).map((bg, i) => (
    { ...MAPS[i], bg: backgrounds.includes(bg) ? bg : "P speed", arrows: false }));
  const isEventBg = (bg) => bg === "posterior" || bg === "likelihood" || bg === "network"
    || bg === "compare";
  // the network's mixture is the answer (no point, no noise draws to show)
  const mixtureOut = () => run().config?.output === "mixture";
  let cur = maps[0];      // the map being drawn
  let view = null;        // a zoomed window {x0, y0, s (km), l, t, p (px)}
  const images = {};      // layer png -> {img, bar}

  const run = () => model.get("run") || {};
  const sv = () => model.get("scene_view") || {};
  const ev = () => model.get("event_view") || {};
  const box = () => model.get("box");
  const trainPending = () => model.get("train_id") !== run().train_id;

  // ---- the bar: train, playback, status -------------------------------------------
  const bar = elt("div", "mls-bar");
  const trainBtn = elt("button", "loc-train");
  trainBtn.onclick = () => { if (!trainPending()) train(); };
  const tp = uiTransport({
    onReset: rewind, onPlay: togglePlay, onStep: stepOnce,
    resetTitle: "back to the start of the run",
  });
  const epochStat = uiStat("EPOCH", "mls-counter");
  const durSel = uiSelect(DURATIONS.map((v) => [v, `${v} s`]), duration, (v) => {
    duration = v;
  });
  const status = elt("div", "mls-status");
  const retrainSeg = uiSegmented([[true, "on change"], [false, "on button"]], auto, (v) => {
    auto = v;
    if (auto) scheduleAuto();
    syncControls();
    draw();
  });
  retrainSeg.title = "retrain at every change, or keep the trained network and only change the data";
  bar.append(trainBtn, tp.el, epochStat.el, uiField(uiLabel("RETRAIN"), retrainSeg),
    uiField(uiLabel("PLAYBACK"), durSel), status);
  const scrub = uiScrub((f) => { pause(); setEpoch(f * lastEpoch()); });
  const sweep = elt("div", "loc-sweep");  // an indeterminate bar while Python works
  scrub.el.append(sweep);

  // ---- the four panels -------------------------------------------------------------
  const setting = (name) => (v) => setInput(name, v);
  const ctl = {};  // selects and segmented controls by setting name
  const field = (name, label, options, symbol = "", unit = "") => {
    ctl[name] = uiSelect(options, model.get(name), setting(name));
    return uiField(uiLabel(label, symbol, unit), ctl[name]);
  };
  const panel = (title, ...rows) => {
    const p = elt("div", "loc-panel");
    p.append(elt("div", "mls-head", title));
    for (const r of rows) {
      const row = elt("div", "loc-row");
      row.append(...r);
      p.append(row);
    }
    return p;
  };
  // stations: a count with + and -
  const stationNum = elt("div", "loc-num");
  const stationPM = uiPlusMinus(() => setStations(+1), () => setStations(-1));
  const stationBox = elt("div");
  stationBox.style.cssText = "display:flex;align-items:center;gap:8px;height:26px;";
  stationBox.append(stationPM, stationNum);
  const fieldF = field("field", "FIELD", opts(FIELDS));
  // network: layers with + and -, one width per layer, the activation
  const layerNum = elt("div", "loc-num");
  const layerPM = uiPlusMinus(() => setDepth(+1), () => setDepth(-1));
  const layerBox = elt("div");
  layerBox.style.cssText = "display:flex;align-items:center;gap:8px;height:26px;";
  layerBox.append(layerPM, layerNum);
  const widthsBox = elt("div", "loc-row");
  widthsBox.style.minHeight = "0";
  // output: one point or a mixture of K Gaussians
  ctl.output = uiSegmented(OUTPUTS, model.get("output"), setting("output"));
  const kNum = elt("div", "loc-num");
  const kPM = uiPlusMinus(() => setK(+1), () => setK(-1));
  const kBox = elt("div");
  kBox.style.cssText = "display:flex;align-items:center;gap:8px;height:26px;";
  kBox.append(kPM, kNum);
  const kField = uiField(uiLabel("COMPONENTS", "K"), kBox);
  ctl.optimiser = uiSegmented([["SGD", "SGD"], ["Adam", "Adam"]],
    model.get("optimiser"), setting("optimiser"));
  const toggle = (name, text) => {
    const b = elt("button", "loc-toggle", text);
    b.onclick = () => setInput(name, !model.get(name));
    b.sync = () => b.classList.toggle("mls-on", !!model.get(name));
    return b;
  };
  const toggles = { normalise: toggle("normalise", "rescale") };
  const panels = elt("div", "loc-panels");
  panels.append(
    panel("SCENE",
      [uiField(uiLabel("STATIONS"), stationBox),
        field("velocity", "P SPEED", VELOCITIES), fieldF],
      [field("sigma", "PICK NOISE", opts(SIGMAS), "σ", "s")]),
    panel("DATA",
      [field("n_train", "TRAINING EVENTS", opts(N_TRAINS)),
        field("train_prior", "TRAIN PRIOR", PRIOR_NAMES)],
      [field("test_prior", "TEST PRIOR", PRIOR_NAMES)]),
    panel("NETWORK",
      [uiField(uiLabel("HIDDEN LAYERS"), layerBox),
        field("activation", "ACTIVATION", ACTS.map((a) => [a, a]), "φ")],
      [widthsBox],
      ...(showOutput ? [[uiField(uiLabel("OUTPUT"), ctl.output), kField]] : [])),
    panel("TRAINING",
      [uiField(uiLabel("OPTIMISER"), ctl.optimiser),
        field("lr", "STEP SIZE", opts(LRS), "η"),
        field("batch_size", "BATCH", BATCHES),
        field("epochs", "EPOCHS", opts(EPOCHS))],
      [toggles.normalise]));

  // ---- the canvas and what sits on it -------------------------------------------------
  const body = elt("div");
  body.style.cssText = `position:relative;width:${W}px;height:${H}px;margin-top:4px;`;
  const cv = stageCanvas(st, W, H);
  const { canvas, ctx } = cv;
  body.append(canvas);
  st.stage.append(bar, scrub.el, panels, body);
  st.onScale(() => { cv.resize(); draw(); });
  const put = (e, x, y, align) => body.appendChild(uiPlace(e, x, y, align));
  const veil = elt("div", "loc-veil");  // dims the figure while training
  const veilText = elt("span"), veilBar = uiProgress();
  const veilPill = elt("div");
  veilPill.append(veilText, veilBar.el);
  veil.append(veilPill);
  body.append(veil);

  const arrowTgl = [0, 1].map((i) => {
    const b = elt("button", "loc-toggle", "arrows");
    b.onclick = () => { maps[i].arrows = !maps[i].arrows; syncHeads(); draw(); };
    return b;
  });
  const pickers = maps.map((m, i) => {
    const s = uiSelect(backgrounds.map((b) => [b, b]), m.bg, (v) => {
      m.bg = v; syncHeads(); draw();
    });
    put(elt("div", "mls-head", i ? "MAP B" : "MAP A"), m.l, 6);
    put(s, m.l + 62, 2);
    put(arrowTgl[i], m.l + 190, 2);
    return s;
  });
  function syncHeads() {
    maps.forEach((m, i) => {
      pickers[i].set(m.bg);
      arrowTgl[i].style.display = m.bg === "error" ? "" : "none";
      arrowTgl[i].classList.toggle("mls-on", m.arrows);
    });
  }
  put(elt("div", "mls-head", "LOCATION ERROR (km)"), LOSS.l - 24, 6);
  put(elt("div", "mls-head", "PREDICTED AGAINST TRUE (test events)"), SCAT.ls[0] - 24,
    SCAT.t - 38);

  // ---- inputs ----------------------------------------------------------------------------
  function setInput(name, value) {
    if (same(model.get(name), value)) return;
    model.set(name, value);
    model.save_changes();
    if (SCENE_KEYS.includes(name)) { waitScene = true; waitEvent = true; }
    if (name === "test_event") waitEvent = true;
    if (TRAIN_KEYS.includes(name)) scheduleAuto();
    syncControls();
    draw();
  }
  // automatic mode: train once a burst of changes is over, after the current run
  function scheduleAuto() {
    clearTimeout(autoTimer);
    autoTimer = setTimeout(() => {
      if (!auto || same(trainingSettings(), model.get("trained"))) return;
      if (estSeconds() > AUTO_MAX) return;  // too slow: wait for the button
      if (trainPending()) queued = true; else train();
    }, 150);
  }
  function trainingSettings() {
    const out = {};
    for (const k of TRAIN_KEYS) out[k] = JSON.parse(JSON.stringify(model.get(k)));
    return out;
  }
  function train() {
    model.set("trained", trainingSettings());
    model.set("train_id", model.get("train_id") + 1);
    model.save_changes();
    playing = false; autoplay = true; progress = 0;
    syncControls();
    draw();
  }
  function estSeconds() {  // fitted to presses timed on a laptop CPU
    const n = model.get("n_train"), bs = model.get("batch_size") || n;
    const steps = model.get("epochs") * Math.ceil(n / bs);
    const cost = COST[model.get("output")] || 1;
    return 0.4 + cost * (0.0009 * steps + 0.0003 * model.get("epochs"));
  }

  // stations: "+" takes the point of the reference sequence farthest from the
  // current stations, "-" removes the last one (and its mute flag)
  function setStations(dd) {
    const s = model.get("stations");
    if (dd > 0) {
      if (s.length >= model.get("max_stations")) return;
      let best = null, bd = -1;
      for (const p of model.get("reference_stations")) {
        const d = Math.min(...s.map((q) => Math.hypot(p[0] - q[0], p[1] - q[1])));
        if (d > bd) { bd = d; best = p; }
      }
      setInput("stations", [...s, [...best]]);
    } else {
      if (s.length <= model.get("min_stations")) return;
      model.set("muted", model.get("muted").filter((i) => i < s.length - 1));
      setInput("stations", s.slice(0, -1));
    }
  }
  function setK(dd) {
    const k = Math.min(MAXK, Math.max(1, model.get("n_components") + dd));
    setInput("n_components", k);
  }
  const widths = () => model.get("widths");
  function setDepth(dd) {
    const w = [...widths()], L = Math.min(MAXL, Math.max(0, w.length + dd));
    if (L === w.length) return;
    while (w.length < L) w.push(w.length ? w[w.length - 1] : 32);
    setInput("widths", w.slice(0, L));
  }
  function setWidth(l, v) {
    const w = [...widths()];
    w[l] = v;
    setInput("widths", w);
  }
  function buildWidths() {
    widthsBox.replaceChildren();
    const w = widths();
    if (!w.length) {
      widthsBox.append(elt("div", "loc-note", "no hidden layer: a linear map"));
      return;
    }
    w.forEach((n, l) => {
      const sel = uiSelect(opts(WIDTHS), n, (v) => setWidth(l, v));
      widthsBox.append(uiField(uiLabel(`WIDTH ${l + 1}`), sel));
    });
  }
  function syncControls() {
    for (const [k, c] of Object.entries(ctl)) c.set(model.get(k));
    for (const t of Object.values(toggles)) t.sync();
    fieldF.setOff(model.get("velocity") !== "heterogeneous");
    const n = model.get("stations").length;
    stationNum.textContent = String(n);
    stationPM.enable(n < model.get("max_stations"), n > model.get("min_stations"));
    const K = model.get("n_components");
    kNum.textContent = String(K);
    kPM.enable(K < MAXK, K > 1);
    kField.style.visibility = model.get("output") === "mixture" ? "" : "hidden";
    const L = widths().length;
    layerNum.textContent = String(L);
    layerPM.enable(L < MAXL, L > 0);
    buildWidths();
    const dirty = !same(trainingSettings(), model.get("trained"));
    trainBtn.classList.toggle("loc-dirty", dirty);
    trainBtn.title = dirty ? "settings changed since the last training" : "train again";
    syncBusy();
    updateStatus();
  }
  function syncBusy() {  // the train button, the veil and the sweeping bar
    const busy = trainPending();
    trainBtn.classList.toggle("loc-working", busy);
    if (busy) trainBtn.innerHTML = `<span class="loc-spin"></span>TRAINING<small></small>`;
    else {
      const s = estSeconds();
      trainBtn.innerHTML = `TRAIN<small>≈ ${s < 10 ? s.toFixed(1) : Math.round(s)} s</small>`;
    }
    veil.classList.toggle("on", busy);
    sweep.classList.toggle("on", busy || waitScene || waitEvent);
    syncProgress();
  }
  function syncProgress() {  // the percentage in the button and in the veil
    const pct = progress > 0 ? `${Math.round(100 * progress)} %` : "";
    veilText.textContent = pct ? `TRAINING … ${pct}` : "TRAINING …";
    veilBar.set(progress);
    const note = trainBtn.querySelector("small");
    if (note && trainPending()) note.textContent = pct || "please wait";
  }
  function updateStatus() {
    const r = run();
    const spin = (dark) => `<span class="loc-spin${dark ? " loc-dark" : ""}"></span>`;
    status.style.color = C.ink2;
    if (trainPending()) { status.innerHTML = `${spin(true)}training…`; status.style.color = C.orange; }
    else if (waitScene || waitEvent) status.innerHTML = `${spin(true)}updating…`;
    else if (r.diverged) { status.textContent = "diverged: lower the step size"; status.style.color = C.orange; }
    else if (r.seconds !== undefined) status.textContent = `trained in ${r.seconds} s`;
    else status.textContent = "";
    syncBusy();
  }

  // ---- playback ---------------------------------------------------------------------
  const E = () => run().snapshot_epochs || [];
  const lastEpoch = () => (E().length ? E().at(-1) : 0);
  const syncPlay = () => tp.setPlaying(playing || (trainPending() && autoplay));
  function framePos() {  // the snapshots either side of the replayed epoch
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
  }
  function frameOf(list) {  // a per-snapshot array, interpolated at the replay epoch
    if (!Array.isArray(list) || !list.length) return null;
    const [k0, k1, t] = framePos();
    const a = list[Math.min(k0, list.length - 1)], b = list[Math.min(k1, list.length - 1)];
    if (!a) return null;
    if (t === 0 || !b || a.length !== b.length) return a;
    const mix = (u, w) => (u === null || w === null ? null : lerp(u, w, t));
    return a.map((v, i) => (Array.isArray(v) ? v.map((u, j) => mix(u, b[i][j]))
      : mix(v, b[i])));
  }
  function setEpoch(e) {
    epoch = Math.min(Math.max(e, 0), lastEpoch());
    draw();
  }
  function tick(now) {
    if (!playing) return;
    const dt = Math.min(now - lastT, 100);
    lastT = now;
    epoch = Math.min(epoch + lastEpoch() * dt / (1000 * duration), lastEpoch());
    if (epoch >= lastEpoch()) { playing = false; syncPlay(); }
    draw();
    if (playing) requestAnimationFrame(tick);
  }
  function play() {
    autoplay = true;
    if (!trainPending() && lastEpoch() > 0) {
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
  function togglePlay() { if (playing || (trainPending() && autoplay)) pause(); else play(); }
  function stepOnce() {
    pause();
    const next = E().find((e) => e > epoch + 1e-6);
    setEpoch(next === undefined ? lastEpoch() : next);
  }
  function rewind() { playing = false; setEpoch(0); if (autoplay) play(); else syncPlay(); }
  function newRun() {
    if (trainPending()) { updateStatus(); draw(); return; }
    playing = false;
    epoch = autoplay ? 0 : lastEpoch();
    if (autoplay) play(); else syncPlay();
    syncControls();
    draw();
    if (queued) { queued = false; scheduleAuto(); }
  }

  // ---- coordinates and marks ---------------------------------------------------------------
  const toPx = ([x, y]) => (view
    ? [view.l + (x - view.x0) / view.s * view.p,
      view.t + view.p - (y - view.y0) / view.s * view.p]
    : [cur.l + x / box() * P, cur.t + P - y / box() * P]);
  const pxPerKm = () => (view ? view.p / view.s : P / box());
  const pxOn = (p, map) => {  // pixel position of a point on a given map
    const old = cur; cur = map;
    const r = toPx(p);
    cur = old;
    return r;
  };
  const kmOf = (px, py, map) => [(px - map.l) / P * box(), (map.t + P - py) / P * box()];
  const clampKm = (v) => Math.min(box(), Math.max(0, v));
  const mapAt = (px, py) => maps.find((m) => px >= m.l && px <= m.l + P &&
    py >= m.t && py <= m.t + P) || null;
  function text(s, x, y, align = "left", colour = C.ink, size = 11, font = FONT,
    style = "", halo = false) {
    ctx.font = `${style ? style + " " : ""}${+(size * FS).toFixed(1)}px ${font}`;
    ctx.fillStyle = colour; ctx.textAlign = align;
    if (halo) {
      ctx.lineWidth = 3.5; ctx.strokeStyle = "#fff"; ctx.lineJoin = "round";
      ctx.strokeText(s, x, y);
    }
    ctx.fillText(s, x, y);
  }
  function star(x, y, R, fill) {
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const a = -Math.PI / 2 + i * Math.PI / 5, r = i % 2 ? R * 0.42 : R;
      ctx[i ? "lineTo" : "moveTo"](x + r * Math.cos(a), y + r * Math.sin(a));
    }
    ctx.closePath(); ctx.fillStyle = fill; ctx.fill();
    ctx.lineWidth = 1.2; ctx.strokeStyle = "#fff"; ctx.stroke();
  }
  function triangle(x, y, s, fill, stroke = "#222", dash = []) {
    ctx.beginPath();
    ctx.moveTo(x, y - 1.1 * s); ctx.lineTo(x + s, y + 0.8 * s); ctx.lineTo(x - s, y + 0.8 * s);
    ctx.closePath();
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    ctx.setLineDash(dash); ctx.lineWidth = 1; ctx.strokeStyle = stroke; ctx.stroke();
    ctx.setLineDash([]);
  }
  function diamond(x, y, r, fill) {
    ctx.beginPath();
    ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y);
    ctx.closePath(); ctx.fillStyle = fill; ctx.fill();
    ctx.lineWidth = 1.2; ctx.strokeStyle = "#fff"; ctx.stroke();
  }
  function dot(x, y, r, fill, stroke = "#fff", lw = 1.2) {
    ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fillStyle = fill; ctx.fill();
    if (stroke) { ctx.lineWidth = lw; ctx.strokeStyle = stroke; ctx.stroke(); }
  }
  function cross(x, y, r) {
    ctx.beginPath();
    ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r);
    ctx.moveTo(x - r, y + r); ctx.lineTo(x + r, y - r); ctx.stroke();
  }

  function scene() {  // the editable things, or their live copy while dragging
    return live || { stations: model.get("stations"), event: model.get("test_event") };
  }
  const muted = () => new Set(model.get("muted"));

  // the network's answer for a source at (x, y) from the grid of predictions
  function gridLookup(gp, [x, y]) {
    const n = model.get("n_grid"), lo = 2, hi = box() - 2;
    const fx = Math.min(Math.max((x - lo) / (hi - lo) * (n - 1), 0), n - 1);
    const fy = Math.min(Math.max((y - lo) / (hi - lo) * (n - 1), 0), n - 1);
    const i0 = Math.min(Math.floor(fx), n - 2), j0 = Math.min(Math.floor(fy), n - 2);
    const tx = fx - i0, ty = fy - j0;
    const at = (i, j, c) => gp[2 * (j * n + i) + c];
    return [0, 1].map((c) => lerp(lerp(at(i0, j0, c), at(i0 + 1, j0, c), tx),
      lerp(at(i0, j0 + 1, c), at(i0 + 1, j0 + 1, c), tx), ty));
  }

  // ---- background layers ------------------------------------------------------------------
  function image(layer) {
    if (!layer) return null;
    if (!images[layer.png]) {
      const img = new Image(), barImg = new Image();
      img.onload = barImg.onload = () => draw();
      img.src = layer.png; barImg.src = layer.bar;
      images[layer.png] = { img, bar: barImg };
    }
    return images[layer.png];
  }
  const errCanvas = document.createElement("canvas");
  function errorImage(gp) {
    const n = model.get("n_grid");
    errCanvas.width = n; errCanvas.height = n;
    const g = errCanvas.getContext("2d"), im = g.createImageData(n, n);
    const lo = 2, hi = box() - 2;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const x = lo + (hi - lo) * i / (n - 1), y = lo + (hi - lo) * j / (n - 1);
      const k = j * n + i;
      const e = Math.hypot(gp[2 * k] - x, gp[2 * k + 1] - y);
      const [r, gg, b] = errColour(e / ERR_MAX);
      const p = 4 * ((n - 1 - j) * n + i);  // image rows run top down
      im.data[p] = r; im.data[p + 1] = gg; im.data[p + 2] = b; im.data[p + 3] = 255;
    }
    g.putImageData(im, 0, 0);
    return errCanvas;
  }
  function colourBar(map, drawBar, lo, hi, label, ticks) {
    const bx = map.l + P + 10, bw = 10, by = map.t, bh = P;
    drawBar(bx, by, bw, bh);
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 0.8; ctx.strokeRect(bx, by, bw, bh);
    const ts = ticks || [lo, (lo + hi) / 2, hi];
    for (const v of ts) {
      const f = (v - lo) / (hi - lo), y = by + bh * (1 - f);
      const s = Math.abs(v) >= 10 || Number.isInteger(v) ? String(+v.toFixed(1)) : v.toFixed(1);
      text(s, bx + bw + 4, y + (f <= 0 ? 0 : f >= 1 ? 8 : 4));
      ctx.beginPath(); ctx.moveTo(bx + bw, y); ctx.lineTo(bx + bw - 3, y); ctx.stroke();
    }
    ctx.save(); ctx.translate(bx + bw + 34, by + bh / 2); ctx.rotate(Math.PI / 2);
    text(label, 0, 0, "center", C.ink2, 11); ctx.restore();
  }
  const layerOf = (map) => {
    const s = sv(), e = ev();
    return { "P speed": s.layers?.speed, prior: s.layers?.prior, posterior: e.posterior,
      likelihood: e.likelihood, network: e.learned, compare: e.posterior }[map.bg];
  };
  function drawBackground(map, gp) {
    const layer = layerOf(map);
    const isEvent = isEventBg(map.bg);
    const stale = (isEvent && (waitEvent || drag?.kind === "event"))
      || (!isEvent && waitScene);
    ctx.globalAlpha = stale ? 0.35 : 1;
    if (map.bg === "error") {
      if (gp) {
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(errorImage(gp), map.l, map.t, P, P);
      } else { ctx.fillStyle = "#f3f4f6"; ctx.fillRect(map.l, map.t, P, P); }
      ctx.globalAlpha = 1;
      colourBar(map, (bx, by, bw, bh) => {
        for (let k = 0; k < bh; k++) {
          const [r, g, b] = errColour(1 - k / bh);
          ctx.fillStyle = `rgb(${r},${g},${b})`; ctx.fillRect(bx, by + k, bw, 1.5);
        }
      }, 0, ERR_MAX, "error of the network (km)", [0, 5, 10]);
      return;
    }
    if (map.bg === "P speed" && layer?.constant) {  // nothing to colour
      ctx.fillStyle = "#eef0f3"; ctx.fillRect(map.l, map.t, P, P);
      ctx.globalAlpha = 1;
      text(`P speed constant, ${layer.v_ref} km/s`, map.l + 8, map.t + 16, "left", C.ink2, 11);
      return;
    }
    const im = image(layer);
    if (im && im.img.complete && im.img.naturalWidth) {
      ctx.imageSmoothingEnabled = true; ctx.drawImage(im.img, map.l, map.t, P, P);
    } else { ctx.fillStyle = "#f3f4f6"; ctx.fillRect(map.l, map.t, P, P); }
    ctx.globalAlpha = 1;
    if (layer) colourBar(map, (bx, by, bw, bh) => {
      if (im && im.bar.complete && im.bar.naturalWidth) {
        ctx.save(); ctx.translate(bx, by + bh); ctx.rotate(-Math.PI / 2);
        ctx.drawImage(im.bar, 0, 0, bh, bw); ctx.restore();
      }
    }, layer.lo, layer.hi, map.bg === "compare" ? "exact posterior / max" : layer.label,
    layer.ticks);
  }

  // ---- one map --------------------------------------------------------------------------------
  function polyline(points) {
    ctx.beginPath();
    points.forEach((p, i) => { const [x, y] = toPx(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
  }
  // rays from the test event to the stations: the model of the data (solid) and,
  // if the analyst assumes another one, its rays (dashed)
  function drawRays() {
    const e = ev(), old = waitEvent || drag?.kind === "event";
    if (!e.rays) return;
    ctx.globalAlpha = old ? 0.25 : 1;
    ctx.lineWidth = 0.9;
    if (e.rays_assumed) {
      ctx.strokeStyle = C.ink2; ctx.setLineDash([4, 3]);
      for (const r of e.rays_assumed) if (r) polyline(r);
      ctx.setLineDash([]);
    }
    ctx.strokeStyle = "rgba(17,17,17,0.7)";
    for (const r of e.rays) if (r) polyline(r);
    ctx.globalAlpha = 1;
  }
  function drawMap(map) {
    cur = map;
    const s = scene(), r = run(), sView = sv();
    const gp = sView.ok && !trainPending() ? frameOf(sView.grid_pred) : null;
    drawBackground(map, gp);
    ctx.save(); ctx.beginPath(); ctx.rect(map.l, map.t, P, P); ctx.clip();
    if (map.bg === "prior" && r.train_events) {  // where the training events were
      for (const p of r.train_events) {  // dark dots with a light rim, visible on any blue
        const [x, y] = toPx(p);
        dot(x, y, 1.7, "rgba(17,17,17,0.8)", "rgba(255,255,255,0.7)", 0.7);
      }
      if (sView.test_true && !waitScene) {  // the test events: open rings
        ctx.lineWidth = 1; ctx.strokeStyle = C.orange;
        for (const p of sView.test_true) {
          const [x, y] = toPx(p); ctx.beginPath(); ctx.arc(x, y, 2.6, 0, 7); ctx.stroke();
        }
      }
      const ly = map.t + P - 30, lx = map.l + 8;  // legend
      ctx.fillStyle = "rgba(255,255,255,0.85)"; ctx.fillRect(lx - 4, ly - 12, 92, 38);
      dot(lx + 3, ly - 2, 1.7, "rgba(17,17,17,0.8)", "rgba(255,255,255,0.7)", 0.7);
      text("training", lx + 12, ly + 1, "left", C.ink, 10);
      ctx.lineWidth = 1; ctx.strokeStyle = C.orange;
      ctx.beginPath(); ctx.arc(lx + 3, ly + 13, 2.6, 0, 7); ctx.stroke();
      text("test", lx + 12, ly + 16, "left", C.ink, 10);
    }
    if (map.bg === "P speed") drawRays();
    if (map.bg === "error" && map.arrows && gp) {  // from each true source to m̂
      const n = model.get("n_grid"), lo = 2, hi = box() - 2;
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1;
      for (let j = 0; j < n; j += 2) for (let i = 0; i < n; i += 2) {
        const k = j * n + i;
        const a = toPx([lo + (hi - lo) * i / (n - 1), lo + (hi - lo) * j / (n - 1)]);
        const b = toPx([gp[2 * k], gp[2 * k + 1]]);
        ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
        dot(b[0], b[1], 1.6, C.ink, null);
      }
    }
    const isEvent = isEventBg(map.bg);
    const post = layerOf(map);
    if (isEvent) drawPosteriorMarks(post, map.bg);
    if (map.bg === "network") drawNetworkMarks(map, post);
    const tr = r.config?.stations;  // ghosts: where the network was trained
    if (tr && !same(tr, s.stations)) {
      for (const p of tr) { const [x, y] = toPx(p); triangle(x, y, 5, null, C.ink2, [2, 2]); }
    }
    const mut = muted();
    s.stations.forEach((p, i) => {
      const [x, y] = toPx(p);
      if (mut.has(i)) {
        triangle(x, y, 5.5, "#e5e7eb", C.ink2);
        ctx.strokeStyle = C.orange; ctx.lineWidth = 1.6; cross(x, y + 1, 4);
      } else triangle(x, y, 5.5, "#fff");
    });
    const cloud = drawEventMarks(s.event, gp);
    const target = drag || hover;
    if (target) {
      const p = target.kind === "station" ? s.stations[target.index] : s.event;
      const [x, y] = toPx(p);
      ctx.beginPath(); ctx.arc(x, y, 13, 0, 7);
      ctx.lineWidth = 1.5; ctx.strokeStyle = C.blue; ctx.stroke();
      if (drag && drag.map === map) {
        const right = x > map.l + P / 2;
        text(`(${p[0].toFixed(1)}, ${p[1].toFixed(1)}) km`, x + (right ? -18 : 18), y - 14,
          right ? "right" : "left", C.ink, 11, FONT, "", true);
      }
    }
    if (isEvent) drawInset(map, s.event, post, cloud, gp);
    if (map.bg === "network" && !post && !trainPending() && !waitEvent && !drag) {
      ctx.fillStyle = "rgba(255,255,255,0.85)";
      ctx.fillRect(map.l + 14, map.t + P / 2 - 28, P - 28, 52);
      text("one point out: no distribution to show.", map.l + P / 2, map.t + P / 2 - 6,
        "center", C.ink, 11.5);
      text("Set OUTPUT to mixture.", map.l + P / 2, map.t + P / 2 + 11,
        "center", C.ink2, 11.5);
    }
    if (sView.ok === false && !trainPending()) {
      ctx.fillStyle = "rgba(255,255,255,0.85)";
      ctx.fillRect(map.l + 14, map.t + P / 2 - 28, P - 28, 52);
      const words = (sView.message || "").split(": ");
      words.forEach((w, i) => text(i < words.length - 1 ? `${w}:` : w, map.l + P / 2,
        map.t + P / 2 - 6 + 17 * i, "center", C.orange, 11.5));
    }
    ctx.restore();

    // frame, ticks and labels (boxed look)
    const differs = differences().length > 0;
    ctx.strokeStyle = differs ? C.orange : C.ink2; ctx.lineWidth = differs ? 2.5 : 1;
    ctx.strokeRect(map.l, map.t, P, P);
    for (let v = 0; v <= box(); v += 20) {
      const [x, y] = toPx([v, v]);
      text(String(v), x, map.t + P + 14, "center");
      text(String(v), map.l - 6, y + 4, "right");
      ctx.beginPath();
      ctx.moveTo(x, map.t + P); ctx.lineTo(x, map.t + P - 4);
      ctx.moveTo(map.l, y); ctx.lineTo(map.l + 4, y); ctx.stroke();
    }
    text("x (km)", map.l + P / 2, map.t + P + 30, "center");
    if (map === maps[0]) {  // both maps share the y axis
      ctx.save(); ctx.translate(map.l - 32, map.t + P / 2); ctx.rotate(-Math.PI / 2);
      text("y (km)", 0, 0, "center"); ctx.restore();
    }
  }
  // the posterior's 68 % and 95 % regions and its mean (the network's in dark orange)
  function drawRegions(layer, ink, lws, halo = false) {
    ctx.lineJoin = "round";
    for (const [lev, dash, lw] of [["95", [4, 3], lws[1]], ["68", [], lws[0]]]) {
      ctx.setLineDash(dash);
      for (const [colour, width] of halo ? [["rgba(255,255,255,0.9)", lw + 2.4], [ink, lw]]
        : [[ink, lw]]) {
        ctx.strokeStyle = colour; ctx.lineWidth = width;
        for (const line of layer.contours[lev]) polyline(line);
      }
    }
    ctx.setLineDash([]);
  }
  function drawPosteriorMarks(post, bg) {
    if (!post || waitEvent || drag) return;
    if (bg === "compare") {  // exact posterior shaded underneath, the network on top
      drawRegions(post, EXACT_INK, [1, 0.8]);
      const learned = ev().learned;
      if (learned) drawRegions(learned, C.orange, [2.2, 1.8], true);
      return;
    }
    const net = bg === "network";
    drawRegions(post, net ? NET_INK : EXACT_INK, [1.6, 1.2]);
    if (net) return;  // the network's mean is its answer, the orange dot
    const [mx, my] = toPx(post.mean);
    diamond(mx, my, 6, C.green);
  }
  // the components of a mixture, each with its weight
  function drawNetworkMarks(map, post) {
    if (!post || waitEvent || drag) return;
    const k = pxPerKm();
    for (const c of post.components || []) {
      const [[sxx, sxy], [, syy]] = c.cov;
      const tr = (sxx + syy) / 2, det = Math.sqrt(((sxx - syy) / 2) ** 2 + sxy ** 2);
      const l1 = tr + det, l2 = Math.max(tr - det, 0);
      const ang = Math.atan2(l1 - sxx, sxy || 1e-12);
      const [cx, cy] = toPx(c.mean);
      ctx.save(); ctx.translate(cx, cy); ctx.rotate(-ang);
      ctx.beginPath(); ctx.ellipse(0, 0, Math.sqrt(l1) * k, Math.sqrt(l2) * k, 0, 0, 7);
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1; ctx.setLineDash([3, 2]); ctx.stroke();
      ctx.setLineDash([]); ctx.restore();
      if (c.w >= 0.01) text(c.w.toFixed(2), cx + 6, cy - 6, "left", C.ink, 10, FONT, "", true);
    }
  }
  // the test event, the network's answer and the scatter over noise draws;
  // returns the scatter (null while stale)
  function drawEventMarks(event, gp) {
    const eView = ev();
    const [ex, ey] = toPx(event);
    let hat = null, cloud = null;
    if (eView.ok && !trainPending() && !waitEvent && !drag) {
      cloud = frameOf(eView.cloud);
      if (cloud && !cloud.every((p) => fin(p[0]) && fin(p[1]))) cloud = null;  // diverged
      if (cloud && !mixtureOut()) {
        ctx.fillStyle = "rgba(213,94,0,0.45)";
        for (const p of cloud) { const [x, y] = toPx(p); ctx.beginPath(); ctx.arc(x, y, 2, 0, 7); ctx.fill(); }
        drawEllipse(cloud);
        hat = cloud[0];
      }
    } else if (gp && drag?.kind === "event" && !mixtureOut()) hat = gridLookup(gp, event);
    if (hat) {
      const [hx, hy] = toPx(hat);
      ctx.strokeStyle = C.orange; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(hx, hy); ctx.stroke();
      dot(hx, hy, 5.5, C.orange);
    }
    star(ex, ey, view ? 8 : 10, "#111");
    return cloud;
  }
  // A magnified window around the test event, when its posterior and scatter
  // are too small to read on the map; placed in the corner away from the event.
  function drawInset(map, event, post, cloud, gp) {
    const points = mixtureOut() ? [] : cloud;
    if (!post || (!cloud && !mixtureOut()) || waitEvent || drag) return;
    let far = 0;
    const reach = (p) => { far = Math.max(far, Math.abs(p[0] - event[0]), Math.abs(p[1] - event[1])); };
    for (const line of post.contours["95"]) line.forEach(reach);
    if (map.bg === "compare" && ev().learned)
      for (const line of ev().learned.contours["95"]) line.forEach(reach);
    points.forEach(reach); reach(post.mean);
    const half = Math.max(1.25 * far, 2.5);
    if (half > 14) return;  // big enough on the map itself
    const side = 140, pad = 8;
    const l = event[0] > box() / 2 ? map.l + pad : map.l + P - side - pad;
    const t = event[1] > box() / 2 ? map.t + P - side - pad : map.t + pad;
    const a = toPx([event[0] - half, event[1] - half]), b = toPx([event[0] + half, event[1] + half]);
    ctx.strokeStyle = C.ink; ctx.lineWidth = 0.8; ctx.setLineDash([2, 2]);
    ctx.strokeRect(a[0], b[1], b[0] - a[0], a[1] - b[1]); ctx.setLineDash([]);
    view = { x0: event[0] - half, y0: event[1] - half, s: 2 * half, l, t, p: side };
    ctx.save();
    ctx.fillStyle = C.bg; ctx.fillRect(l, t, side, side);
    ctx.beginPath(); ctx.rect(l, t, side, side); ctx.clip();
    const im = image(post);
    if (im && im.img.complete && im.img.naturalWidth) {  // the posterior, cropped
      const n = im.img.naturalWidth, f = n / box();
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(im.img, view.x0 * f, (box() - view.y0 - view.s) * f, view.s * f,
        view.s * f, l, t, side, side);
    }
    drawPosteriorMarks(post, map.bg);
    if (map.bg === "network") drawNetworkMarks(map, post);
    drawEventMarks(event, gp);
    ctx.restore();
    view = null;
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1; ctx.strokeRect(l, t, side, side);
    const km = 2 * half < 10 ? (2 * half).toFixed(1) : Math.round(2 * half);
    text(`${km} km`, l + side - 5, t + side - 6, "right", C.ink, 10, FONT, "", true);
  }
  function drawEllipse(cloud) {  // 1-standard-deviation ellipse of the scatter
    const n = cloud.length;
    if (n < 3) return;
    const mx = cloud.reduce((a, p) => a + p[0], 0) / n;
    const my = cloud.reduce((a, p) => a + p[1], 0) / n;
    let sxx = 0, syy = 0, sxy = 0;
    for (const [x, y] of cloud) { sxx += (x - mx) ** 2; syy += (y - my) ** 2; sxy += (x - mx) * (y - my); }
    sxx /= n - 1; syy /= n - 1; sxy /= n - 1;
    const tr = (sxx + syy) / 2, det = Math.sqrt(((sxx - syy) / 2) ** 2 + sxy ** 2);
    const l1 = tr + det, l2 = Math.max(tr - det, 0);
    const ang = Math.atan2(l1 - sxx, sxy || 1e-12);
    const k = pxPerKm();
    const [cx, cy] = toPx([mx, my]);
    ctx.save(); ctx.translate(cx, cy); ctx.rotate(-ang);
    ctx.beginPath(); ctx.ellipse(0, 0, Math.sqrt(l1) * k, Math.sqrt(l2) * k, 0, 0, 7);
    ctx.strokeStyle = C.orange; ctx.lineWidth = 1.2; ctx.setLineDash([3, 2]); ctx.stroke();
    ctx.setLineDash([]); ctx.restore();
  }

  function drawKey() {  // one line of markers under the maps, then the hints
    let x = maps[0].l;
    const y = maps[0].t + P + 108;
    const item = (mark, s) => {
      mark(x + 7, y - 4);
      text(s, x + 18, y, "left", C.ink, 11);
      ctx.font = `${11 * FS}px ${FONT}`;
      x += 18 + ctx.measureText(s).width + 14;
    };
    const line = (colour, width, dash = []) => (a, b) => {
      ctx.strokeStyle = colour; ctx.lineWidth = width; ctx.setLineDash(dash);
      ctx.beginPath(); ctx.moveTo(a - 7, b); ctx.lineTo(a + 7, b); ctx.stroke();
      ctx.setLineDash([]);
    };
    const has = (...bgs) => maps.some((m) => bgs.includes(m.bg));
    item((a, b) => star(a, b, 7, "#111"), "test event");
    if (!mixtureOut()) {
      item((a, b) => dot(a, b, 4.5, C.orange), "network");
      item((a, b) => { ctx.fillStyle = "rgba(213,94,0,0.6)"; for (const [u, v] of [[-4, 1], [0, -3], [3, 2]]) ctx.fillRect(a + u, b + v, 2, 2); }, "64 noise draws");
    }
    if (has("posterior", "likelihood"))
      item((a, b) => diamond(a, b, 5, C.green), "posterior mean");
    if (has("compare")) {
      item((a, b) => { ctx.fillStyle = "#8fd0b5"; ctx.fillRect(a - 7, b - 5, 14, 10); }, "exact posterior");
      if (mixtureOut()) item(line(C.orange, 2.2), "network 68 %");
      if (mixtureOut()) item(line(C.orange, 1.8, [4, 3]), "95 %");
    }
    item((a, b) => triangle(a, b, 4.5, "#fff"), "station");
    item((a, b) => { ctx.strokeStyle = "rgba(17,17,17,0.7)"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(a - 7, b); ctx.lineTo(a + 7, b); ctx.stroke(); }, "ray");
    if (ev().rays_assumed)
      item((a, b) => { ctx.strokeStyle = C.ink2; ctx.lineWidth = 1; ctx.setLineDash([3, 2]); ctx.beginPath(); ctx.moveTo(a - 7, b); ctx.lineTo(a + 7, b); ctx.stroke(); ctx.setLineDash([]); }, "assumed ray");
    text("drag the star and the stations · double-click: add or remove a station · " +
      "shift-click: mute it", maps[0].l, y + 20, "left", C.ink2, 10.5);
  }

  // what differs from the training: the data (scene) and the settings of the network
  const SCENE_ONLY = ["stations", "muted", "velocity", "field", "sigma"];
  const NAMES = { n_train: "training events", train_prior: "train prior", widths: "widths",
    lr: "step size", batch_size: "batch", normalise: "rescale",
    output: "output", n_components: "components K" };
  function differences() {
    const c = model.get("trained");
    if (!c || !c.stations) return [];
    const out = [], s = model.get("stations");
    // in automatic mode the network follows every change, so only the test prior can differ
    if (!(auto && estSeconds() <= AUTO_MAX)) {
      if (s.length !== c.stations.length) out.push(`${s.length} stations (trained: ${c.stations.length})`);
      else {
        const moved = s.filter((p, i) => Math.hypot(p[0] - c.stations[i][0], p[1] - c.stations[i][1]) > 0.05).length;
        if (moved) out.push(`${moved} station${moved > 1 ? "s" : ""} moved`);
      }
      const mNow = [...model.get("muted")].sort(), mTr = [...c.muted].sort();
      if (!same(mNow, mTr)) out.push(mNow.length ? `muted: ${mNow.map((i) => i + 1).join(", ")}` : "muting changed");
      const vNow = model.get("velocity"), fNow = model.get("field");
      if (vNow !== c.velocity || (vNow === "heterogeneous" && fNow !== c.field))
        out.push(`P speed ${vNow === "heterogeneous" ? `field ${fNow}` : "constant"} (trained: ${c.velocity === "heterogeneous" ? `field ${c.field}` : "constant"})`);
      if (model.get("sigma") !== c.sigma) out.push(`σ = ${model.get("sigma")} s (trained: ${c.sigma} s)`);
    }
    if (model.get("test_prior") !== c.train_prior)
      out.push(`test events from the ${model.get("test_prior")} prior`);
    if (!(auto && estSeconds() <= AUTO_MAX)) {
      const retrain = TRAIN_KEYS.filter((k) => !SCENE_ONLY.includes(k) && !same(model.get(k), c[k]));
      if (retrain.length) {
        const names = retrain.map((k) => NAMES[k] || k).join(", ");
        out.push(auto ? `${names} changed: too slow to train automatically, press TRAIN`
          : `${names} changed: press TRAIN`);
      }
    }
    return out;
  }
  function drawBanner() {  // a band under the maps: loud when something differs
    const out = differences();
    const x = maps[0].l, y = maps[0].t + P + 42, w = maps[1].l + P - x;
    if (!out.length) {
      text("✓ data and settings match the training", x, y + 19, "left", C.ink2, 11);
      return;
    }
    const label = "DIFFERS FROM THE TRAINING:";
    ctx.font = `bold ${11 * FS}px ${FONT}`;
    const x0 = x + 44 + ctx.measureText(label).width + 10, avail = x + w - x0 - 10;
    // pack the phrases into at most two lines of readable size
    const wide = (str, size) => { ctx.font = `${size * FS}px ${FONT}`; return ctx.measureText(str).width; };
    const pack = (size) => {
      const lines = [""];
      for (const item of out) {
        const joined = lines.at(-1) ? `${lines.at(-1)}  ·  ${item}` : item;
        if (lines.at(-1) && wide(joined, size) > avail) lines.push(item);
        else lines[lines.length - 1] = joined;
      }
      return lines;
    };
    let size = 12.5, lines = pack(size);
    while (lines.length > 2 && size > 10) { size -= 0.5; lines = pack(size); }
    const h = lines.length > 1 ? 48 : 30, mid = y + h / 2;
    ctx.fillStyle = "rgba(213,94,0,0.14)"; ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = C.orange; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
    ctx.fillStyle = C.orange; ctx.fillRect(x, y, 5, h);
    ctx.beginPath();  // a warning triangle with "!"
    ctx.moveTo(x + 24, mid - 9); ctx.lineTo(x + 33, mid + 8); ctx.lineTo(x + 15, mid + 8); ctx.closePath();
    ctx.fill();
    text("!", x + 24, mid + 6, "center", "#fff", 12, FONT, "bold");
    text(label, x + 44, y + 19, "left", C.orange, 11, FONT, "bold");
    lines.forEach((line, k) => text(line, x0, y + 19 + 17 * k, "left", C.ink, size));
  }

  // ---- learning curves ------------------------------------------------------------------
  function drawLoss() {
    const r = run(), tr = r.rms_train || [], va = r.rms_val || [];
    const { l, t, w, h } = LOSS;
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(l, t + h); ctx.lineTo(l + w, t + h); ctx.stroke();
    const n = Math.max(tr.length - 1, 1), nTotal = r.config ? r.config.epochs : n;
    const ok = [...tr, ...va, r.floor].filter((v) => v > 0 && fin(v));
    if (!ok.length) return;
    // the first epochs start far above the rest and would stretch the axis: the limits come
    // from the curves after the first fifth of the epochs (the start runs off the top), the
    // lower one just below the training loss (the best possible line may fall outside)
    const skip = Math.ceil(0.2 * tr.length);
    const fit = (vals) => vals.slice(skip).filter((v) => v > 0 && fin(v));
    const rest = [...fit(tr), ...fit(va)];
    const trainRest = fit(tr);
    const low = trainRest.length ? Math.min(...trainRest) : Math.min(...ok);
    const top = rest.length ? Math.max(...rest, r.floor > 0 ? r.floor : 0) : Math.max(...ok);
    const lo = Math.log10(Math.min(low, ...(rest.length ? rest : ok))) - 0.04;
    const hi = Math.log10(Math.min(top, 300)) + 0.06;
    const X = (k) => l + k / nTotal * w;
    const Y = (v) => t + h - (Math.log10(Math.min(Math.max(v, 1e-6), 1e6)) - lo) / (hi - lo) * h;
    const ticks = [];
    for (let e = Math.floor(lo); e <= Math.ceil(hi); e++)
      for (const m of [1, 2, 5]) { const v = m * 10 ** e; if (Math.log10(v) >= lo && Math.log10(v) <= hi) ticks.push(v); }
    for (const v of ticks) {
      ctx.strokeStyle = C.grid; ctx.lineWidth = 0.8;
      ctx.beginPath(); ctx.moveTo(l, Y(v)); ctx.lineTo(l + w, Y(v)); ctx.stroke();
      text(String(v), l - 6, Y(v) + 4, "right", C.ink, 10);
    }
    const dx = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000].find((d) => nTotal / d <= 5) || 1000;
    for (let k = 0; k <= nTotal; k += dx) text(String(k), X(k), t + h + 14, "center", C.ink, 10);
    text("epoch", l + w / 2, t + h + 28, "center", C.ink2, 10.5);
    ctx.save(); ctx.beginPath(); ctx.rect(l, t - 4, w + 4, h + 8); ctx.clip();
    const stale = trainPending();
    ctx.globalAlpha = stale ? 0.3 : 1;
    if (r.floor > 0) {
      ctx.setLineDash([5, 4]); ctx.strokeStyle = C.green; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(l, Y(r.floor)); ctx.lineTo(l + w, Y(r.floor)); ctx.stroke();
      ctx.setLineDash([]);
    }
    const upto = stale ? tr.length - 1 : Math.min(Math.floor(epoch), tr.length - 1);
    const curve = (vals, colour) => {
      ctx.strokeStyle = colour; ctx.lineWidth = 1.7; ctx.lineJoin = "round"; ctx.beginPath();
      for (let k = 0; k <= upto; k++) { const v = vals[k]; if (!fin(v)) break; k ? ctx.lineTo(X(k), Y(v)) : ctx.moveTo(X(k), Y(v)); }
      ctx.stroke();
    };
    curve(tr, C.blue); curve(va, C.orange);
    if (!stale && r.best_epoch !== undefined && epoch >= r.best_epoch && va.length) {
      const [x, y] = [X(r.best_epoch), Y(va[r.best_epoch])];
      ctx.fillStyle = C.orange; ctx.beginPath();
      ctx.moveTo(x, y - 5); ctx.lineTo(x - 5, y - 13); ctx.lineTo(x + 5, y - 13); ctx.closePath(); ctx.fill();
    }
    if (!stale && upto >= 0 && va.length) {
      if (fin(tr[upto])) dot(X(upto), Y(tr[upto]), 3.5, C.blue);
      if (fin(va[upto])) dot(X(upto), Y(va[upto]), 3.5, C.orange);
    }
    ctx.restore();
    ctx.globalAlpha = 1;
    let kx = l + w;  // key above the plot
    for (const [name, colour, dash] of [["best possible", C.green, [4, 3]], ["validation", C.orange, []], ["training", C.blue, []]]) {
      ctx.font = `${10.5 * FS}px ${FONT}`;
      const tw = ctx.measureText(name).width;
      text(name, kx, LOSS.t - 14, "right", C.ink, 10.5); kx -= tw + 5;
      ctx.strokeStyle = colour; ctx.lineWidth = 1.7; ctx.setLineDash(dash);
      ctx.beginPath(); ctx.moveTo(kx - 14, LOSS.t - 18); ctx.lineTo(kx, LOSS.t - 18); ctx.stroke(); ctx.setLineDash([]);
      kx -= 20;
    }
  }

  // ---- predicted against true -----------------------------------------------------------
  function drawScatter() {
    const s = sv(), ok = s.ok && !trainPending() && !waitScene;
    const pred = ok ? frameOf(s.test_pred) : null;
    ["x", "y"].forEach((name, c) => {
      const l = SCAT.ls[c], t = SCAT.t, sz = SCAT.s;
      const X = (v) => l + v / box() * sz, Y = (v) => t + sz - v / box() * sz;
      ctx.strokeStyle = C.ink2; ctx.lineWidth = 1; ctx.strokeRect(l, t, sz, sz);
      ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(box()), Y(box())); ctx.stroke();
      for (const v of [0, 50, 100]) {
        text(String(v), X(v), t + sz + 13, "center", C.ink, 10);
        text(String(v), l - 5, Y(v) + 4, "right", C.ink, 10);
      }
      text("true", l + sz / 2 - 6, t + sz + 27, "center", C.ink2, 10.5);
      text("m", l + sz / 2 + 14, t + sz + 27, "left", C.ink, 12, SYM, "italic");
      text(name, l + sz / 2 + 24, t + sz + 30, "left", C.ink, 9, SYM, "italic");
      if (c === 0) {
        ctx.save(); ctx.translate(l - 30, t + sz / 2); ctx.rotate(-Math.PI / 2);
        text("predicted (km)", 0, 0, "center", C.ink2, 10.5); ctx.restore();
      }
      if (!pred) return;
      ctx.save(); ctx.beginPath(); ctx.rect(l, t, sz, sz); ctx.clip();
      ctx.fillStyle = "rgba(213,94,0,0.55)";
      pred.forEach((p, i) => {
        if (!fin(p[c])) return;
        ctx.beginPath(); ctx.arc(X(s.test_true[i][c]), Y(p[c]), 1.8, 0, 7); ctx.fill();
      });
      ctx.restore();
    });
  }

  // ---- one line of errors --------------------------------------------------------------
  function drawTable() {
    const r = run(), s = sv(), e = ev();
    const k = Math.min(Math.floor(epoch), (r.rms_train || []).length - 1);
    const pend = trainPending();
    const okS = s.ok && !pend && !waitScene;
    const rt = okS ? frameOf(s.rms_test?.map((v) => [v])) : null;
    const okE = e.ok && !pend && !waitEvent && !drag;
    let cloud = okE ? frameOf(e.cloud) : null;
    if (cloud && !cloud.every((p) => fin(p[0]) && fin(p[1]))) cloud = null;  // diverged
    let err = "—", spread = "—";
    if (cloud) {
      const ev0 = model.get("test_event");
      err = fmtKm(Math.hypot(cloud[0][0] - ev0[0], cloud[0][1] - ev0[1]));
      const n = cloud.length;
      const mx = cloud.reduce((a, p) => a + p[0], 0) / n;
      const my = cloud.reduce((a, p) => a + p[1], 0) / n;
      spread = fmtKm(Math.sqrt(cloud.reduce((a, p) => a + (p[0] - mx) ** 2 + (p[1] - my) ** 2, 0) / n));
    }
    const us = okS && s.us_per_event
      ? (s.us_per_event < 10 ? s.us_per_event.toFixed(1) : Math.round(s.us_per_event)) : "—";
    const items = [
      ["RMS ERROR (km)", null],
      ["training", !pend && k >= 0 ? fmtKm(r.rms_train[k]) : "—"],
      ["validation", !pend && k >= 0 ? fmtKm(r.rms_val[k]) : "—"],
      ["test", rt ? fmtKm(rt[0]) : "—"],
      ["best possible", !pend && r.floor ? fmtKm(r.floor) : "—", C.green],
      ["TEST EVENT (km)", null],
      ["error", err],
      ["spread of its answers", spread],
      ["time per event", `${us} µs`],
    ];
    let x = TABLE.x;
    const y = TABLE.y;
    for (const [label, value, colour] of items) {
      if (value === null) {
        if (x > TABLE.x) {  // a rule between the two groups
          x += 6; ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(x, y - 12); ctx.lineTo(x, y + 4); ctx.stroke(); x += 16;
        }
        text(label, x, y, "left", C.ink2, 10);
        ctx.font = `${10 * FS}px ${FONT}`;
        x += ctx.measureText(label).width + 14;
        continue;
      }
      text(label, x, y, "left", C.ink2, 11);
      ctx.font = `${11 * FS}px ${FONT}`;
      x += ctx.measureText(label).width + 6;
      text(value, x, y, "left", colour || C.ink, 12);
      ctx.font = `${12 * FS}px ${FONT}`;
      x += ctx.measureText(value).width + 20;
    }
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    for (const m of maps) drawMap(m);
    drawKey(); drawBanner(); drawLoss(); drawScatter(); drawTable();
    const last = lastEpoch();
    epochStat.set(String(Math.round(epoch)), `/ ${last || model.get("epochs")}`);
    scrub.set(last ? epoch / last : 0);
    syncPlay();
  }

  // ---- interaction ------------------------------------------------------------------------
  function pick(px, py) {
    const s = scene();
    let best = null, bd = Infinity;
    for (const map of maps) {
      const consider = (hit, p, limit) => {
        const [x, y] = pxOn(p, map), d = Math.hypot(x - px, y - py);
        if (d < limit && d < bd) { bd = d; best = { ...hit, map }; }
      };
      consider({ kind: "event" }, s.event, HIT + 2);
      s.stations.forEach((p, i) => consider({ kind: "station", index: i }, p, HIT));
    }
    return best;
  }
  const inLoss = (px, py) => px >= LOSS.l - 4 && px <= LOSS.l + LOSS.w + 4 && py >= LOSS.t && py <= LOSS.t + LOSS.h;
  function scrubTo(px) { pause(); setEpoch((px - LOSS.l) / LOSS.w * (run().config?.epochs || lastEpoch())); }
  const sameHit = (a, b) => (!a && !b) || (a && b && a.kind === b.kind && a.index === b.index && a.map === b.map);

  canvas.addEventListener("pointerdown", (e) => {
    const [px, py] = cv.xy(e), hit = pick(px, py);
    if (!hit) {
      if (inLoss(px, py)) { scrubbing = true; canvas.setPointerCapture(e.pointerId); scrubTo(px); }
      return;
    }
    if (hit.kind === "station" && e.shiftKey) {  // mute or unmute
      const m = new Set(model.get("muted"));
      if (m.has(hit.index)) m.delete(hit.index); else m.add(hit.index);
      setInput("muted", [...m].sort((a, b) => a - b));
      return;
    }
    drag = hit;
    live = { stations: model.get("stations").map((p) => [...p]), event: [...model.get("test_event")] };
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = "grabbing";
    draw();
  });
  canvas.addEventListener("pointermove", (e) => {
    const [px, py] = cv.xy(e);
    if (scrubbing) return scrubTo(px);
    if (!drag) {
      const hit = pick(px, py);
      if (!sameHit(hit, hover)) { hover = hit; draw(); }
      canvas.style.cursor = hit ? "grab" : inLoss(px, py) ? "ew-resize" : "default";
      return;
    }
    const [x, y] = kmOf(px, py, drag.map), p = [clampKm(x), clampKm(y)];
    if (drag.kind === "station") live.stations[drag.index] = p; else live.event = p;
    draw();
  });
  const release = () => {
    if (scrubbing) { scrubbing = false; return; }
    if (!drag) return;
    const kind = drag.kind, l2 = live;
    drag = null; live = null; canvas.style.cursor = "default";
    const r2 = (p) => [+p[0].toFixed(2), +p[1].toFixed(2)];
    if (kind === "station") setInput("stations", l2.stations.map(r2));
    else setInput("test_event", r2(l2.event));
    draw();
  };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
  canvas.addEventListener("pointerleave", () => { if (hover && !drag) { hover = null; draw(); } });
  canvas.addEventListener("dblclick", (e) => {
    const [px, py] = cv.xy(e), hit = pick(px, py);
    const stations = model.get("stations");
    if (hit?.kind === "station") {
      if (stations.length <= model.get("min_stations")) return;
      const m = model.get("muted").filter((i) => i !== hit.index).map((i) => (i > hit.index ? i - 1 : i));
      model.set("muted", m);
      setInput("stations", stations.filter((_, i) => i !== hit.index));
    } else if (!hit) {
      const map = mapAt(px, py);
      if (map && stations.length < model.get("max_stations")) {
        const [x, y] = kmOf(px, py, map);
        setInput("stations", [...stations, [+x.toFixed(2), +y.toFixed(2)]]);
      }
    }
    hover = null;
  });

  model.on("change:run", newRun);
  model.on("change:scene_view", () => { waitScene = false; updateStatus(); draw(); });
  model.on("change:event_view", () => { waitEvent = false; updateStatus(); draw(); });
  model.on("change:train_id", () => { updateStatus(); syncPlay(); draw(); });
  model.on("change:progress", () => {
    if (trainPending()) { progress = model.get("progress"); syncProgress(); }
  });
  syncHeads();
  syncControls();
  if (!trainPending() && lastEpoch() > 0) newRun(); else draw();
}
export default { render };
