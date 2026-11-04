// The Part 2 convolution playground (see convwidget.py); controls.js is put in front
// of this module (tokens C, FONT, SYM and the ui* controls).
//
// Notation as in the notebook: u is the record, w_tau the filter of K weights, b the
// bias, h the feature map, h_t = b + sum_tau w_tau u_{t+tau}.
//
// Left: the whole record u with its feature map h under it (one time axis), then a
// zoom on the shaded window, again u over h. Right, top: the filter (a row of presets,
// size, bias, ReLU). Right, bottom: the arithmetic of the marked position. One bar per
// weight w_tau (draggable) and under it the product w_tau u_{t+tau}, with the products
// adding up to h_t. Pointing at a column rings its sample in the zoom. The convolution
// runs here, in the browser, on the few records Python sent.

const W = 1280;
const ZOOM = 120;  // samples in the zoom window (2.4 s at 50 Hz)
const PL = 94, PR = 690;  // left plots, logical px
const RX = 760, RR = 1264;  // right column
const LCL = 836, LCR = 1148;  // lens columns
const RB0 = 1196, RB1 = 1242;  // the sum, drawn as one bar
const SIZES = [3, 5, 7, 9, 15];
const NICE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.5, 2, 5];  // kernel axis ranges
const AMPS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 20];
const PRESETS = [  // two rows of four: what it keeps, then what it looks for
  ["identity", "identity"], ["smooth", "smooth"], ["highpass", "high-pass"],
  ["bandpass", "band-pass"], ["rise", "rise edge"], ["fall", "fall edge"],
  ["slope", "slope"], ["bump", "bump"],
];
const ABOUT = {
  identity: "Copies the record, so h equals u. The starting point for every filter.",
  smooth: "Mean of the K neighbours, a low-pass: slow wiggles stay, fast ones average out.",
  highpass: "Sample minus the local mean: slow drift goes, fast wiggles stay.",
  bandpass: "A tapered cosine: large where the record oscillates near the frequency.",
  rise: "−1 before, +1 after: large where the record steps up, an onset.",
  fall: "The mirror image of the rise edge: large where the record steps down.",
  slope: "Fits a straight line through the window: large on steady ramps.",
  bump: "Positive middle, negative sides: large at an isolated peak, zero on a ramp.",
  custom: "Your own weights: drag the bars, or pick a preset to start over.",
  learned: "Found by the network during training: no name, but see what it responds to.",
};

function presetWeights(name, K, rate, freq) {
  const c = (K - 1) / 2, ks = Array.from({ length: K }, (_, k) => k);
  const unit = (w) => {  // unit absolute sum, so no preset is louder than the record
    const s = w.reduce((a, v) => a + Math.abs(v), 0) || 1;
    return w.map((v) => v / s);
  };
  let w;
  if (name === "identity") w = ks.map((k) => (k === Math.round(c) ? 1 : 0));
  else if (name === "smooth") w = ks.map(() => 1 / K);
  else if (name === "highpass") w = ks.map((k) => (k === Math.round(c) ? 1 : 0) - 1 / K);
  else if (name === "rise" || name === "fall") {  // a mean difference, as in the PyTorch cell
    const sign = name === "rise" ? 1 : -1;
    w = ks.map((k) => (k < c ? -sign : k > c ? sign : 0) / Math.floor(K / 2));
  } else if (name === "slope") w = unit(ks.map((k) => k - c));
  else if (name === "bump") {  // a Ricker wavelet, with its mean removed
    const s = Math.max(c / 2, 0.75);
    const r = ks.map((k) => (1 - ((k - c) / s) ** 2) * Math.exp(-(((k - c) / s) ** 2) / 2));
    const m = r.reduce((a, v) => a + v, 0) / K;
    w = unit(r.map((v) => v - m));
  } else if (name === "bandpass") {
    w = unit(ks.map((k) => Math.cos(2 * Math.PI * freq * (k - c) / rate)
      * 0.5 * (1 + Math.cos(Math.PI * (k - c) / (c + 1)))));
  } else w = ks.map(() => 0);
  return w.map((v) => Math.round(1000 * v) / 1000);
}

function render({ model, el }) {
  const st = makeStage(el, W, 1.45);
  const local = elt("style");
  // vertical positions of the stage, logical px
  const R = {
    tilesTop: 28, tw: (RR - RX) / 4, tp: 54, ctrlTop: 186,
    wIn: [80, 126], wOut: [132, 178],  // whole record: u, then h
    zIn: [272, 392], zOut: [402, 522],  // zoom: u, then h
    lw: [278, 368], lp: [414, 504],  // lens: weights, products
  };
  R.zCap = R.zIn[0] - 12;
  const H = Math.max(R.lp[1] + 56, R.zOut[1] + 46);
  local.textContent = `
  .cvx .mls-label { font-size: 11.5px; }
  .cvx .mls-sym { font-size: 15px; }
  .cvx .mls-seg button { font-size: 14px; }
  .cvx-range { width: 84px; accent-color: ${C.orange}; }
  .cvx-read { font-size: 15px; min-width: 4.5ch; text-align: right;
    font-variant-numeric: tabular-nums; }
  .cvx-row { display: flex; align-items: center; gap: 8px; }
  .cvx-head { position: absolute; left: 12px; top: 4px; display: flex;
    align-items: flex-end; gap: 10px 26px; }
  .cvx-ctrl { position: absolute; left: ${RX}px; top: ${R.ctrlTop}px; display: flex;
    align-items: flex-start; gap: 22px; }
  .cvx-ctrl .mls-seg button { padding: 3px 9px; }
  .cvx-tr { position: absolute; left: ${PR - 112}px; top: ${R.zCap - 36}px; }
  .cvx-tr .mls-transport { gap: 6px; }
  .cvx-tr .mls-btn { width: 28px; height: 28px; }
  .cvx-tr .mls-btn svg { width: 14px; height: 14px; }
  .cvx-tr .mls-btn.mls-play { width: 36px; height: 36px; }
  .cvx-tr .mls-btn.mls-play svg { width: 18px; height: 18px; }`;
  st.stage.classList.add("cvx");
  st.stage.append(local);

  // ---- the records ---------------------------------------------------------------
  const classes = model.get("classes"), colours = model.get("colours");
  const label = model.get("label"), T = model.get("samples"), rate = model.get("rate");
  const recs = (() => {
    const raw = atob(model.get("records"));
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    const q = new Int16Array(bytes.buffer);
    return Float32Array.from(q, (v) => v / 32767);
  })();
  const byClass = classes.map((_, k) => label.flatMap((l, i) => (l === k ? [i] : [])));
  const record = (i) => recs.subarray(i * T, (i + 1) * T);

  // ---- state ----------------------------------------------------------------------
  let cls = 0, pos = 0, K = 7, preset = "identity", freq = 6, relu = false, bias = 0;
  let w = presetWeights(preset, K, rate, freq);
  let kscale = 0.5;  // the weight axis runs over +-kscale; fixed while a bar is dragged
  let zoom0 = 0, t = 0, playing = false, timer = null, drag = null, hover = -1;
  let hoverTau = -1;  // lens column under the pointer
  let hAll = new Float32Array(0);  // feature map of the current record
  const current = () => byClass[cls][pos];
  const learnedKernels = () => model.get("kernels");
  const niceScale = () => {
    const m = Math.max(...w.map(Math.abs), 1e-9) / 0.68;
    return NICE.find((v) => v >= m) ?? 5;
  };

  // ---- the convolution, as in torch: h_t = b + sum_tau w_tau u_{t+tau} -------------
  function conv(u) {
    const n = T - K + 1, out = new Float32Array(n);
    for (let s = 0; s < n; s++) {
      let a = bias;
      for (let k = 0; k < K; k++) a += w[k] * u[s + k];
      out[s] = a;
    }
    return out;
  }
  const act = (v) => (relu ? Math.max(v, 0) : v);
  const compute = () => { hAll = conv(record(current())); };

  // ---- header: what belongs to the data ----------------------------------------------
  const head = elt("div", "cvx-head");
  const classSeg = uiSegmented(classes.map((c, k) => [k, c]), cls, (v) => setClass(v));
  const prev = elt("button", "mls-pm", "‹"), next = elt("button", "mls-pm", "›");
  prev.title = "previous record"; next.title = "next record";
  prev.onclick = () => setRecord(pos - 1);
  next.onclick = () => setRecord(pos + 1);
  const recText = elt("div", "cvx-read");
  recText.style.minWidth = "5.5ch";
  const recRow = elt("div", "cvx-row");
  recRow.append(prev, recText, next);
  head.append(uiField(uiLabel("CLASS"), classSeg), uiField(uiLabel("RECORD"), recRow));

  // playback sits with the zoom it moves
  const transport = uiTransport({
    onReset: () => { setPlaying(false); t = zoom0 + 10; draw(); },
    onPlay: () => setPlaying(!playing),
    onStep: () => { setPlaying(false); advance(1); },
    resetTitle: "back to the start of the zoom",
  });
  const trBox = elt("div", "cvx-tr");
  trBox.append(transport.el);

  // ---- the filter's controls, in the right column ------------------------------------------
  const ctrl = elt("div", "cvx-ctrl");
  const sizeSeg = uiSegmented(SIZES.map((k) => [k, String(k)]), K, (v) => setKernel(v));
  const biasRange = elt("input", "cvx-range");
  biasRange.type = "range"; biasRange.min = "-0.3"; biasRange.max = "0.3";
  biasRange.step = "0.01"; biasRange.value = "0";
  biasRange.oninput = () => { bias = Number(biasRange.value); draw(); };
  const biasText = elt("div", "cvx-read");
  biasText.style.minWidth = "3.8ch";
  const biasRow = elt("div", "cvx-row");
  biasRow.append(biasRange, biasText);
  const reluSeg = uiSegmented([[0, "off"], [1, "on"]], 0, (v) => { relu = v === 1; draw(); });
  ctrl.append(uiField(uiLabel("SIZE", "K"), sizeSeg), uiField(uiLabel("BIAS", "b"), biasRow),
    uiField(uiLabel("RELU"), reluSeg));

  // frequency of the band-pass, next to the FILTER caption
  const freqNum = uiNumber(freq, { min: 1, max: 24 }, (v) => { freq = v; refreshPreset(); });
  freqNum.size = 3;
  const freqField = uiField(uiLabel("FREQUENCY", "", "Hz"), freqNum);
  freqField.style.cssText = `position:absolute;right:${W - RR}px;top:2px;flex-direction:row;
    align-items:baseline;gap:8px;`;

  // the filters a trained network found, stepped through next to the FILTER caption
  const lprev = elt("button", "mls-pm", "‹"), lnext = elt("button", "mls-pm", "›");
  lprev.title = "previous learned filter"; lnext.title = "next learned filter";
  const ltext = elt("div", "cvx-read");
  ltext.style.minWidth = "5.5ch";
  const lrow = elt("div", "cvx-row");
  lrow.append(lprev, ltext, lnext);
  const learnedField = uiField(uiLabel("LEARNED"), lrow);
  learnedField.style.cssText = `position:absolute;left:${RX + 112}px;top:2px;flex-direction:row;
    align-items:center;gap:8px;`;
  const learnedIndex = () => (preset.startsWith("learned") ? Number(preset.slice(8)) : -1);
  const stepLearned = (d) => {
    const n = learnedKernels().length, i = learnedIndex();
    setPreset(`learned ${Math.min(Math.max(i < 0 ? (d > 0 ? 0 : n - 1) : i + d, 0), n - 1)}`);
  };
  lprev.onclick = () => stepLearned(-1);
  lnext.onclick = () => stepLearned(1);
  st.stage.append(head, trBox, ctrl, freqField, learnedField);

  const fig = stageCanvas(st, W, H);
  st.stage.prepend(fig.canvas);
  const ctx = fig.ctx;

  // ---- state changes -------------------------------------------------------------------
  function peak(u) {  // the loudest sample: where the zoom opens on a new record
    let best = 0;
    for (let i = 1; i < T; i++) if (Math.abs(u[i]) > Math.abs(u[best])) best = i;
    return best;
  }
  function openAt(s) {
    zoom0 = Math.min(Math.max(Math.round(s - ZOOM / 2), 0), T - ZOOM);
    t = Math.min(Math.max(Math.round(s - K / 2), zoom0), zoom0 + ZOOM - K);
  }
  function setClass(k) { cls = k; pos = 0; setRecord(0); }
  function setRecord(p) {
    const n = byClass[cls].length;
    pos = Math.min(Math.max(p, 0), n - 1);
    openAt(peak(record(current())));
    draw();
  }
  function setKernel(k) {
    K = k;
    if (preset.startsWith("learned")) preset = "custom";  // a resized filter is no longer it
    if (preset !== "custom") w = presetWeights(preset, K, rate, freq);
    else w = Array.from({ length: K }, (_, i) => w[i] ?? 0);
    kscale = niceScale();
    t = Math.min(t, zoom0 + ZOOM - K);
    draw();
  }
  function refreshPreset() {
    if (preset.startsWith("learned")) {
      const learned = learnedKernels()[Number(preset.slice(8))];
      w = learned.slice(); K = learned.length;
    } else if (preset !== "custom") w = presetWeights(preset, K, rate, freq);
    kscale = niceScale();
    t = Math.min(t, zoom0 + ZOOM - K);
    draw();
  }
  function setPreset(v) { preset = v; refreshPreset(); }
  function advance(d) {
    t += d;
    if (t + K > T) { t = 0; zoom0 = 0; }
    if (t + K > zoom0 + ZOOM - 8) zoom0 = Math.min(Math.max(t - 8, 0), T - ZOOM);
    draw();
  }
  function setPlaying(on) {
    playing = on;
    transport.setPlaying(on);
    clearInterval(timer);
    if (on) timer = setInterval(() => advance(1), 40);
  }
  model.on("change:kernels", () => draw());

  // ---- drawing helpers -------------------------------------------------------------------
  const font = (px, style = "") => { ctx.font = `${style} ${px}px ${FONT}`; };
  const text = (s, x, y0, colour = C.ink, align = "left", px = 13) => {
    font(px); ctx.fillStyle = colour; ctx.textAlign = align; ctx.textBaseline = "alphabetic";
    ctx.fillText(s, x, y0);
  };
  const sym = (s, x, y0, colour = C.ink, align = "left", px = 16) => {
    ctx.font = `italic ${px}px ${SYM}`; ctx.fillStyle = colour; ctx.textAlign = align;
    ctx.textBaseline = "alphabetic"; ctx.fillText(s, x, y0);
  };
  // symbols with subscripts on one baseline: parts are {t, sub, c, plain}; returns the
  // end x (draw = false only measures, to right-align a formula)
  const mathText = (parts, x, y0, px = 14, draw = true) => {
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
    for (const p of parts) {
      ctx.font = p.plain ? `${px - 1}px ${FONT}` : `italic ${px}px ${SYM}`;
      if (p.plain) x += 3;
      ctx.fillStyle = p.c || C.ink;
      if (draw) ctx.fillText(p.t, x, y0);
      x += ctx.measureText(p.t).width + (p.plain ? 4 : 1);
      if (p.sub) {
        ctx.font = `italic ${px * 0.72}px ${SYM}`;
        if (draw) ctx.fillText(p.sub, x, y0 + 4);
        x += ctx.measureText(p.sub).width + 2;
      }
    }
    return x;
  };
  function wrap(s, x, y0, maxW, lead, colour = C.ink2, px = 13.5) {
    font(px); ctx.fillStyle = colour; ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
    let line = "";
    for (const word of s.split(" ")) {
      const nxt = line ? `${line} ${word}` : word;
      if (ctx.measureText(nxt).width > maxW && line) {
        ctx.fillText(line, x, y0); y0 += lead; line = word;
      } else line = nxt;
    }
    ctx.fillText(line, x, y0);
  }
  const alpha = (hex, a) => {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${n >> 16}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
  };
  const caption = (s, x, y0) => text(s, x, y0, C.ink, "left", 13);
  const num = (v, d = 2) => (v < 0 && +Math.abs(v).toFixed(d) !== 0 ? "−" : "")
    + Math.abs(v).toFixed(d);
  const tick = (v) => (v < 0 ? "−" : "") + String(+Math.abs(v).toPrecision(2));
  const wText = (v) => (v === 0 ? "0" : num(v, 2).replace(/^(−?)0\./, "$1.").replace(/\.?0+$/, ""));
  // ticks: [value, label or null]; inward ticks, labels outside, the zero line through signed data
  function axes(l, top, r, b, { y0 = null, ticks = [], ymap }) {
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(l, top); ctx.lineTo(l, b); ctx.stroke();
    const base = y0 === null ? b : y0;
    ctx.beginPath(); ctx.moveTo(l, base); ctx.lineTo(r, base); ctx.stroke();
    for (const [v, lab] of ticks) {
      const yy = ymap(v);
      ctx.beginPath(); ctx.moveTo(l, yy); ctx.lineTo(l + 4, yy); ctx.stroke();
      if (lab !== null) text(lab, l - 6, yy + 4, C.ink2, "right", 12);
    }
  }
  function timeTicks(l, r, b, s0, s1, lab) {
    const sec0 = s0 / rate, sec1 = s1 / rate, span = sec1 - sec0;
    const step = span > 30 ? 10 : span > 8 ? 2 : span > 3 ? 0.5 : 0.2;
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1;
    for (let v = Math.ceil(sec0 / step) * step; v <= sec1 + 1e-9; v += step) {
      const x = l + (v - sec0) / span * (r - l);
      ctx.beginPath(); ctx.moveTo(x, b); ctx.lineTo(x, b - 4); ctx.stroke();
      text(String(+v.toFixed(2)), x, b + 16, C.ink2, "center", 12);
    }
    if (lab) text(lab, (l + r) / 2, b + 33, C.ink2, "center", 12.5);
  }
  const range = (arr, floor) => {
    let m = floor;
    for (const v of arr) if (Math.abs(v) > m) m = Math.abs(v);
    return m;
  };
  const xOf = (s, s0, s1, l = PL, r = PR) => l + (s - s0) / (s1 - s0) * (r - l);
  const sampleAt = (px, s0, s1) => s0 + (px - PL) / (PR - PL) * (s1 - s0);
  const centre = (s) => s + (K - 1) / 2;  // a window starting at s is drawn at its middle

  // a long trace as a polyline
  function trace(arr, offset, s0, s1, top, b, ymap, colour) {
    ctx.save();
    ctx.beginPath(); ctx.rect(PL, top - 4, PR - PL, b - top + 8); ctx.clip();
    ctx.strokeStyle = colour; ctx.lineWidth = 0.9; ctx.beginPath();
    arr.forEach((v, i) => {
      const x = xOf(i + offset, s0, s1), yy = ymap(v);
      i ? ctx.lineTo(x, yy) : ctx.moveTo(x, yy);
    });
    ctx.stroke(); ctx.restore();
  }
  function windowBox(top, b) {
    const x0 = xOf(zoom0, 0, T), x1 = xOf(zoom0 + ZOOM, 0, T);
    ctx.fillStyle = alpha(C.orange, 0.16); ctx.fillRect(x0, top, x1 - x0, b - top);
    ctx.strokeStyle = C.orange; ctx.lineWidth = 1;
    ctx.strokeRect(x0 + 0.5, top + 0.5, x1 - x0 - 1, b - top - 1);
    ctx.fillStyle = C.orange; ctx.fillRect(xOf(t + K / 2, 0, T) - 1, b - 7, 2, 7);
  }
  // what the symbols u and h stand for, in the left margin of a panel
  function sideLabel(s, words, colour, top, b) {
    const mid = (top + b) / 2;
    sym(s, 10, mid + (words.length ? 2 : 6), colour, "left", 18);
    words.forEach((wd, i) => text(wd, 10, mid + 17 + 13 * i, C.ink2, "left", 11.5));
  }

  // ---- panels --------------------------------------------------------------------------------
  function draw() {
    compute();
    const u = record(current());
    ctx.clearRect(0, 0, W, H);
    recText.textContent = `${pos + 1} / ${byClass[cls].length}`;
    biasText.textContent = num(bias);
    classSeg.set(cls); sizeSeg.set(SIZES.includes(K) ? K : -1);
    freqField.style.display = preset === "bandpass" ? "" : "none";
    const nLearned = learnedKernels().length, li = learnedIndex();
    learnedField.style.display = nLearned ? "" : "none";
    ltext.textContent = li >= 0 ? `${li + 1} / ${nLearned}` : `– / ${nLearned}`;
    lprev.disabled = li === 0; lnext.disabled = li === nLearned - 1;
    prev.disabled = pos <= 0; next.disabled = pos >= byClass[cls].length - 1;
    const colour = colours[cls];

    // one vertical scale for u and h, so their sizes can be compared by eye
    const hAct = Array.from(hAll, act);
    const need = Math.max(range(u, 0.05), range(hAct, 0.05));
    const amp = AMPS.find((v) => v >= need - 1e-9) ?? Math.ceil(need);
    const lim = amp * 1.05;
    const hLo = relu ? -0.05 * lim : -lim;
    const umap = (top, b) => (v) => (top + b) / 2 - v * (b - top) / 2 / lim;
    const hmap = (top, b) => (v) => b - (v - hLo) / (lim - hLo) * (b - top);
    const uTicks = [[-amp, tick(-amp)], [0, "0"], [amp, tick(amp)]];
    const hTicks = relu ? [[0, "0"], [amp, tick(amp)]] : uTicks;
    const z1 = zoom0 + ZOOM;

    // the whole record, and below it its feature map
    caption("WHOLE RECORD", PL, R.wIn[0] - 12);
    text(`${classes[cls]}, ${Math.round(T / rate)} s`, PL + 112, R.wIn[0] - 12, colour,
      "left", 13);
    let [top, b] = R.wIn;
    let ym = umap(top, b);
    axes(PL, top, PR, b, { y0: ym(0), ticks: [[-amp, null], [0, "0"], [amp, tick(amp)]], ymap: ym });
    windowBox(top, b);
    trace(u, 0, 0, T, top, b, ym, C.ink);
    sideLabel("u", ["record"], C.ink, top, b);
    [top, b] = R.wOut;
    let hm = hmap(top, b);
    axes(PL, top, PR, b, { y0: relu ? null : hm(0),
      ticks: relu ? hTicks : [[-amp, null], [0, "0"], [amp, tick(amp)]], ymap: hm });
    windowBox(top, b);
    trace(hAct, (K - 1) / 2, 0, T, top, b, hm, C.blue);
    timeTicks(PL, PR, b, 0, T, "");
    sideLabel("h", ["feature", "map"], C.blue, top, b);

    // the zoom: samples as dots with the window over them, and the feature map under it
    caption("ZOOM", PL, R.zCap);
    text("drag along a trace", PL + 58, R.zCap, C.ink2, "left", 12.5);
    [top, b] = R.zIn;
    ym = umap(top, b);
    axes(PL, top, PR, b, { y0: ym(0), ticks: uTicks, ymap: ym });
    const bx0 = xOf(t, zoom0, z1), bx1 = xOf(t + K, zoom0, z1);
    ctx.fillStyle = alpha(C.orange, 0.16); ctx.fillRect(bx0, top, bx1 - bx0, b - top);
    ctx.strokeStyle = C.orange; ctx.lineWidth = 1;
    ctx.strokeRect(bx0 + 0.5, top + 0.5, bx1 - bx0 - 1, b - top - 1);
    ctx.save(); ctx.beginPath(); ctx.rect(PL, top - 4, PR - PL, b - top + 8); ctx.clip();
    ctx.strokeStyle = alpha(C.ink, 0.45); ctx.lineWidth = 0.8; ctx.beginPath();
    for (let s = zoom0; s < z1; s++) {
      const px = xOf(s + 0.5, zoom0, z1), py = ym(u[s]);
      s === zoom0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.stroke();
    for (let s = zoom0; s < z1; s++) {
      const inside = s >= t && s < t + K;
      ctx.fillStyle = inside ? C.orange : C.ink;
      ctx.beginPath(); ctx.arc(xOf(s + 0.5, zoom0, z1), ym(u[s]), inside ? 3.2 : 2, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.restore();
    sideLabel("u", [], C.ink, top, b);

    [top, b] = R.zOut;
    hm = hmap(top, b);
    axes(PL, top, PR, b, { y0: relu ? null : hm(0), ticks: hTicks, ymap: hm });
    const shown = [];
    for (let s = Math.max(zoom0, 0); s + K <= z1 && s < hAll.length; s++) shown.push(act(hAll[s]));
    ctx.save(); ctx.beginPath(); ctx.rect(PL, top - 4, PR - PL, b - top + 8); ctx.clip();
    ctx.strokeStyle = C.blue; ctx.lineWidth = 1.1; ctx.beginPath();
    shown.forEach((v, i) => {
      const px = xOf(centre(zoom0 + i) + 0.5, zoom0, z1), py = hm(v);
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    });
    ctx.stroke();
    ctx.fillStyle = C.blue;
    shown.forEach((v, i) => {
      ctx.beginPath(); ctx.arc(xOf(centre(zoom0 + i) + 0.5, zoom0, z1), hm(v), 2, 0, 2 * Math.PI);
      ctx.fill();
    });
    const cx = xOf(centre(t) + 0.5, zoom0, z1), cy = hm(act(hAll[t]));
    ctx.setLineDash([2, 3]); ctx.strokeStyle = C.ink2; ctx.lineWidth = 0.8;
    ctx.beginPath(); ctx.moveTo(cx, R.zIn[1]); ctx.lineTo(cx, cy); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = C.orange; ctx.beginPath(); ctx.arc(cx, cy, 4, 0, 2 * Math.PI); ctx.fill();
    ctx.restore();
    timeTicks(PL, PR, b, zoom0, z1, "time (s)");
    sideLabel("h", [], C.blue, top, b);

    drawFilter();
    drawLens(u, umap);
  }

  // ---- the filter: a row of presets, then what the selected one does ----------------------------
  const tileWeights = (v) => presetWeights(v, K, rate, freq);
  const tileBox = (i) => ({ x: RX + (i % 4) * R.tw, y: R.tilesTop + Math.floor(i / 4) * R.tp });
  function drawFilter() {
    caption("FILTER", RX, 14);
    PRESETS.forEach(([v, name], i) => {
      const { x: x0, y: y0 } = tileBox(i), tw = R.tw - 6, sel = v === preset;
      if (sel || i === hover) {  // only the chosen (or pointed-at) tile gets a frame
        ctx.lineWidth = sel ? 1.5 : 1;
        ctx.strokeStyle = sel ? C.ink : C.ink2;
        ctx.strokeRect(x0 + 0.5, y0 + 0.5, tw, R.tp - 9);
      }
      const kw = tileWeights(v), m = Math.max(...kw.map(Math.abs), 1e-9);
      const gl = x0 + 14, gr = x0 + tw - 13, mid = y0 + 17, half = 11, bw = (gr - gl) / kw.length;
      ctx.fillStyle = sel ? C.orange : C.ink2;
      kw.forEach((q, k) => {
        const h = q / m * half;
        ctx.fillRect(gl + k * bw + (bw > 6 ? 1 : 0), Math.min(mid - h, mid), Math.max(bw - (bw > 6 ? 2 : 0.5), 1),
          Math.abs(h) || 0.8);
      });
      text(name, x0 + tw / 2 + 0.5, y0 + 39, sel ? C.ink : C.ink2, "center", 13);
    });
    const key = preset.startsWith("learned") ? "learned" : preset;
    wrap(ABOUT[key], RX, R.tilesTop + 2 * R.tp + 22, RR - RX, 18, C.ink2, 13.5);
  }

  // ---- the lens: weights, products, and their sum -------------------------------------
  const lensW = () => (LCR - LCL) / K;
  const lensCol = (xy) => {
    if (xy[0] < LCL || xy[0] > LCR || xy[1] < R.lw[0] - 4 || xy[1] > R.lp[1] + 4) return -1;
    return Math.min(Math.floor((xy[0] - LCL) / lensW()), K - 1);
  };
  function drawLens(u, umap) {
    const cw = lensW(), showNums = cw >= 32, gap = cw > 18 ? 6 : 3;
    const colX = (k) => LCL + k * cw;
    const p = w.map((v, k) => v * u[t + k]);
    const total = bias + p.reduce((s, v) => s + v, 0), out = act(total);
    // column under the pointer, behind everything
    if (hoverTau >= 0) {
      ctx.fillStyle = alpha(C.orange, 0.1);
      ctx.fillRect(colX(hoverTau), R.lw[0], cw, R.lp[1] - R.lw[0]);
      // ring the sample it belongs to in the zoom
      const zy = umap(R.zIn[0], R.zIn[1])(u[t + hoverTau]);
      ctx.strokeStyle = C.orange; ctx.lineWidth = 1.5; ctx.beginPath();
      ctx.arc(xOf(t + hoverTau + 0.5, zoom0, zoom0 + ZOOM), zy, 7, 0, 2 * Math.PI); ctx.stroke();
    }

    // weights, as bars you drag
    let [top, b] = R.lw;
    const mid = (top + b) / 2, a = kscale;
    caption("WEIGHTS", LCL, top - 12);
    mathText([{ t: "w", sub: "τ", c: C.orange }], LCL + 76, top - 12, 14);
    text("drag the bars", LCL + 112, top - 12, C.ink2, "left", 12.5);
    const wm = (v) => mid - v / a * (b - top) / 2;
    const aText = String(+a.toFixed(1));
    axes(LCL, top, LCR, b, { y0: mid, ticks: [[-a, "−" + aText], [0, "0"], [a, aText]], ymap: wm });
    w.forEach((v, k) => {
      ctx.fillStyle = C.orange;
      ctx.fillRect(colX(k) + gap / 2, Math.min(wm(v), mid), cw - gap, Math.abs(wm(v) - mid) || 1);
      if (showNums) text(wText(v), colX(k) + cw / 2,
        Math.min(Math.max(v >= 0 ? wm(v) - 4 : wm(v) + 12, top + 9), b - 3), C.ink2, "center", 11.5);
    });

    // products, and their sum
    [top, b] = R.lp;
    const pmid = (top + b) / 2;
    const pa = Math.max(range(p, 0.05), Math.abs(total), 0.05) / 0.68;
    const pm = (v) => pmid - v / pa * (b - top) / 2;
    caption("PRODUCTS", LCL, top - 12);
    mathText([{ t: "w", sub: "τ", c: C.orange }, { t: "u", sub: "t+τ" }], LCL + 96, top - 12, 14);
    const fx = mathText([{ t: "h", sub: "t", c: C.blue }, { t: "=", plain: true }, { t: "b" },
      { t: "+", plain: true }, { t: "Σ", plain: true }, { t: "w", sub: "τ", c: C.orange },
      { t: "u", sub: "t+τ" }], 0, 0, 14, false);
    mathText([{ t: "h", sub: "t", c: C.blue }, { t: "=", plain: true }, { t: "b" },
      { t: "+", plain: true }, { t: "Σ", plain: true }, { t: "w", sub: "τ", c: C.orange },
      { t: "u", sub: "t+τ" }], RR - fx, top - 12, 14);
    axes(LCL, top, RB1 + 2, b, { y0: pmid, ticks: [[-pa, null], [0, "0"], [pa, null]], ymap: pm });
    p.forEach((v, k) => {
      ctx.fillStyle = C.ink2;
      ctx.fillRect(colX(k) + gap / 2, Math.min(pm(v), pmid), cw - gap, Math.abs(pm(v) - pmid) || 1);
      if (showNums) text(wText(v), colX(k) + cw / 2,
        Math.min(Math.max(v >= 0 ? pm(v) - 4 : pm(v) + 12, top + 9), b - 3), C.ink2, "center", 11.5);
    });
    sym("Σ", (LCR + RB0) / 2, pmid - 2, C.ink, "center", 20);
    mathText([{ t: "+", plain: true }, { t: "b" }], (LCR + RB0) / 2 - 15, pmid + 17, 14);
    ctx.fillStyle = C.blue;  // h_t as the feature map shows it, after ReLU if it is on
    ctx.fillRect(RB0, Math.min(pm(out), pmid), RB1 - RB0, Math.abs(pm(out) - pmid) || 1);

    // tap numbers under the columns, then the hover read-out
    for (let k = 0; k < K; k++) {
      if (cw >= 17 || k % 2 === 0) text(String(k), colX(k) + cw / 2, b + 16, C.ink2, "center", 11);
    }
    mathText([{ t: "τ", c: C.ink2 }], LCL - 18, b + 17, 14);
    const vend = mathText([{ t: "h", sub: "t", c: C.blue }, { t: "=", plain: true }], 0, 0, 14, false);
    const hText = num(out);
    font(14);
    const lw0 = RR - (vend + ctx.measureText(hText).width + 4);
    const end = mathText([{ t: "h", sub: "t", c: C.blue }, { t: "=", plain: true }], lw0, b + 18, 14);
    text(hText, end + 4, b + 18, C.blue, "left", 14);
    if (hoverTau >= 0) {
      const k = hoverTau;
      text(`τ = ${k}    u = ${num(u[t + k])}    w = ${num(w[k])}    w·u = ${num(p[k])}`,
        LCL, b + 42, C.ink, "left", 13.5);
    } else text("point at a column for its numbers", LCL, b + 42, C.ink2, "left", 12.5);
  }

  // ---- pointer ----------------------------------------------------------------------------------------
  const inBox = (xy, [top, b], l = PL, r = PR) =>
    xy[0] >= l - 8 && xy[0] <= r + 8 && xy[1] >= top - 12 && xy[1] <= b + 14;
  const tileAt = (xy) => PRESETS.findIndex((_, i) => {
    const { x, y } = tileBox(i);
    return xy[0] >= x && xy[0] <= x + R.tw - 6 && xy[1] >= y && xy[1] <= y + R.tp - 6;
  });
  const overBars = (xy) => xy[0] >= LCL - 4 && xy[0] <= LCR + 4
    && xy[1] >= R.lw[0] - 8 && xy[1] <= R.lw[1] + 8;
  const overWindow = (xy) => inBox(xy, R.wIn) || inBox(xy, R.wOut);
  const overKernelTrack = (xy) => inBox(xy, R.zIn) || inBox(xy, R.zOut);
  function setWeight(xy) {
    const [top, b] = R.lw;
    const k = Math.min(Math.max(Math.floor((xy[0] - LCL) / lensW()), 0), K - 1);
    let v = (((top + b) / 2 - xy[1]) / ((b - top) / 2)) * kscale;
    v = Math.min(Math.max(v, -kscale), kscale);
    w[k] = Math.abs(v) < 0.04 * kscale ? 0 : Math.round(v * 1000) / 1000;  // zero is easy to hit
    preset = "custom";
    draw();
  }
  function moveWindow(xy) {  // the zoom window follows a click in the whole-record panels
    openAt(sampleAt(xy[0], 0, T));
    draw();
  }
  function moveKernel(xy) {
    const s = sampleAt(xy[0], zoom0, zoom0 + ZOOM) - K / 2;
    t = Math.min(Math.max(Math.round(s), zoom0), zoom0 + ZOOM - K);
    draw();
  }
  fig.canvas.addEventListener("pointerdown", (ev) => {
    const xy = fig.xy(ev), tile = tileAt(xy);
    if (tile >= 0) { setPlaying(false); setPreset(PRESETS[tile][0]); return; }
    if (overBars(xy)) drag = "weight";
    else if (overWindow(xy)) drag = "window";
    else if (overKernelTrack(xy)) drag = "kernel";
    else return;
    setPlaying(false);
    try { fig.canvas.setPointerCapture(ev.pointerId); } catch { /* synthetic events */ }
    onMove(xy);
  });
  function onMove(xy) {
    if (drag === "weight") setWeight(xy);
    else if (drag === "window") moveWindow(xy);
    else if (drag === "kernel") moveKernel(xy);
  }
  fig.canvas.addEventListener("pointermove", (ev) => {
    const xy = fig.xy(ev);
    if (drag) { onMove(xy); return; }
    const tile = tileAt(xy), col = lensCol(xy);
    fig.canvas.style.cursor = tile >= 0 ? "pointer" : overBars(xy) ? "ns-resize"
      : overWindow(xy) || overKernelTrack(xy) ? "grab" : "default";
    if (tile !== hover || col !== hoverTau) { hover = tile; hoverTau = col; draw(); }
  });
  fig.canvas.addEventListener("pointerleave", () => {
    if (hover !== -1 || hoverTau !== -1) { hover = -1; hoverTau = -1; draw(); }
  });
  const release = () => {
    if (drag === "weight") { kscale = niceScale(); draw(); }
    drag = null;
  };
  fig.canvas.addEventListener("pointerup", release);
  fig.canvas.addEventListener("pointercancel", release);

  const learned0 = learnedKernels();
  if (learned0.length) { preset = "learned 0"; w = learned0[0].slice(); K = learned0[0].length; }
  kscale = niceScale();
  setRecord(0);
  return () => clearInterval(timer);
}

export default { render };
