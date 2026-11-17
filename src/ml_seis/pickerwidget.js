// The Part 2d explorer (see pickerwidget.py); controls.js is put in front of this module
// (tokens C, FONT and the ui* controls).
//
// mode "data": a map of the earthquakes and stations, the record section of the selected
// earthquake (vertical component of each station, by distance, aligned on P), and one
// trace with its three components and the analyst P and S.
// mode "results": a list of the test traces (sorted, filtered), one trace with the
// picker's P and S probability curves under it, the model's picks against the analyst's
// with the error of each, a switch between networks and a threshold.
// Samples of the traces on screen are asked from Python (want / chunk).

const W = 1200;
const SPLITS = ["train", "validation", "test"];
const COMP = ["E", "N", "Z"];
const LOCAL_CSS = `
.pkx-pill { font: 12px ${FONT}; padding: 3px 12px; border: 1px solid ${C.grid};
  border-radius: 999px; background: ${C.bg}; color: ${C.ink}; cursor: pointer; }
.pkx-pill:hover { border-color: ${C.ink2}; }
.pkx-row { display: flex; align-items: center; gap: 8px; }
.pkx-num { font-size: 13px; min-width: 9ch; text-align: center;
  font-variant-numeric: tabular-nums; }
.pkx-thr { display: flex; align-items: center; gap: 8px; width: 190px; }
.pkx-thr .mls-scrub { flex: 1; }
.pkx-thr span { font-size: 15px; min-width: 3.6ch; font-variant-numeric: tabular-nums; }
`;

function hash01(i, seed) {
  let h = (Math.imul(i + 1, 0x9e3779b1) ^ Math.imul(seed + 7, 0x85ebca6b)) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d) >>> 0;
  h = Math.imul(h ^ (h >>> 12), 0x297a2d39) >>> 0;
  return ((h ^ (h >>> 15)) >>> 0) / 4294967296;
}
const median = (a) => {
  const s = a.filter((v) => isFinite(v)).sort((x, y) => x - y);
  return s.length ? (s[(s.length - 1) >> 1] + s[s.length >> 1]) / 2 : NaN;
};

function render({ model, el }) {
  const DATA = model.get("mode") === "data";
  // results mode takes the full cell width, unbounded, even if it grows past the
  // text below it on the slide
  const st = makeStage(el, W, DATA ? 1.35 : Infinity);
  const local = elt("style");
  local.textContent = LOCAL_CSS;
  st.stage.append(local);

  const colours = model.get("colours"), rate = model.get("rate"), LEN = model.get("length");
  const meta = model.get("meta"), ev = model.get("events"), stations = model.get("stations");
  const models = model.get("models");
  const N = meta.p.length, NE = ev.lon.length;

  // ---- state -------------------------------------------------------------------
  const splitOn = [true, true, true];
  let seed = 1, evSel = -1, sel = -1, sortKey = DATA ? "distance" : "p", listFilter = "all";
  let mi = 0, thr = model.get("threshold"), zoom = null, brush = null, scroll = 0;
  let hits = [], evOrder = [], tOrder = [], ranked = [];
  const byEvent = Array.from({ length: NE }, () => []);
  for (let i = 0; i < N; i++) byEvent[meta.event[i]].push(i);

  // ---- geometry ----------------------------------------------------------------
  const MAPBOX = { x: 0, y: 36, w: 396, h: 352 };
  const SEC = { l: 566, r: 1186, t: 54, b: 362 };
  const DETBOX = DATA ? { x: 0, y: 408, w: 1190, h: 250 } : { x: 428, y: 36, w: 764, h: 620 };
  const LIST = { x: 0, y: 36, w: 396, rows: 19, h: 24 };
  // results mode: the list and the probability panel end on the same line
  const LIST_BOTTOM = LIST.y + 28 + LIST.rows * LIST.h;
  const H = DATA ? 672 : LIST_BOTTOM + 52;
  function dg() {
    if (DATA) {
      const l = DETBOX.x + 232, r = DETBOX.x + DETBOX.w - 4;
      return { l, r, ct: DETBOX.y + 22, cb: DETBOX.y + DETBOX.h - 30, pt: 0, pb: 0 };
    }
    const l = DETBOX.x + 44, r = DETBOX.x + DETBOX.w - 6;
    const pb = LIST_BOTTOM, pt = pb - 130, cb = pt - 44, ct = cb - 3 * 62;
    return { l, r, ct, cb, pt, pb };
  }

  // ---- header bar --------------------------------------------------------------
  const bar = elt("div", "mls-bar");
  const status = elt("div", "mls-status");
  status.style.minWidth = "18em";
  const fig = stageCanvas(st, W, H);
  fig.canvas.tabIndex = 0;
  fig.canvas.style.outline = "none";
  const ctx = fig.ctx;
  let orderSeg, evNum, modelSeg, thrValue, thrScrub, sortSel, showSeg;
  if (DATA) {
    orderSeg = uiSegmented([["distance", "distance"], ["snr", "SNR"]], sortKey,
      (v) => { sortKey = v; buildTraceOrder(); draw(); });
    const prev = elt("button", "mls-pm", "‹"), next = elt("button", "mls-pm", "›");
    prev.title = "previous earthquake"; next.title = "next earthquake";
    prev.onclick = () => stepEvent(-1); next.onclick = () => stepEvent(1);
    evNum = elt("div", "pkx-num");
    const rnd = elt("button", "pkx-pill", "random earthquake");
    rnd.onclick = () => {   // a random other earthquake; the order (and so the number) stays fixed
      if (evOrder.length < 2) return;
      let e = evSel;
      while (e === evSel) e = evOrder[Math.floor(Math.random() * evOrder.length)];
      setEvent(e);
    };
    const pager = elt("div", "pkx-row");
    pager.append(prev, evNum, next, rnd);
    bar.append(uiField(uiLabel("EARTHQUAKE"), pager), elt("div", "mls-rule"),
      uiField(uiLabel("STATIONS ORDERED BY"), orderSeg), status);
  } else {
    if (models.length > 1) {
      modelSeg = uiSegmented(models.map((m, k) => [k, m.name]), mi, (v) => { mi = v; refresh(); });
      bar.append(uiField(uiLabel("NETWORK"), modelSeg), elt("div", "mls-rule"));
    }
    const box = elt("div", "pkx-thr");
    thrValue = elt("span", "", thr.toFixed(2));
    thrScrub = uiScrub((f) => {
      thr = Math.round((0.05 + f * 0.85) * 100) / 100;
      model.set("threshold", thr); model.save_changes();
      thrValue.textContent = thr.toFixed(2); thrScrub.set((thr - 0.05) / 0.85);
      refresh();
    });
    thrScrub.set((thr - 0.05) / 0.85);
    box.append(thrScrub.el, thrValue);
    sortSel = uiSelect([["p", "P error"], ["s", "S error"], ["distance", "distance"],
      ["snr", "SNR"], ["random", "random"]], sortKey, (v) => { sortKey = v; scroll = 0; refresh(); });
    showSeg = uiSegmented([["all", "all"], ["missed", "missed"], ["off", "> 1 s off"]], listFilter,
      (v) => { listFilter = v; scroll = 0; refresh(); });
    bar.append(uiField(uiLabel("PICK THRESHOLD"), box), elt("div", "mls-rule"),
      uiField(uiLabel("SORT BY LARGEST"), sortSel), uiField(uiLabel("SHOW"), showSeg), status);
  }
  st.stage.append(bar, fig.canvas);

  // ---- helpers -----------------------------------------------------------------
  const font = (px) => { ctx.font = `${px}px ${FONT}`; };
  const text = (s, x, y, colour = C.ink, align = "left", px = 11) => {
    font(px); ctx.fillStyle = colour; ctx.textAlign = align; ctx.textBaseline = "alphabetic";
    ctx.fillText(s, x, y);
  };
  const alpha = (hex, a) => {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${n >> 16}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
  };
  const niceTicks = (a, b, count = 6) => {
    const raw = (b - a) / count;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw);
    const out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(v);
    return out;
  };
  const P = colours.P, S = colours.S;
  const fmtErr = (e) => (isNaN(e) ? "missed" : `${e >= 0 ? "+" : "−"}${Math.abs(e).toFixed(2)} s`);

  // ---- picks of the current network (results mode) -------------------------------
  const pickAt = (i, ph) => (models[mi].peak[ph][i] >= thr ? models[mi].at[ph][i] : -1);
  const truth = (i, ph) => (ph ? meta.s[i] : meta.p[i]);
  const errOf = (i, ph) => { const a = pickAt(i, ph); return a < 0 ? NaN : (a - truth(i, ph)) / rate; };

  function buildRanking() {
    let idx = Array.from({ length: N }, (_, i) => i);
    if (listFilter === "missed") idx = idx.filter((i) => isNaN(errOf(i, 0)) || isNaN(errOf(i, 1)));
    if (listFilter === "off") idx = idx.filter((i) => Math.abs(errOf(i, 0)) > 1 || Math.abs(errOf(i, 1)) > 1);
    const bad = (i, ph) => { const e = Math.abs(errOf(i, ph)); return isNaN(e) ? -Infinity : -e; };
    const key = { p: (i) => bad(i, 0), s: (i) => bad(i, 1), distance: (i) => meta.distance[i],
      snr: (i) => meta.snr[i], random: (i) => hash01(i, seed) }[sortKey];
    idx.sort((a, b) => key(a) - key(b) || a - b);
    ranked = idx;
  }

  // ---- earthquakes (data mode) ---------------------------------------------------
  function buildEventOrder() {
    evOrder = [];
    for (let e = 0; e < NE; e++) if (splitOn[ev.split[e]] && byEvent[e].length >= 1) evOrder.push(e);
    // fixed order (by event number), so "k / n" always names the same earthquake
  }
  function buildTraceOrder() {
    tOrder = byEvent[evSel].slice();
    tOrder.sort(sortKey === "snr" ? (a, b) => meta.snr[b] - meta.snr[a]
      : (a, b) => meta.distance[a] - meta.distance[b]);
    if (!tOrder.includes(sel)) setSel(tOrder[0]);
  }
  function setEvent(e) {
    if (e === undefined) return;
    evSel = e; zoom = null; buildTraceOrder(); setSel(tOrder[0]); draw();
  }
  function stepEvent(d) {
    const k = evOrder.indexOf(evSel);
    setEvent(evOrder[(k + d + evOrder.length) % evOrder.length]);
  }
  function setSel(i) {
    if (i === sel) return;
    sel = i; zoom = null;
    model.set("selected", i); model.save_changes();
  }

  // ---- samples from Python -------------------------------------------------------
  const records = new Map();  // trace -> { q: Int16Array(3 * LEN), scale, prob: {name: Uint8Array(2 * LEN)} }
  const asked = new Set();
  let nonce = 0;
  function ask(ids) {
    const missing = ids.filter((i) => !records.has(i) && !asked.has(i));
    if (!missing.length) return;
    missing.forEach((i) => asked.add(i));
    if (window.PKX_DEBUG) console.log(`PKX ask ${missing.length} t=${performance.now().toFixed(0)}`);
    // a changing negative entry, ignored by Python, so a repeated request still counts as a change
    model.set("want", [...missing, -(++nonce)]);
    model.save_changes();
  }
  const bytesOf = (b64) => {
    const raw = atob(b64), out = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i);
    return out;
  };
  // the samples travel as 8-bit mu-law, one scale per trace (see pickerwidget.py)
  const MU = 255, LUT = new Float32Array(255);
  for (let q = -127; q <= 127; q++) {
    LUT[q + 127] = Math.sign(q) * (Math.pow(1 + MU, Math.abs(q) / 127) - 1) / MU * 32767;
  }
  function addRecords(ids, samples, scale, probs) {
    const q8 = new Int8Array(bytesOf(samples).buffer), n3 = 3 * LEN;
    const prob = {};
    for (const name in probs || {}) prob[name] = bytesOf(probs[name]);
    ids.forEach((i, j) => {
      const q = new Float32Array(n3);
      for (let k = 0; k < n3; k++) q[k] = LUT[q8[j * n3 + k] + 127];
      const rec = { q, scale: scale[j], prob: {} };
      for (const name in prob) rec.prob[name] = prob[name].subarray(j * 2 * LEN, (j + 1) * 2 * LEN);
      records.set(i, rec);
    });
  }
  function takeChunk() {
    const ch = model.get("chunk");
    if (!ch.indices) return;
    const t0 = performance.now();
    addRecords(ch.indices, ch.samples, ch.scale, ch.prob);
    const t1 = performance.now();
    draw();
    if (window.PKX_DEBUG) console.log(`PKX chunk n=${ch.indices.length} t=${t0.toFixed(0)} decode=${(t1 - t0).toFixed(0)}ms draw=${(performance.now() - t1).toFixed(0)}ms`);
  }

  // ---- drawing -----------------------------------------------------------------
  function frame(l, t, r, b) { ctx.strokeStyle = C.ink2; ctx.lineWidth = 1; ctx.strokeRect(l + 0.5, t + 0.5, r - l, b - t); }
  function timeAxis(l, r, y, v0, v1, label) {
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(l, y); ctx.lineTo(r, y); ctx.stroke();
    for (const v of niceTicks(v0, v1)) {
      const x = l + (v - v0) / (v1 - v0) * (r - l);
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y - 4); ctx.stroke();
      text(String(+v.toFixed(2)), x, y + 13, C.ink2, x > r - 8 ? "right" : "center", 10);
    }
    text(label, (l + r) / 2, y + 28, C.ink2, "center", 10.5);
  }
  // a vertical pick line with a halo in the background colour, so that it holds up
  // against the black waveforms it crosses
  function pickLine(x, y0, y1, colour, width, dash = [], withHalo = true) {
    ctx.setLineDash(dash);
    for (const [c, w] of [[alpha(C.bg, 0.9), withHalo ? width + 3 : 0], [colour, width]]) {
      if (w === 0) continue;
      ctx.strokeStyle = c; ctx.lineWidth = w;
      ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke();
    }
    ctx.setLineDash([]);
  }
  // the largest sample of each block of `step`, so that spikes survive the thinning
  function wiggle(rec, comp, i0, i1, xOf, ymid, perUnit, colour) {
    const n = i1 - i0 + 1, step = Math.max(1, Math.floor(n / (Math.abs(xOf(i1) - xOf(i0)) * 1.3)));
    ctx.beginPath();
    let first = true;
    for (let i = i0; i <= i1; i += step) {
      let best = i, mag = -1;
      for (let j = i; j < Math.min(i + step, i1 + 1); j++) {
        const m = Math.abs(rec.q[comp * LEN + j]);
        if (m > mag) { mag = m; best = j; }
      }
      const x = xOf(best), y = ymid - rec.q[comp * LEN + best] * perUnit;
      first ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      first = false;
    }
    ctx.strokeStyle = colour; ctx.lineWidth = 0.8; ctx.stroke();
  }
  const zero = (i) => (DATA ? meta.p[i] : 0);
  const fullRange = (i) => [(0 - zero(i)) / rate, (LEN - 1 - zero(i)) / rate];
  const clampI = (i) => Math.min(Math.max(Math.round(i), 0), LEN - 1);

  // -- the map: coast and faults, the earthquakes that have records, the selected one
  const decodeLines = (lines) => (lines || []).map((flat) => {
    const pts = [];
    let x = flat[0], y = flat[1];
    pts.push([x / 1000, y / 1000]);
    for (let i = 2; i < flat.length; i += 2) { x += flat[i]; y += flat[i + 1]; pts.push([x / 1000, y / 1000]); }
    return pts;
  });
  const base = model.get("basemap") || {};
  const coast = decodeLines(base.coast);
  const faults = Object.entries(base.faults || {}).map(([rate, lines]) => [Number(rate), decodeLines(lines)]);
  let mapZoom = false;  // the map opens on the overview; the switch frames the selection
  const geo = {};  // the current view, set by fitView
  const lonAll = [...ev.lon, ...stations.map((s_) => s_[1])], latAll = [...ev.lat, ...stations.map((s_) => s_[2])];
  // The frame has one size and shape, that of the overview, so the map does not change
  // shape between the two views or from one earthquake to the next. "zoom to selection"
  // goes in as far as the frame allows: the earthquake and all its stations, with just
  // enough room (in pixels) around them for the ring and the station names.
  const lonRange = [Math.min(...lonAll), Math.max(...lonAll)];
  const latRange = [Math.min(...latAll), Math.max(...latAll)];
  const kxAll = Math.cos(((latRange[0] + latRange[1]) / 2) * Math.PI / 180);
  const frameScale = Math.min((MAPBOX.w - 48) / ((lonRange[1] - lonRange[0]) * kxAll),
    (MAPBOX.h - 64) / (latRange[1] - latRange[0]));
  const FRAME = { w: (lonRange[1] - lonRange[0]) * kxAll * frameScale, h: (latRange[1] - latRange[0]) * frameScale };
  const EDGE = { l: 18, r: 46, t: 22, b: 22 };  // free pixels; names are written to the right
  const MIN_SPAN = 0.15;  // degrees: never closer than this, whatever the stations
  function fitView() {
    let lo0 = lonRange[0], la0 = latRange[0], scale = frameScale;
    geo.kx = kxAll;
    if (mapZoom && evSel >= 0) {  // the selected earthquake and its stations
      const lons = [ev.lon[evSel]], lats = [ev.lat[evSel]];
      for (const i of byEvent[evSel]) { const [, lo, la] = stations[meta.station[i]]; lons.push(lo); lats.push(la); }
      const [lonMin, lonMax, latMin, latMax] = [Math.min(...lons), Math.max(...lons), Math.min(...lats), Math.max(...lats)];
      geo.kx = Math.cos(((latMin + latMax) / 2) * Math.PI / 180);
      const dx = Math.max((lonMax - lonMin) * geo.kx, MIN_SPAN), dy = Math.max(latMax - latMin, MIN_SPAN);
      const freeW = FRAME.w - EDGE.l - EDGE.r, freeH = FRAME.h - EDGE.t - EDGE.b;
      scale = Math.min(freeW / dx, freeH / dy);  // pixels per degree of latitude
      // centred in the free area
      lo0 = (lonMin + lonMax) / 2 - (EDGE.l + freeW / 2) / (scale * geo.kx);
      la0 = (latMin + latMax) / 2 - (EDGE.b + freeH / 2) / scale;
    }
    geo.scale = scale;
    geo.w = FRAME.w; geo.h = FRAME.h;
    geo.l = MAPBOX.x + 44 + (MAPBOX.w - 48 - geo.w) / 2; geo.b = MAPBOX.y + 30 + geo.h;
    geo.lo0 = lo0; geo.la0 = la0;
  }
  const mx = (lon) => geo.l + (lon - geo.lo0) * geo.kx * geo.scale;
  const my = (lat) => geo.b - (lat - geo.la0) * geo.scale;
  const halo = (str, x, y, colour, align = "left", px = 10) => {
    font(px); ctx.textAlign = align; ctx.textBaseline = "alphabetic";
    ctx.lineWidth = 3; ctx.strokeStyle = C.bg; ctx.strokeText(str, x, y);
    ctx.fillStyle = colour; ctx.fillText(str, x, y);
  };
  function drawMap() {
    fitView();
    text("EARTHQUAKES AND STATIONS", 0, 12, C.ink2, "left", 10.5);
    SPLITS.forEach((name, s_) => {  // chips: the legend and the filter
      const x = [0, 58, 146][s_], y = 18, on = splitOn[s_];
      ctx.fillStyle = alpha(colours.split[s_], on ? 0.85 : 0.15);
      ctx.beginPath(); ctx.arc(x + 6, y + 9, 5, 0, 2 * Math.PI); ctx.fill();
      text(name, x + 16, y + 13, on ? C.ink : C.ink2, "left", 11);
      hits.push({ x, y, w: s_ === 1 ? 84 : 52, h: 18, type: "split", i: s_ });
    });
    // a switch that frames the selected earthquake and its stations
    const zx = MAPBOX.w - 118, zy = 18;
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1;
    ctx.fillStyle = mapZoom ? C.ink : C.bg;
    ctx.beginPath(); ctx.roundRect(zx, zy, 118, 20, 10); ctx.fill(); ctx.stroke();
    text("zoom to selection", zx + 59, zy + 14, mapZoom ? C.bg : C.ink, "center", 11);
    hits.push({ x: zx, y: zy, w: 118, h: 20, type: "zoom" });

    const l = geo.l, r = geo.l + geo.w, t = geo.b - geo.h, b = geo.b;
    ctx.save();
    ctx.beginPath(); ctx.rect(l, t, geo.w, geo.h); ctx.clip();
    const line = (pts) => {
      ctx.beginPath();
      pts.forEach(([lo, la], k) => (k ? ctx.lineTo(mx(lo), my(la)) : ctx.moveTo(mx(lo), my(la))));
      ctx.stroke();
    };
    ctx.lineJoin = "round";
    for (const [rate, lines] of faults) {  // faults: dashed, the fast ones heavier
      ctx.strokeStyle = alpha(C.ink2, rate >= 5 ? 0.85 : 0.5); ctx.lineWidth = rate >= 5 ? 1.3 : 0.8;
      ctx.setLineDash([4, 2]); lines.forEach(line); ctx.setLineDash([]);
    }
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.1;
    coast.forEach(line);
    const used = new Set(evSel >= 0 ? byEvent[evSel].map((i) => meta.station[i]) : []);
    stations.forEach(([, lon, lat], k) => {  // stations: small, until selected
      if (used.has(k)) return;
      const x = mx(lon), y = my(lat), q = 2.4;
      ctx.fillStyle = alpha(C.blue, 0.3);
      ctx.beginPath(); ctx.moveTo(x, y - q); ctx.lineTo(x + q, y + q * 0.8); ctx.lineTo(x - q, y + q * 0.8);
      ctx.closePath(); ctx.fill();
    });
    for (let e = 0; e < NE; e++) {  // earthquakes without records: faint; with records: coloured by set
      if (!splitOn[ev.split[e]] || byEvent[e].length) continue;
      ctx.fillStyle = alpha(C.ink2, 0.2);
      ctx.beginPath(); ctx.arc(mx(ev.lon[e]), my(ev.lat[e]), 0.9, 0, 2 * Math.PI); ctx.fill();
    }
    for (let e = 0; e < NE; e++) {
      if (!splitOn[ev.split[e]] || !byEvent[e].length || e === evSel) continue;
      ctx.fillStyle = alpha(colours.split[ev.split[e]], 0.5);
      ctx.strokeStyle = C.bg; ctx.lineWidth = 0.6;
      ctx.beginPath(); ctx.arc(mx(ev.lon[e]), my(ev.lat[e]), 1.8 + 0.5 * Math.max(ev.mag[e], 0), 0, 2 * Math.PI);
      ctx.fill(); ctx.stroke();
    }
    if (evSel >= 0) {
      const x = mx(ev.lon[evSel]), y = my(ev.lat[evSel]);
      ctx.lineWidth = 1;
      for (const i of byEvent[evSel]) {  // the stations that recorded it, joined to it and named
        const [name, lon, lat] = stations[meta.station[i]], sx = mx(lon), sy = my(lat);
        ctx.strokeStyle = alpha(C.ink, i === sel ? 0.9 : 0.5); ctx.lineWidth = i === sel ? 1.6 : 0.8;
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(sx, sy); ctx.stroke();
        ctx.fillStyle = i === sel ? C.orange : C.blue; ctx.strokeStyle = C.bg; ctx.lineWidth = 1;
        const q = i === sel ? 6 : 5;
        ctx.beginPath(); ctx.moveTo(sx, sy - q); ctx.lineTo(sx + q, sy + q * 0.8); ctx.lineTo(sx - q, sy + q * 0.8);
        ctx.closePath(); ctx.fill(); ctx.stroke();
        halo(name, sx + 7, sy + 4, i === sel ? C.orange : C.ink, "left", 9.5);
      }
      ctx.fillStyle = alpha(colours.split[ev.split[evSel]], 1); ctx.strokeStyle = C.bg; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(x, y, 5 + 0.7 * Math.max(ev.mag[evSel], 0), 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      ctx.strokeStyle = C.bg; ctx.lineWidth = 5;
      ctx.beginPath(); ctx.arc(x, y, 11 + 0.7 * Math.max(ev.mag[evSel], 0), 0, 2 * Math.PI); ctx.stroke();
      ctx.strokeStyle = C.orange; ctx.lineWidth = 2.4;
      ctx.beginPath(); ctx.arc(x, y, 11 + 0.7 * Math.max(ev.mag[evSel], 0), 0, 2 * Math.PI); ctx.stroke();
    }
    ctx.restore();
    frame(l, t, r, b);  // the frame goes on top, so nothing hangs over its edge
    ctx.lineWidth = 1;
    const ticks = 5;
    for (const v of niceTicks(geo.lo0, geo.lo0 + geo.w / (geo.kx * geo.scale), ticks)) {
      const x = mx(v); ctx.beginPath(); ctx.moveTo(x, b); ctx.lineTo(x, b - 4); ctx.stroke();
      text(v.toFixed(1), x, b + 13, C.ink2, "center", 10);
    }
    for (const v of niceTicks(geo.la0, geo.la0 + geo.h / geo.scale, ticks)) {
      const y = my(v); ctx.beginPath(); ctx.moveTo(l, y); ctx.lineTo(l + 4, y); ctx.stroke();
      text(v.toFixed(1), l - 5, y + 3.5, C.ink2, "right", 10);
    }
    text("longitude (deg)", l + geo.w / 2, b + 27, C.ink2, "center", 10.5);
    ctx.save(); ctx.translate(8, t + geo.h / 2); ctx.rotate(-Math.PI / 2);
    text("latitude (deg)", 0, 0, C.ink2, "center", 10.5); ctx.restore();
    if (evSel >= 0) {  // what is selected, in words
      text(`${ev.id[evSel]}   M ${ev.mag[evSel].toFixed(1)}   ${byEvent[evSel].length} stations`,
        l, t - 6, C.orange, "left", 10.5);
    }
    // the key for the grey layers, and for the dots without records
    const kx_ = l, ky_ = b + 44;
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.1; ctx.beginPath(); ctx.moveTo(kx_, ky_ - 3); ctx.lineTo(kx_ + 22, ky_ - 3); ctx.stroke();
    text("coast", kx_ + 27, ky_, C.ink2, "left", 10);
    ctx.strokeStyle = alpha(C.ink2, 0.85); ctx.lineWidth = 1.3; ctx.setLineDash([4, 2]);
    ctx.beginPath(); ctx.moveTo(kx_ + 70, ky_ - 3); ctx.lineTo(kx_ + 92, ky_ - 3); ctx.stroke(); ctx.setLineDash([]);
    text("faults (USGS)", kx_ + 97, ky_, C.ink2, "left", 10);
    ctx.fillStyle = alpha(C.ink2, 0.4); ctx.beginPath(); ctx.arc(kx_ + 196, ky_ - 3, 2, 0, 2 * Math.PI); ctx.fill();
    text("recorded by 1 or 2 stations", kx_ + 203, ky_, C.ink2, "left", 10);
    hits.push({ x: l, y: t, w: geo.w, h: geo.h, type: "map" });
  }

  // -- the record section
  const SEC_T = [-5, 30];
  function drawSection() {
    const e = evSel, n = tOrder.length;
    const sp = SPLITS[ev.split[e]];
    text(`RECORD SECTION: EARTHQUAKE ${ev.id[e]}, M ${ev.mag[e].toFixed(1)}, ${n} stations, ${sp}`,
      SEC.l - 130, 12, C.ink2, "left", 10.5);
    text("vertical component, one row per station, aligned on the analyst P pick", SEC.l - 130, 28, C.ink2, "left", 10.5);
    const pitch = Math.min(30, (SEC.b - SEC.t) / n), l = SEC.l, r = SEC.r;
    const xT = (t) => l + (t - SEC_T[0]) / (SEC_T[1] - SEC_T[0]) * (r - l);
    ask(tOrder);
    const sTicks = [];
    tOrder.forEach((i, row) => {
      const y = SEC.t + (row + 0.5) * pitch, rec = records.get(i);
      if (i === sel) { ctx.fillStyle = alpha(C.orange, 0.1); ctx.fillRect(SEC.l - 130, y - pitch / 2, r - SEC.l + 130, pitch); }
      text(`${stations[meta.station[i]][0]}  ${meta.distance[i].toFixed(0)} km`, l - 8, y + 3.5, C.ink, "right", 10.5);
      const sT = (meta.s[i] - meta.p[i]) / rate;
      if (rec) {  // black, so that the picks (drawn on top) stand out
        const i0 = clampI(meta.p[i] + SEC_T[0] * rate), i1 = clampI(meta.p[i] + SEC_T[1] * rate);
        let amp = 1;
        for (let j = i0; j <= i1; j++) amp = Math.max(amp, Math.abs(rec.q[2 * LEN + j]));
        wiggle(rec, 2, i0, i1, (j) => xT((j - meta.p[i]) / rate), y, pitch * 0.45 / amp, C.ink);
      }
      pickLine(xT(0), y - pitch * 0.5, y + pitch * 0.5, P, 2.8);
      if (sT <= SEC_T[1]) {
        pickLine(xT(sT), y - pitch * 0.5, y + pitch * 0.5, S, 2.8);
        sTicks.push([xT(sT), y]);
      }
      hits.push({ x: SEC.l - 130, y: y - pitch / 2, w: r - SEC.l + 130, h: pitch, type: "trace", i });
    });
    if (sortKey === "distance" && sTicks.length > 1) {  // the S - P moveout, joined
      ctx.strokeStyle = alpha(S, 0.55); ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
      ctx.beginPath();
      sTicks.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.stroke(); ctx.setLineDash([]);
    }
    const bottom = SEC.t + n * pitch + 4;
    timeAxis(l, r, bottom, SEC_T[0], SEC_T[1], "time relative to the analyst P pick (s)");
  }

  // -- the list of test traces
  function drawList() {
    const { x, y, w, rows, h } = LIST;
    text(`TEST TRACES (${ranked.length})`, 0, 12, C.ink2, "left", 10.5);
    text("sorted by " + { p: "largest P error", s: "largest S error", distance: "distance",
      snr: "SNR", random: "a random order" }[sortKey], 0, 28, C.ink2, "left", 10.5);
    // each column is right-aligned at its x, header and values alike
    const cols = [28, 104, 196, 290, 384];
    ["#", "station", "dist (km)", "P error", "S error"].forEach((c, k) =>
      text(c, x + cols[k] - (k === 1 ? 44 : 0), y + 16, C.ink2, k === 1 ? "left" : "right", 10));
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, y + 24.5); ctx.lineTo(x + w, y + 24.5); ctx.stroke();
    scroll = Math.max(0, Math.min(scroll, Math.max(0, ranked.length - rows)));
    const shown = ranked.slice(scroll, scroll + rows);
    ask(shown.concat(sel >= 0 ? [sel] : []));
    shown.forEach((i, k) => {
      const yy = y + 28 + k * h;
      if (i === sel) { ctx.fillStyle = alpha(C.orange, 0.12); ctx.fillRect(x, yy, w, h); }
      text(String(i), x + cols[0], yy + 16, C.ink2, "right", 10.5);
      text(stations[meta.station[i]][0], x + cols[1] - 44, yy + 16, C.ink, "left", 10.5);
      text(meta.distance[i].toFixed(0), x + cols[2], yy + 16, C.ink, "right", 10.5);
      [0, 1].forEach((ph) => {
        const e = errOf(i, ph);
        text(fmtErr(e), x + cols[3 + ph], yy + 16, isNaN(e) || Math.abs(e) > 1 ? C.orange : C.ink, "right", 10.5);
      });
      hits.push({ x, y: yy, w, h, type: "trace", i });
    });
    if (ranked.length > rows) {  // a thin scroll bar
      const f0 = scroll / ranked.length, f1 = (scroll + rows) / ranked.length, hh = rows * h;
      ctx.fillStyle = C.grid; ctx.fillRect(w + 8, y + 28, 3, hh);
      ctx.fillStyle = C.ink2; ctx.fillRect(w + 8, y + 28 + f0 * hh, 3, (f1 - f0) * hh);
    }
  }

  // -- the open trace
  function drawDetail() {
    const g = dg();
    if (sel < 0) return;
    const rec = records.get(sel), [f0, f1] = fullRange(sel), [v0, v1] = zoom || [f0, f1];
    const xT = (t) => g.l + (t - v0) / (v1 - v0) * (g.r - g.l);
    const i0 = clampI(zero(sel) + v0 * rate), i1 = clampI(zero(sel) + v1 * rate);
    const xI = (j) => xT((j - zero(sel)) / rate);
    const st_ = stations[meta.station[sel]][0];
    const sp = SPLITS[meta.split[sel]], sMinusP = (meta.s[sel] - meta.p[sel]) / rate;
    const facts = [`station ${st_}`, `earthquake ${ev.id[meta.event[sel]]}`, `magnitude ${meta.mag[sel].toFixed(1)}`,
      `distance ${meta.distance[sel].toFixed(0)} km`, `S − P  ${sMinusP.toFixed(1)} s`,
      `SNR ${meta.snr[sel].toFixed(1)}`, `split: ${sp}`];
    text(DATA ? "SELECTED TRACE" : `TRACE #${sel}`, DETBOX.x, DETBOX.y + 8, C.ink2, "left", 10.5);
    if (DATA) facts.forEach((f, k) => text(f, DETBOX.x, DETBOX.y + 34 + k * 17, C.ink, "left", 11.5));
    else {  // the facts, spaced by their measured widths
      let fx = DETBOX.x;
      font(11.5);
      for (const f of facts) { text(f, fx, DETBOX.y + 30, C.ink, "left", 11.5); fx += ctx.measureText(f).width + 26; }
    }
    if (!DATA) {
      const e0 = errOf(sel, 0), e1 = errOf(sel, 1);
      text(`P error ${fmtErr(e0)}`, DETBOX.x, DETBOX.y + 58, isNaN(e0) || Math.abs(e0) > 1 ? C.orange : P, "left", 12);
      text(`S error ${fmtErr(e1)}`, DETBOX.x + 160, DETBOX.y + 58, isNaN(e1) || Math.abs(e1) > 1 ? C.orange : S, "left", 12);
      // legend of the two kinds of line
      const ly = DETBOX.y + 84;
      ctx.lineWidth = 1.4; ctx.strokeStyle = C.ink2;
      ctx.setLineDash([1.5, 3]); ctx.beginPath(); ctx.moveTo(DETBOX.x, ly); ctx.lineTo(DETBOX.x + 26, ly); ctx.stroke();
      ctx.setLineDash([7, 3]); ctx.beginPath(); ctx.moveTo(DETBOX.x + 160, ly); ctx.lineTo(DETBOX.x + 186, ly); ctx.stroke();
      ctx.setLineDash([]);
      text("analyst", DETBOX.x + 32, ly + 3.5, C.ink2, "left", 10.5);
      text(`${models[mi].name} network`, DETBOX.x + 192, ly + 3.5, C.ink2, "left", 10.5);
    }
    const rowH = (g.cb - g.ct) / 3;
    // the three components share one amplitude scale
    if (rec) {
      let amp = 1;
      for (let c = 0; c < 3; c++) for (let j = i0; j <= i1; j++) amp = Math.max(amp, Math.abs(rec.q[c * LEN + j]));
      for (let c = 0; c < 3; c++) {
        const ymid = g.ct + (c + 0.5) * rowH;
        wiggle(rec, c, i0, i1, xI, ymid, rowH * 0.46 / amp, C.ink);
        text(COMP[c], g.l - 8, ymid + 4, C.ink, "right", 12);
      }
    } else text("loading ...", (g.l + g.r) / 2, (g.ct + g.cb) / 2, C.ink2, "center", 12);
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(g.l, g.ct); ctx.lineTo(g.l, g.cb); ctx.stroke();
    // the analyst picks (dotted) through every panel; the network's (dashed) in results mode
    const bottom = DATA ? g.cb : g.pb;
    const line = (t, colour, dash, top, base, width = 1.6) => {
      if (t < v0 || t > v1) return;
      pickLine(xT(t), top, base, colour, width, dash, width > 2);  // halo: analyst picks only
    };
    const aP = (meta.p[sel] - zero(sel)) / rate, aS = (meta.s[sel] - zero(sel)) / rate;
    line(aP, P, [2, 3], g.ct, bottom, 2.6); line(aS, S, [2, 3], g.ct, bottom, 2.6);
    for (const [t, col] of [[aP, P], [aS, S]]) {  // a flag on top of each analyst pick
      if (t < v0 || t > v1) continue;
      const x = xT(t);
      ctx.fillStyle = col;
      ctx.beginPath(); ctx.moveTo(x - 5.5, g.ct - 9); ctx.lineTo(x + 5.5, g.ct - 9); ctx.lineTo(x, g.ct); ctx.closePath(); ctx.fill();
    }
    if (DATA) { halo("analyst P", xT(aP) + 7, g.ct + 12, P, "left", 11.5); halo("analyst S", xT(aS) + 7, g.ct + 12, S, "left", 11.5); }
    if (DATA) { timeAxis(g.l, g.r, g.cb + 4, v0, v1, "time relative to the analyst P pick (s)"); }
    else {
      const m = models[mi], prob = rec && rec.prob[m.name];
      text("PROBABILITY", g.l - 40, g.pt - 14, C.ink2, "left", 10.5);
      ctx.strokeStyle = C.ink2; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(g.l, g.pt - 4); ctx.lineTo(g.l, g.pb); ctx.lineTo(g.r, g.pb); ctx.stroke();
      for (const v of [0, 0.5, 1]) {
        const y = g.pb - v * (g.pb - g.pt);
        ctx.beginPath(); ctx.moveTo(g.l, y); ctx.lineTo(g.l + 4, y); ctx.stroke();
        text(String(v), g.l - 6, y + 3.5, C.ink2, "right", 10);
      }
      const yT = g.pb - thr * (g.pb - g.pt);  // the threshold
      ctx.strokeStyle = C.ink2; ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(g.l, yT); ctx.lineTo(g.r, yT); ctx.stroke(); ctx.setLineDash([]);
      text(`threshold ${thr.toFixed(2)}`, g.r - 2, yT - 4, C.ink2, "right", 10);
      if (prob) {
        [[0, P], [1, S]].forEach(([ph, col]) => {
          ctx.beginPath();
          const step = Math.max(1, Math.floor((i1 - i0) / ((g.r - g.l) * 1.3)));
          let first = true;
          for (let j = i0; j <= i1; j += step) {
            let best = j, hi = -1;
            for (let k = j; k < Math.min(j + step, i1 + 1); k++) if (prob[ph * LEN + k] > hi) { hi = prob[ph * LEN + k]; best = k; }
            const x = xI(best), y = g.pb - prob[ph * LEN + best] / 255 * (g.pb - g.pt);
            first ? ctx.moveTo(x, y) : ctx.lineTo(x, y); first = false;
          }
          ctx.strokeStyle = col; ctx.lineWidth = 1.4; ctx.stroke();
        });
      }
      [[0, P, "P"], [1, S, "S"]].forEach(([ph, col, nm]) => {
        const at = m.at[ph][sel], peak = m.peak[ph][sel], tt = (at - zero(sel)) / rate, picked = peak >= thr;
        if (tt < v0 || tt > v1) return;
        line(tt, col, picked ? [7, 3] : [2, 4], g.ct, g.pb);
        ctx.fillStyle = picked ? col : C.bg; ctx.strokeStyle = col; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.arc(xT(tt), g.pb - peak * (g.pb - g.pt), 4, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
        const e = errOf(sel, ph);
        const lab = picked ? `${nm} ${fmtErr(e)}` : `${nm} below threshold`;
        font(10.5);
        const flip = xT(tt) + 12 + ctx.measureText(lab).width > g.r;
        text(lab, xT(tt) + (flip ? -5 : 5), g.ct - 8 - 14 * ph, col, flip ? "right" : "left", 10.5);
      });
      timeAxis(g.l, g.r, g.pb, v0, v1, "time in the window (s)");
    }
    text(zoom ? "click to reset the zoom" : "drag to zoom", g.r, DETBOX.y + 8, C.ink2, "right", 10);
    if (brush) {
      ctx.fillStyle = alpha(C.ink2, 0.15);
      const a = Math.min(brush[0], brush[1]), b = Math.max(brush[0], brush[1]);
      ctx.fillRect(a, g.ct, b - a, bottom - g.ct);
    }
    if (rec) ask([sel]);
    hits.push({ x: g.l, y: g.ct, w: g.r - g.l, h: bottom - g.ct, type: "plot" });
  }

  // ---- whole figure --------------------------------------------------------------
  function syncHeader() {
    if (DATA) {
      const k = evOrder.indexOf(evSel);
      evNum.textContent = k >= 0 ? `${k + 1} / ${evOrder.length}` : `— / ${evOrder.length}`;
      status.textContent = `${evOrder.length} of ${NE} earthquakes have 3 or more stations`;
    } else {
      const p = Array.from({ length: N }, (_, i) => errOf(i, 0)), s = Array.from({ length: N }, (_, i) => errOf(i, 1));
      const med = (a) => median(a.map(Math.abs));
      const miss = (a) => (100 * a.filter(isNaN).length / N).toFixed(0);
      status.textContent = `P: median |error| ${med(p).toFixed(2)} s, ${miss(p)}% missed;  S: ${med(s).toFixed(2)} s, ${miss(s)}% missed`;
    }
  }
  function draw() {
    ctx.clearRect(0, 0, W, H);
    hits = [];
    if (DATA) { drawMap(); drawSection(); } else drawList();
    drawDetail();
    syncHeader();
  }
  function refresh() { if (!DATA) buildRanking(); draw(); }

  // ---- interaction ---------------------------------------------------------------
  const hitAt = ([x, y]) => hits.slice().reverse().find((h) => x >= h.x && x <= h.x + h.w && y >= h.y && y <= h.y + h.h);
  const inPlot = (p) => { const h = hitAt(p); return h && h.type === "plot"; };
  fig.canvas.addEventListener("click", (ev_) => {
    const p = fig.xy(ev_), h = hitAt(p);
    if (!h) return;
    if (h.type === "split") {
      splitOn[h.i] = !splitOn[h.i];
      if (!splitOn.some(Boolean)) splitOn.fill(true);
      buildEventOrder();
      if (!evOrder.includes(evSel)) setEvent(evOrder[0]); else draw();
    } else if (h.type === "zoom") { mapZoom = !mapZoom; draw();
    } else if (h.type === "trace") { setSel(h.i); draw(); }
    else if (h.type === "map") {
      let best = -1, d2 = 14 * 14;
      for (const e of evOrder.length ? evOrder : []) {
        const d = (mx(ev.lon[e]) - p[0]) ** 2 + (my(ev.lat[e]) - p[1]) ** 2;
        if (d < d2) { d2 = d; best = e; }
      }
      if (best >= 0) setEvent(best);
    }
  });
  fig.canvas.addEventListener("pointerdown", (ev_) => {
    fig.canvas.focus();
    const p = fig.xy(ev_);
    if (!inPlot(p) || sel < 0) return;
    fig.canvas.setPointerCapture(ev_.pointerId);
    brush = [p[0], p[0]];
  });
  fig.canvas.addEventListener("pointermove", (ev_) => {
    const p = fig.xy(ev_);
    if (brush) { const g = dg(); brush[1] = Math.min(Math.max(p[0], g.l), g.r); draw(); return; }
    const h = hitAt(p);
    fig.canvas.style.cursor = h ? (h.type === "plot" ? "crosshair" : "pointer") : "default";
  });
  const endBrush = () => {
    if (!brush) return;
    const g = dg(), [f0, f1] = fullRange(sel), [v0, v1] = zoom || [f0, f1];
    const a = Math.min(brush[0], brush[1]), b = Math.max(brush[0], brush[1]);
    const tOf = (x) => v0 + (x - g.l) / (g.r - g.l) * (v1 - v0);
    zoom = b - a > 4 ? [Math.max(tOf(a), f0), Math.min(tOf(b), f1)] : null;
    brush = null; draw();
  };
  fig.canvas.addEventListener("pointerup", endBrush);
  fig.canvas.addEventListener("pointercancel", endBrush);
  fig.canvas.addEventListener("wheel", (ev_) => {
    if (DATA) return;
    ev_.preventDefault();
    scroll += Math.sign(ev_.deltaY) * 3; draw();
  }, { passive: false });
  fig.canvas.addEventListener("keydown", (ev_) => {
    const list = DATA ? tOrder : ranked, k = list.indexOf(sel);
    const go = (d) => { const j = Math.min(Math.max(k + d, 0), list.length - 1); setSel(list[j]);
      if (!DATA) { if (j < scroll) scroll = j; if (j >= scroll + LIST.rows) scroll = j - LIST.rows + 1; } draw(); };
    const moves = { ArrowDown: () => go(1), ArrowUp: () => go(-1),
      ArrowRight: () => (DATA ? stepEvent(1) : go(1)), ArrowLeft: () => (DATA ? stepEvent(-1) : go(-1)) };
    if (!moves[ev_.key]) return;
    ev_.preventDefault(); moves[ev_.key]();
  });

  model.on("change:chunk", takeChunk);

  // ---- start ---------------------------------------------------------------------
  const store = model.get("store");
  if (store && store.samples) addRecords(Array.from({ length: N }, (_, i) => i), store.samples, store.scale, null);
  if (DATA) {
    buildEventOrder();
    // start on a well-recorded earthquake, so that the section is full
    const big = evOrder.slice().sort((a, b) => byEvent[b].length - byEvent[a].length).slice(0, 12);
    evSel = big[3] !== undefined ? big[3] : evOrder[0];
    buildTraceOrder(); setSel(tOrder[0]);
  } else {
    buildRanking();
    setSel(ranked[0]);
  }
  draw();
  // the first request can be sent before Python listens (or while it trains): ask for the
  // open trace again now and then, not for everything
  let tries = 0;
  const retry = setInterval(() => {
    if (sel < 0 || records.has(sel) || ++tries > 12) { clearInterval(retry); return; }
    asked.delete(sel); ask([sel]);
  }, 2000);
}

export default { render };
