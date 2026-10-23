// The Lecture 1 optimiser playground (see mapwidget.py); controls.js is put in
// front of this module (tokens C, FONT and the ui* controls). The header bar
// holds playback (reset, play/pause, step, a step counter) and every setting,
// in two groups: the scene (P speed, random field, pick noise) and the
// optimiser (method, step size, momentum, number of steps). Below: the two
// maps (P speed and misfit) sharing one draggable scene, a one-line key, the
// misfit curve of the run and three result tiles that follow the replay.
const HIT = 11;  // pick radius (logical px)

// Layout (logical px). Two maps side by side, the loss curve and results below.
const W = 1100;
const P = 300;                                  // side of one square map
const PANEL = 500;                              // width of one map panel
const ML = 60, MT = 30;                         // map plot origin inside its panel
const MAPS = [
  { key: "velocity", title: "P SPEED (km/s)", x0: 0 },
  { key: "misfit", title: "MISFIT  log₁₀ S (km²)", x0: PANEL },
];
const KEY_Y = MT + P + 56;                      // the one-line key under the maps
const LOSS = { l: ML, t: KEY_Y + 64, w: 600, h: 90 };
const STATS_X = 740;
const H = LOSS.t + LOSS.h + 60;

// The settings and their choices (defaults live in mapwidget.py).
const VELOCITIES = [["homogeneous", "homogeneous"], ["heterogeneous", "heterogeneous"]];
const FIELDS = [1, 2, 3, 4, 5, 6];
const SIGMAS = [0, 0.1, 0.3, 0.5, 1, 1.5];
const METHODS = [["SGD", "SGD"], ["Adam", "Adam"]];
const MOMENTA = [0, 0.3, 0.5, 0.7, 0.8, 0.9];
const LRS = [0.5, 1, 2, 3, 4, 5];
const STEPS = [5, 10, 25, 50, 100, 200, 500, 1000];

const fmtLoss = (v) => {
  const e = Math.floor(Math.log10(v) + 1e-9);
  return e >= -2 && e <= 3 ? String(+v.toPrecision(2)) : `${+(v / 10 ** e).toPrecision(1)}×10${sup(e)}`;
};
const opts = (vals) => vals.map((v) => [v, String(v)]);

function render({ model, el }) {
  const st = makeStage(el, W);

  // ---- state ------------------------------------------------------------------
  let drag = null;      // {kind, index, map}
  let live = null;      // local copy of the editable things while dragging
  let hover = null;     // {kind, index}
  let scrubbing = false;
  let pos = 0;          // replay position in steps (a float while playing)
  let step = 0;         // the step shown: floor(pos)
  let playing = false;  // the playback loop is running
  let autoplay = false;  // set by the first play; then every new run plays
  let pending = !model.get("run_id");  // scene or settings changed, run not yet here
  let lastT = 0;

  // ---- header bar -------------------------------------------------------------
  const bar = elt("div", "mls-bar");
  // all settings in one row; the status floats above it
  bar.style.cssText = "flex-wrap:nowrap;gap:10px 14px;position:relative;";
  const tp = uiTransport({
    onReset: rewind, onPlay: togglePlay, onStep: stepOnce,
    resetTitle: "back to the start of the run",
  });
  const stepStat = uiStat("STEP", "mls-counter");
  const setting = (name) => (v) => setInput(name, v);
  const velSel = uiSelect(VELOCITIES, model.get("velocity"), setting("velocity"));
  const fieldSel = uiSelect(opts(FIELDS), model.get("field"), setting("field"));
  const sigmaSel = uiSelect(opts(SIGMAS), model.get("sigma"), setting("sigma"));
  const methodSeg = uiSegmented(METHODS, model.get("method"), setting("method"));
  const lrSel = uiSelect(opts(LRS), model.get("lr"), setting("lr"));
  const momSel = uiSelect(opts(MOMENTA), model.get("momentum"), setting("momentum"));
  const stepsSel = uiSelect(opts(STEPS), model.get("steps"), setting("steps"));
  const fieldF = uiField(uiLabel("RANDOM FIELD"), fieldSel);
  const momF = uiField(uiLabel("MOMENTUM", "μ"), momSel);
  const status = elt("div", "mls-status");
  status.style.cssText = "position:absolute;right:0;top:-12px;min-width:0;margin:0;";
  const nText = elt("div", "mls-value");
  const stationPM = uiPlusMinus(() => setStations(+1), () => setStations(-1));
  const stationBox = elt("div");
  stationBox.style.cssText = "display:flex;align-items:center;gap:8px;height:26px;";
  stationBox.append(stationPM, nText);
  bar.append(tp.el, stepStat.el, elt("div", "mls-rule"),
    uiField(uiLabel("P SPEED"), velSel), fieldF,
    uiField(uiLabel("PICK NOISE", "σ", "s"), sigmaSel),
    uiField(uiLabel("STATIONS"), stationBox), elt("div", "mls-rule"),
    uiField(uiLabel("METHOD"), methodSeg), uiField(uiLabel("STEP SIZE", "η"), lrSel),
    momF, uiField(uiLabel("STEPS"), stepsSel), status);
  const scrub = uiScrub((f) => { pause(); setPos(f * nSteps()); });

  const body = elt("div");
  body.style.cssText = `position:relative;width:${W}px;height:${H}px;margin-top:8px;`;
  const cv = stageCanvas(st, W, H);
  const { canvas, ctx } = cv;
  body.append(canvas);
  st.stage.append(bar, scrub.el, body);
  st.onScale(() => { cv.resize(); draw(); });
  const put = (e, x, y) => body.appendChild(uiPlace(e, x, y));
  for (const map of MAPS) put(elt("div", "mls-head", map.title), map.x0 + ML, 4);
  put(elt("div", "mls-head", "MISFIT S (km²) DURING THE RUN"), LOSS.l, LOSS.t - 30);
  const errTrue = uiStat("ERROR TO THE TRUE SOURCE");
  const errGrid = uiStat("ERROR TO THE GRID MINIMUM");
  const misfitStat = uiStat(uiLabel("MISFIT", "S"));
  put(errTrue.el, STATS_X, LOSS.t - 30);
  put(errGrid.el, STATS_X, LOSS.t + 30);
  put(misfitStat.el, STATS_X, LOSS.t + 90);
  const syncFields = () => {
    fieldF.setOff(model.get("velocity") !== "heterogeneous");
    momF.setOff(model.get("method") !== "SGD");
    const n = model.get("stations").length;
    nText.textContent = String(n);
    stationPM.enable(n < model.get("max_stations"), n > model.get("min_stations"));
  };

  // one image and one colour bar per layer ("velocity", "misfit")
  const images = {};
  function loadImages() {
    for (const [name, layer] of Object.entries(model.get("layers"))) {
      const img = new Image(), cbar = new Image();
      img.onload = cbar.onload = draw;
      img.src = layer.png; cbar.src = layer.bar;
      images[name] = { img, bar: cbar };
    }
    draw();
  }

  const box = () => model.get("box");
  const nSteps = () => Math.max(model.get("path").length - 1, 0);
  const toPx = ([x, y], map) =>
    [map.x0 + ML + x / box() * P, MT + P - y / box() * P];
  const clamp = (v) => Math.min(box(), Math.max(0, v));
  const kmOf = (px, py, map) =>
    [(px - map.x0 - ML) / P * box(), (MT + P - py) / P * box()];
  const inMap = (px, py, map, slack = 0) =>
    px >= map.x0 + ML - slack && px <= map.x0 + ML + P + slack &&
    py >= MT - slack && py <= MT + P + slack;
  const inLoss = (px, py) =>
    px >= LOSS.l - 6 && px <= LOSS.l + LOSS.w + 6 &&
    py >= LOSS.t - 6 && py <= LOSS.t + LOSS.h + 6;

  function state() {
    return live || {
      stations: model.get("stations"),
      source: model.get("source"),
      start: model.get("start"),
    };
  }

  // ---- playback: a run of n steps plays in 2 to 6 seconds ---------------------
  const syncPlay = () => tp.setPlaying(playing || (pending && autoplay));
  function setPos(k) {
    pos = Math.min(Math.max(k, 0), nSteps());
    step = Math.floor(pos + 1e-9);
    draw();
  }
  function tick(now) {
    if (!playing) return;
    const n = nSteps(), dur = Math.min(2000 + 4 * n, 6000);
    pos = Math.min(pos + n * Math.min(now - lastT, 100) / dur, n);
    lastT = now;
    if (pos >= n) { playing = false; syncPlay(); }
    setPos(pos);
    if (playing) requestAnimationFrame(tick);
  }
  function play() {
    autoplay = true;
    if (!pending && !drag && nSteps() > 0) {
      if (pos >= nSteps()) setPos(0);
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
  function stepOnce() { pause(); setPos(step + 1); }
  function rewind() {
    playing = false;
    setPos(0);
    if (autoplay) play(); else syncPlay();
  }
  function newRun() {
    pending = false;
    playing = false;
    status.textContent = "";
    setPos(0);
    if (autoplay) play(); else syncPlay();
  }
  // every setting and every drag ends in a new run from Python
  function setInput(name, value) {
    model.set(name, value);
    model.save_changes();
    waitForRun();
  }
  // add the station farthest from all others (on a coarse grid), or drop the last
  function setStations(delta) {
    const stations = model.get("stations").map((p) => [...p]);
    if (delta < 0) {
      if (stations.length <= model.get("min_stations")) return;
      stations.pop();
    } else {
      if (stations.length >= model.get("max_stations")) return;
      let best = null, bd = -1;
      for (let i = 1; i < 12; i++) for (let j = 1; j < 12; j++) {
        const c = [i / 12 * box(), j / 12 * box()];
        const d = Math.min(...stations.map((p) => Math.hypot(p[0] - c[0], p[1] - c[1])));
        if (d > bd) { bd = d; best = c; }
      }
      stations.push(best);
    }
    model.set("stations", stations);
    model.save_changes();
    waitForRun();
  }
  function waitForRun() {
    pending = true;
    playing = false;
    status.textContent = "computing…";
    syncFields();
    syncPlay();
    draw();
  }

  // ---- drawing helpers ------------------------------------------------------
  function star(x, y, R, fill) {
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const a = -Math.PI / 2 + i * Math.PI / 5, r = i % 2 ? R * 0.42 : R;
      ctx[i ? "lineTo" : "moveTo"](x + r * Math.cos(a), y + r * Math.sin(a));
    }
    ctx.closePath(); ctx.fillStyle = fill; ctx.fill();
    ctx.lineWidth = 1.2; ctx.strokeStyle = "#fff"; ctx.stroke();
  }
  function triangle(x, y, s, fill) {
    ctx.beginPath();
    ctx.moveTo(x, y - 1.1 * s); ctx.lineTo(x + s, y + 0.8 * s); ctx.lineTo(x - s, y + 0.8 * s);
    ctx.closePath(); ctx.fillStyle = fill; ctx.fill();
    ctx.lineWidth = 1; ctx.strokeStyle = "#222"; ctx.stroke();
  }
  function startMark(x, y) {
    ctx.beginPath(); ctx.arc(x, y, 6, 0, 7); ctx.fillStyle = "#fff"; ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = C.orange; ctx.stroke();
  }
  function cross(x, y, r) {
    ctx.beginPath();
    ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r);
    ctx.moveTo(x - r, y + r); ctx.lineTo(x + r, y - r); ctx.stroke();
  }
  function polyline(points, map) {
    ctx.beginPath();
    points.forEach((p, i) => {
      const [x, y] = toPx(p, map);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  }
  const inside = (p) => p[0] >= 0 && p[0] <= box() && p[1] >= 0 && p[1] <= box();
  // first point where a path leaves the map (null if it stays inside)
  function firstExit(points) {
    for (let i = 1; i < points.length; i++) {
      if (inside(points[i]) || !inside(points[i - 1])) continue;
      const [ax, ay] = points[i - 1], [bx, by] = points[i];
      let t = 1;  // fraction of the segment at which it reaches the edge
      for (const [a, b] of [[ax, bx], [ay, by]]) {
        if (b < 0) t = Math.min(t, (0 - a) / (b - a));
        if (b > box()) t = Math.min(t, (box() - a) / (b - a));
      }
      return [ax + t * (bx - ax), ay + t * (by - ay)];
    }
    return null;
  }
  // triangle with its tip on the map edge, pointing in the direction of an off-map point
  function edgeArrow(p, map) {
    const q = [clamp(p[0]), clamp(p[1])];
    const [x, y] = toPx(q, map);
    const a = Math.atan2(-(p[1] - q[1]), p[0] - q[0]);  // canvas y points down
    ctx.save(); ctx.translate(x, y); ctx.rotate(a);
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(-12, -6); ctx.lineTo(-12, 6);
    ctx.closePath(); ctx.fillStyle = C.orange; ctx.fill();
    ctx.lineWidth = 1.2; ctx.strokeStyle = "#fff"; ctx.stroke(); ctx.restore();
  }
  function text(s, x, y, align = "left", colour = C.ink, size = 11, halo = false) {
    ctx.font = `${size}px ${FONT}`; ctx.fillStyle = colour; ctx.textAlign = align;
    if (halo) {
      ctx.lineWidth = 3.5; ctx.strokeStyle = "#fff"; ctx.lineJoin = "round";
      ctx.strokeText(s, x, y);
    }
    ctx.fillText(s, x, y);
  }

  function colourBar(layer, im, map) {
    // vertical bar right of the map, high values at the top
    const bx = map.x0 + ML + P + 12, bw = 10, by = MT, bh = P;
    if (im && im.bar.complete && im.bar.naturalWidth) {
      ctx.save(); ctx.translate(bx, by + bh); ctx.rotate(-Math.PI / 2);
      ctx.drawImage(im.bar, 0, 0, bh, bw);  // bar image runs low -> high
      ctx.restore();
    }
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 0.8; ctx.strokeRect(bx, by, bw, bh);
    for (const f of [0, 0.5, 1]) {
      const v = layer.lo + f * (layer.hi - layer.lo), y = by + bh * (1 - f);
      text(v.toFixed(1), bx + bw + 4, y + (f === 0 ? 0 : f === 1 ? 8 : 4));
      ctx.beginPath(); ctx.moveTo(bx + bw, y); ctx.lineTo(bx + bw - 3, y); ctx.stroke();
    }
  }

  function drawMap(map, s, layers) {
    const layer = layers[map.key], im = images[map.key];
    const left = map.x0 + ML;
    if (im && im.img.complete && im.img.naturalWidth) {
      ctx.imageSmoothingEnabled = true; ctx.drawImage(im.img, left, MT, P, P);
    } else { ctx.fillStyle = "#eee"; ctx.fillRect(left, MT, P, P); }
    ctx.save(); ctx.beginPath(); ctx.rect(left, MT, P, P); ctx.clip();
    // rays, grid minimum and path are stale while dragging: hide them
    if (!drag) {
      if (map.key === "velocity") {
        ctx.strokeStyle = "rgba(17,17,17,0.35)"; ctx.lineWidth = 0.7;
        for (const ray of model.get("rays")) polyline(ray, map);
      } else {
        const g = model.get("grid_min");
        if (g.length) {
          const [gx, gy] = toPx(g, map);
          ctx.strokeStyle = "#fff"; ctx.lineWidth = 4.5; cross(gx, gy, 6);
          ctx.strokeStyle = C.green; ctx.lineWidth = 2.5; cross(gx, gy, 6);
        }
      }
      const path = model.get("path");
      if (path.length) {
        const k = Math.min(step, path.length - 1);
        ctx.lineJoin = "round";
        if (k < path.length - 1) {  // the part not yet walked
          ctx.setLineDash([3, 3]); ctx.strokeStyle = "rgba(213,94,0,0.45)";
          ctx.lineWidth = 1.2; polyline(path.slice(k), map); ctx.setLineDash([]);
        }
        const walked = path.slice(0, k + 1);
        ctx.strokeStyle = "#fff"; ctx.lineWidth = 3.6; polyline(walked, map);
        ctx.strokeStyle = C.orange; ctx.lineWidth = 1.7; polyline(walked, map);
        if (nSteps() <= 20) {  // few steps: mark every iterate
          for (const q of walked) {
            const [qx, qy] = toPx(q, map);
            ctx.beginPath(); ctx.arc(qx, qy, 2.6, 0, 7); ctx.fillStyle = C.orange;
            ctx.fill(); ctx.lineWidth = 1; ctx.strokeStyle = "#fff"; ctx.stroke();
          }
        }
        const exit = firstExit(walked);
        if (exit) {  // where the walked path crossed the map edge
          const [qx, qy] = toPx(exit, map);
          ctx.beginPath(); ctx.arc(qx, qy, 4, 0, 7); ctx.fillStyle = "#fff"; ctx.fill();
          ctx.lineWidth = 1.7; ctx.strokeStyle = C.orange; ctx.stroke();
        }
        const now = path[k];
        if (inside(now)) {
          const [ex, ey] = toPx(now, map);
          ctx.beginPath(); ctx.arc(ex, ey, 4.5, 0, 7); ctx.fillStyle = C.orange;
          ctx.fill(); ctx.lineWidth = 1.2; ctx.strokeStyle = "#fff"; ctx.stroke();
        } else {  // off the map: arrow on the edge pointing towards the iterate
          edgeArrow(now, map);
        }
      }
    }
    for (const p of s.stations) { const [x, y] = toPx(p, map); triangle(x, y, 5.5, "#fff"); }
    const [sx, sy] = toPx(s.start, map); startMark(sx, sy);
    const [mx, my] = toPx(s.source, map); star(mx, my, 10, "#111");
    // hover ring around the thing that would be picked up
    const target = drag || hover;
    if (target) {
      const p = target.kind === "station" ? s.stations[target.index] : s[target.kind];
      const [x, y] = toPx(p, map);
      ctx.beginPath(); ctx.arc(x, y, 13, 0, 7);
      ctx.lineWidth = 1.5; ctx.strokeStyle = C.blue; ctx.stroke();
      if (drag) {
        const label = `(${p[0].toFixed(1)}, ${p[1].toFixed(1)}) km`;
        const rightHalf = x > left + P / 2;
        text(label, x + (rightHalf ? -18 : 18), y - 14, rightHalf ? "right" : "left",
          C.ink, 11, true);
      }
    }
    ctx.restore();

    // frame, ticks, labels (boxed look)
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1; ctx.strokeRect(left, MT, P, P);
    for (let v = 0; v <= box(); v += 20) {
      const [x, y] = toPx([v, v], map);
      text(String(v), x, MT + P + 14, "center");
      text(String(v), left - 6, y + 4, "right");
      ctx.beginPath();
      ctx.moveTo(x, MT + P); ctx.lineTo(x, MT + P - 4);
      ctx.moveTo(left, y); ctx.lineTo(left + 4, y); ctx.stroke();
    }
    text("x (km)", left + P / 2, MT + P + 30, "center");
    ctx.save(); ctx.translate(map.x0 + 10, MT + P / 2); ctx.rotate(-Math.PI / 2);
    text("y (km)", 0, 0, "center"); ctx.restore();
    if (layer) colourBar(layer, im, map);
  }


  function drawKey() {  // one line: markers, then how to edit the scene
    let x = ML;
    const y = KEY_Y;
    const item = (drawMark, s) => {
      drawMark(x + 8, y - 4);
      text(s, x + 22, y, "left", C.ink, 12);
      ctx.font = `12px ${FONT}`;
      x += 22 + ctx.measureText(s).width + 22;
    };
    item((a, b) => star(a, b, 7, "#111"), "true source");
    item((a, b) => startMark(a, b), "start");
    item((a, b) => triangle(a, b, 4.5, "#fff"), "station");
    item((a, b) => {
      ctx.strokeStyle = C.orange; ctx.lineWidth = 1.7;
      ctx.beginPath(); ctx.moveTo(a - 8, b); ctx.lineTo(a + 8, b); ctx.stroke();
    }, "optimiser path");
    item((a, b) => { ctx.strokeStyle = C.green; ctx.lineWidth = 2.5; cross(a, b, 5); },
      "grid minimum");
    text("drag markers to move them",
      W - 8, y, "right", C.ink2, 12);
  }

  function drawLoss() {
    const full = model.get("full_loss"), batch = model.get("batch_loss");
    const best = model.get("loss_min"), n = nSteps();
    const { l, t, w, h } = LOSS;
    // legend on the title row, right-aligned
    let lx = l + w;
    const entries = [["grid minimum", C.green, true], ["all stations", C.orange, false]];
    if (batch.length) entries.splice(1, 0, ["mini-batch", C.grey, false]);
    ctx.font = `12px ${FONT}`;
    for (const [name, colour, dashed] of entries.reverse()) {
      const tw = ctx.measureText(name).width;
      text(name, lx, t - 20, "right", C.ink, 12); lx -= tw + 6;
      ctx.strokeStyle = colour; ctx.lineWidth = 1.8; ctx.setLineDash(dashed ? [4, 3] : []);
      ctx.beginPath(); ctx.moveTo(lx - 18, t - 24); ctx.lineTo(lx, t - 24); ctx.stroke();
      ctx.setLineDash([]); lx -= 32;
    }
    const ok = (v) => v > 0 && isFinite(v);
    // the lower limit follows the all-station loss and the grid minimum, so noisy
    // mini-batch dips (clipped at the frame) do not stretch the axis
    const ref = [...full, best].filter(ok);
    const all = [...ref, ...batch.filter(ok)];
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1; ctx.strokeRect(l, t, w, h);
    text("step", l + w / 2, t + h + 34, "center", C.ink, 12);
    if (!ref.length || n < 1) return;
    const lo = Math.log10(Math.max(Math.min(...ref), 1e-8)) - 0.15;
    const hi = Math.log10(Math.max(...all)) + 0.15;
    const X = (k) => l + k / n * w;
    const Y = (v) => t + h - (Math.log10(Math.max(v, 1e-8)) - lo) / (hi - lo) * h;
    // y ticks at 1, 2, 5 times powers of ten (decades only if that is too many)
    const ticks = [];
    for (let e = Math.floor(lo); e <= Math.ceil(hi); e++)
      for (const m of [1, 2, 5]) {
        const v = m * 10 ** e, lv = Math.log10(v);
        if (lv >= lo && lv <= hi) ticks.push([v, m]);
      }
    const shown = ticks.length > 8 ? ticks.filter(([, m]) => m === 1) : ticks;
    for (const [v] of shown) {
      const y = Y(v);
      ctx.strokeStyle = C.grid; ctx.lineWidth = 0.8;
      ctx.beginPath(); ctx.moveTo(l, y); ctx.lineTo(l + w, y); ctx.stroke();
      text(fmtLoss(v), l - 6, y + 4, "right");
    }
    const dx = [1, 2, 5, 10, 20, 25, 50, 100, 200].find((d) => n / d <= 6) || 500;
    for (let k = 0; k <= n; k += dx) {
      text(String(k), X(k), t + h + 15, "center");
      ctx.strokeStyle = C.ink2; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(X(k), t + h); ctx.lineTo(X(k), t + h - 4); ctx.stroke();
    }
    ctx.save(); ctx.beginPath(); ctx.rect(l, t, w, h); ctx.clip();
    ctx.globalAlpha = drag || pending ? 0.3 : 1;
    const k = Math.min(step, full.length - 1);
    const curve = (values, colour, width) => {
      ctx.strokeStyle = colour; ctx.lineWidth = width; ctx.beginPath();
      values.forEach((v, j) => { j ? ctx.lineTo(X(j), Y(v)) : ctx.moveTo(X(j), Y(v)); });
      ctx.stroke();
    };
    if (best > 0) {
      ctx.setLineDash([5, 4]); ctx.strokeStyle = C.green; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(l, Y(best)); ctx.lineTo(l + w, Y(best)); ctx.stroke();
      ctx.setLineDash([]);
    }
    // the curve grows with the replay
    if (batch.length) curve(batch.slice(0, Math.max(k, 1)), "rgba(92,99,112,0.75)", 0.9);
    curve(full.slice(0, k + 1), C.orange, 1.8);
    if (n <= 20) {  // few steps: mark every iterate
      ctx.fillStyle = C.orange;
      full.slice(0, k + 1).forEach((v, j) => {
        ctx.beginPath(); ctx.arc(X(j), Y(v), 2.6, 0, 7); ctx.fill();
      });
    }
    ctx.restore();
    if (!drag && full.length) {  // where the map markers currently are
      ctx.beginPath(); ctx.arc(X(k), Y(full[k]), 4.5, 0, 7); ctx.fillStyle = C.orange;
      ctx.fill(); ctx.lineWidth = 1.2; ctx.strokeStyle = "#fff"; ctx.stroke();
    }
  }

  function drawStats() {  // the results at the replayed step
    const path = model.get("path"), full = model.get("full_loss");
    const k = Math.min(step, path.length - 1);
    if (k < 0 || pending || drag) {
      for (const s of [errTrue, errGrid, misfitStat]) s.set("—");
      return;
    }
    const p = path[k], src = state().source, g = model.get("grid_min");
    const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
    const km = (v) => (isFinite(v) ? `${v.toFixed(1)}` : "—");
    errTrue.set(km(dist(p, src)), "km");
    errGrid.set(g.length ? km(dist(p, g)) : "—", "km");
    misfitStat.set(fmtStat(full[k]), "km²");
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const layers = model.get("layers"), s = state();
    for (const map of MAPS) drawMap(map, s, layers);
    drawKey();
    drawLoss();
    drawStats();
    stepStat.set(String(step), `/ ${nSteps()}`);
    scrub.set(nSteps() ? pos / nSteps() : 0);
  }

  // ---- interaction ----------------------------------------------------------
  function pick(px, py) {
    const s = state();
    let best = null, bd = Infinity;
    for (const map of MAPS) {
      const d = (p) => { const [x, y] = toPx(p, map); return Math.hypot(x - px, y - py); };
      const consider = (hit, dist, limit) => {
        if (dist < limit && dist < bd) { bd = dist; best = { ...hit, map }; }
      };
      consider({ kind: "source" }, d(s.source), HIT + 2);
      consider({ kind: "start" }, d(s.start), HIT);
      s.stations.forEach((p, i) => consider({ kind: "station", index: i }, d(p), HIT));
    }
    return best;
  }
  const mapAt = (px, py) => MAPS.find((map) => inMap(px, py, map)) || null;

  function scrubTo(px) { pause(); setPos((px - LOSS.l) / LOSS.w * nSteps()); }

  canvas.addEventListener("pointerdown", (ev) => {
    const [px, py] = cv.xy(ev), hit = pick(px, py);
    if (!hit) {
      if (inLoss(px, py)) {
        scrubbing = true; canvas.setPointerCapture(ev.pointerId); scrubTo(px);
      }
      return;
    }
    playing = false;
    drag = hit;
    live = {
      stations: model.get("stations").map((p) => [...p]),
      source: [...model.get("source")], start: [...model.get("start")],
    };
    canvas.setPointerCapture(ev.pointerId);
    canvas.style.cursor = "grabbing";
    draw();
  });
  canvas.addEventListener("pointermove", (ev) => {
    const [px, py] = cv.xy(ev);
    if (scrubbing) return scrubTo(px);
    if (!drag) {
      const hit = pick(px, py);
      const same = (hit && hover && hit.kind === hover.kind && hit.index === hover.index);
      if (!same && (hit || hover)) { hover = hit; draw(); }
      canvas.style.cursor = hit ? "grab" : inLoss(px, py) ? "ew-resize" : "default";
      return;
    }
    const [x, y] = kmOf(px, py, drag.map), p = [clamp(x), clamp(y)];
    if (drag.kind === "station") live.stations[drag.index] = p;
    else live[drag.kind] = p;
    draw();
  });
  const release = () => {
    if (scrubbing) { scrubbing = false; return; }
    if (!drag) return;
    const kind = drag.kind;
    if (kind === "station") model.set("stations", live.stations);
    else model.set(kind, live[kind]);
    model.save_changes();
    drag = null; live = null; canvas.style.cursor = "default";
    waitForRun();
  };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
  canvas.addEventListener("pointerleave", () => { if (hover && !drag) { hover = null; draw(); } });
  for (const name of ["stations", "source", "start", "rays", "grid_min", "full_loss",
    "batch_loss", "loss_min"])
    model.on(`change:${name}`, draw);
  model.on("change:run_id", newRun);
  model.on("change:layers", loadImages);
  syncFields();
  loadImages();
  if (!pending) newRun(); else { status.textContent = "computing…"; syncPlay(); }
}
export default { render };
