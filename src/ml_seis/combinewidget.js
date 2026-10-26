// A density can be combined (see combinewidget.py); controls.js is put in front of
// this module (tokens C, FONT, SYM, makeStage, stageCanvas, elt, ui*).
// Colours follow the course convention: prior blue, likelihood green, posterior red.
// Two maps on the same view. Left: the likelihood (the network's answer, green) and a
// new prior, a band across the map (blue). Right: their product, the posterior (red),
// with the prior as its line only. One line per distribution, its 1σ level: the
// prior's ±w, the 68 % region of the likelihood and of the posterior. The prior's
// line is drawn in both maps: drag it, in either map, to change its width; drag one
// of its two anchors (left map) to move it. One number reads off what changed. The
// band is a Gaussian across its centre line with standard deviation w.

const W = 1100, H = 410;
const S = 300, T = 58;                       // map side and top (logical px)
const X0 = [60, 376];                        // left edges of the two maps
const XT = 716;                              // left edge of the readout
const R = 150;                               // samples per side of a shaded map
const WIDTHS = [1, 12];                      // range of the prior's width (km)
const GREEN = [0, 158, 115], DARK_GREEN = "#007a58";     // likelihood
const RED = [213, 94, 0], VERM = "#D55E00";              // posterior
const BAND = [86, 180, 233], BAND_LINE = C.blue;         // prior
const MONO = "'DejaVu Sans Mono', Menlo, Consolas, monospace";

function decode(b64) {                       // little-endian float32 -> Float32Array
  const bin = atob(b64), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function render({ model, el }) {
  const st = makeStage(el, W);
  const wrap = elt("div");
  wrap.style.cssText = `position:relative;width:${W}px;height:${H}px;`;
  const cv = stageCanvas(st, W, H);
  const { canvas, ctx } = cv;
  wrap.append(canvas);
  st.stage.append(wrap);
  st.onScale(() => { cv.resize(); draw(); });

  const shade = elt("canvas");
  shade.width = shade.height = R;
  const sctx = shade.getContext("2d");
  const pixels = sctx.createImageData(R, R);

  const n = () => model.get("n"), box = () => model.get("box");
  const lo = () => model.get("lo"), hi = () => model.get("hi");
  const span = () => hi()[0] - lo()[0];
  const cell = () => (box() / (n() - 1)) ** 2;
  const mapX = (p) => (x) => X0[p] + (x - lo()[0]) / span() * S;
  const mapY = (y) => T + S - (y - lo()[1]) / span() * S;

  // ---- the likelihood grid and what is known about it before the prior ----------
  let pLike, before;
  function load() {
    pLike = decode(model.get("p_likelihood"));
    before = summarize(pLike);
    before.lines = contours(pLike, before.l68);
  }

  // mean, side, 95 % region of a grid of probabilities that sums to one
  function summarize(p) {
    const N = n(), b = box(), ly = model.get("line_y"), truth = model.get("m_true");
    let mx = 0, my = 0, north = 0;
    for (let j = 0; j < N; j++) {
      const y = j / (N - 1) * b;
      let row = 0;
      for (let i = 0; i < N; i++) { const v = p[j * N + i]; row += v; mx += v * i / (N - 1) * b; }
      my += row * y;
      if (ly !== null) north += row * (Math.abs(y - ly) < 1e-9 ? 0.5 : y > ly ? 1 : 0);
    }
    const sorted = Float32Array.from(p).sort();
    let acc = 0, l68 = 0, l95 = 0, count = 0;
    for (let k = sorted.length - 1; k >= 0; k--) {
      acc += sorted[k];
      if (!l68 && acc >= 0.68) l68 = sorted[k];
      if (acc >= 0.95) { l95 = sorted[k]; count = sorted.length - k; break; }
    }
    return { p, north: ly === null ? null : north, area: count * cell(), l68, l95,
      mean: [mx, my], error: Math.hypot(mx - truth[0], my - truth[1]) };
  }

  // p times the band's weight, normalised; null when the band misses all the mass
  function combine(p, g) {
    const out = new Float64Array(p.length);
    let total = 0;
    for (let k = 0; k < p.length; k++) { out[k] = p[k] * g[k]; total += out[k]; }
    if (!(total > 1e-300)) return null;
    for (let k = 0; k < out.length; k++) out[k] /= total;
    return out;
  }

  function bandWeights() {
    const [x0, y0, x1, y1] = model.get("band"), w = model.get("width");
    const N = n(), b = box(), len = Math.hypot(x1 - x0, y1 - y0);
    const nx = -(y1 - y0) / len, ny = (x1 - x0) / len;   // unit normal of the centre line
    const g = new Float64Array(N * N);
    for (let j = 0; j < N; j++) {
      const y = j / (N - 1) * b;
      for (let i = 0; i < N; i++) {
        const d = ((i / (N - 1) * b - x0) * nx + (y - y0) * ny) / w;
        g[j * N + i] = Math.exp(-0.5 * d * d);
      }
    }
    return g;
  }

  // ---- contours of a grid: marching squares, stitched into polylines (km) -------
  function contours(p, level) {
    const N = n(), b = box();
    const val = (i, j) => p[j * N + i];
    const at = (id) => {                       // where a cell edge crosses the level
      const idx = id >> 1, i = idx % N, j = (idx - i) / N, vert = id & 1;
      const a = val(i, j), c = vert ? val(i, j + 1) : val(i + 1, j);
      const t = (level - a) / (c - a);
      return [(i + (vert ? 0 : t)) / (N - 1) * b, (j + (vert ? t : 0)) / (N - 1) * b];
    };
    const segs = [];
    for (let j = 0; j < N - 1; j++) {
      for (let i = 0; i < N - 1; i++) {
        const c0 = val(i, j) >= level, c1 = val(i + 1, j) >= level;
        const c2 = val(i + 1, j + 1) >= level, c3 = val(i, j + 1) >= level;
        const k = c0 + 2 * c1 + 4 * c2 + 8 * c3;
        if (k === 0 || k === 15) continue;
        const e0 = 2 * (j * N + i), e2 = 2 * ((j + 1) * N + i);        // bottom, top
        const e3 = 2 * (j * N + i) + 1, e1 = 2 * (j * N + i + 1) + 1;  // left, right
        const high = (val(i, j) + val(i + 1, j) + val(i + 1, j + 1) + val(i, j + 1)) / 4 >= level;
        const table = {
          1: [[e3, e0]], 2: [[e0, e1]], 3: [[e3, e1]], 4: [[e1, e2]],
          5: high ? [[e0, e1], [e2, e3]] : [[e3, e0], [e1, e2]],
          6: [[e0, e2]], 7: [[e3, e2]], 8: [[e2, e3]], 9: [[e0, e2]],
          10: high ? [[e3, e0], [e1, e2]] : [[e0, e1], [e2, e3]],
          11: [[e1, e2]], 12: [[e1, e3]], 13: [[e0, e1]], 14: [[e3, e0]],
        };
        for (const s of table[k]) segs.push(s);
      }
    }
    const ends = new Map();                    // edge id -> segments that touch it
    segs.forEach((s, k) => { for (const id of s) { if (!ends.has(id)) ends.set(id, []); ends.get(id).push(k); } });
    const used = new Uint8Array(segs.length), lines = [];
    const grow = (chain, id, front) => {       // follow unused segments from an end
      for (;;) {
        const next = (ends.get(id) || []).find((k) => !used[k]);
        if (next === undefined) return;
        used[next] = 1;
        id = segs[next][0] === id ? segs[next][1] : segs[next][0];
        chain[front ? "unshift" : "push"](id);
      }
    };
    segs.forEach((s, k) => {
      if (used[k]) return;
      used[k] = 1;
      const chain = [s[0], s[1]];
      grow(chain, s[1], false); grow(chain, s[0], true);
      lines.push(chain.map(at));
    });
    return lines;
  }

  // ---- drawing helpers -------------------------------------------------------------
  const line = (x0, y0, x1, y1) => { ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); };
  function clipped(p, fn) {
    ctx.save(); ctx.beginPath(); ctx.rect(X0[p], T, S, S); ctx.clip(); fn(); ctx.restore();
  }
  function title(p, words) {                   // lines of coloured pieces above a map
    ctx.font = `13px ${FONT}`; ctx.textAlign = "left";
    words.forEach((w, i) => {
      let x = X0[p];
      for (const piece of w) {
        ctx.fillStyle = piece.colour || C.ink; ctx.fillText(piece.t, x, T - 40 + 17 * i);
        x += ctx.measureText(piece.t).width;
      }
    });
  }
  function frame(p, left) {
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1; ctx.setLineDash([]);
    ctx.strokeRect(X0[p] + 0.5, T + 0.5, S, S);
    const step = span() > 60 ? 20 : span() > 30 ? 10 : 5;
    ctx.font = `12px ${FONT}`; ctx.fillStyle = C.ink;
    for (let v = Math.ceil(lo()[0] / step) * step; v <= hi()[0] + 1e-9; v += step) {
      const x = mapX(p)(v);
      line(x, T + S, x, T + S - 5); line(x, T, x, T + 5);
      ctx.textAlign = "center"; ctx.fillText(String(v), x, T + S + 16);
    }
    for (let v = Math.ceil(lo()[1] / step) * step; v <= hi()[1] + 1e-9; v += step) {
      const y = mapY(v);
      line(X0[p], y, X0[p] + 5, y); line(X0[p] + S, y, X0[p] + S - 5, y);
      if (left) { ctx.textAlign = "right"; ctx.fillText(String(v), X0[p] - 7, y + 4); }
    }
  }
  function axisLabel(sub, x, y) {              // m with a subscript, then (km), centred
    ctx.font = `italic 14px ${SYM}`;
    const wm = ctx.measureText("m").width;
    ctx.font = `italic 10px ${SYM}`;
    const ws = ctx.measureText(sub).width;
    ctx.font = `13px ${FONT}`;
    const wk = ctx.measureText(" (km)").width;
    let cx = x - (wm + ws + wk) / 2;
    ctx.textAlign = "left"; ctx.fillStyle = C.ink;
    ctx.font = `italic 14px ${SYM}`; ctx.fillText("m", cx, y); cx += wm;
    ctx.font = `italic 10px ${SYM}`; ctx.fillText(sub, cx, y + 4); cx += ws;
    ctx.font = `13px ${FONT}`; ctx.fillText(" (km)", cx, y);
  }
  function marker(shape, x, y, colour, r) {
    ctx.beginPath(); ctx.lineJoin = "round";
    if (shape === "star") {
      for (let k = 0; k < 10; k++) {
        const a = -Math.PI / 2 + k * Math.PI / 5, rr = k % 2 ? r * 0.88 : r * 2.1;
        ctx[k ? "lineTo" : "moveTo"](x + rr * Math.cos(a), y + rr * Math.sin(a));
      }
    } else if (shape === "diamond") {
      ctx.moveTo(x, y - r * 1.25); ctx.lineTo(x + r * 1.25, y);
      ctx.lineTo(x, y + r * 1.25); ctx.lineTo(x - r * 1.25, y);
    } else {                                   // triangle
      ctx.moveTo(x, y - r * 1.2); ctx.lineTo(x + r, y + r * 0.8); ctx.lineTo(x - r, y + r * 0.8);
    }
    ctx.closePath();
    ctx.lineWidth = 2.8; ctx.strokeStyle = "#fff"; ctx.stroke();
    ctx.fillStyle = colour; ctx.fill();
  }

  function shadeMap(p, grid, rgb) {            // a grid as shading in one colour, bilinear
    const N = n(), b = box(), sp = span(), max = grid.reduce((m, v) => (v > m ? v : m), 0);
    const data = pixels.data;
    for (let r = 0; r < R; r++) {
      const y = hi()[1] - (r + 0.5) / R * sp;
      const fy = Math.min(Math.max(y / b * (N - 1), 0), N - 1 - 1e-9), j = Math.floor(fy), ty = fy - j;
      for (let c = 0; c < R; c++) {
        const x = lo()[0] + (c + 0.5) / R * sp;
        const fx = Math.min(Math.max(x / b * (N - 1), 0), N - 1 - 1e-9), i = Math.floor(fx), tx = fx - i;
        const v = ((1 - tx) * (1 - ty) * grid[j * N + i] + tx * (1 - ty) * grid[j * N + i + 1]
          + (1 - tx) * ty * grid[(j + 1) * N + i] + tx * ty * grid[(j + 1) * N + i + 1]) / max;
        const o = 4 * (r * R + c);
        data[o] = rgb[0]; data[o + 1] = rgb[1]; data[o + 2] = rgb[2];
        data[o + 3] = Math.round(255 * 0.85 * Math.pow(Math.max(v, 0), 0.6));
      }
    }
    sctx.putImageData(pixels, 0, 0);
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
    ctx.drawImage(shade, X0[p], T, S, S);
  }

  function regions(p, lines, colour) {         // the 68 % region: one line per distribution
    const px = mapX(p);
    ctx.strokeStyle = colour; ctx.lineJoin = "round"; ctx.lineWidth = 2.2; ctx.setLineDash([]);
    for (const pts of lines) {
      ctx.beginPath();
      pts.forEach(([x, y], k) => (k ? ctx.lineTo(px(x), mapY(y)) : ctx.moveTo(px(x), mapY(y))));
      ctx.stroke();
    }
  }

  // the prior's centre line and its px geometry: two points, unit direction, width
  function bandPx(p) {
    const [x0, y0, x1, y1] = model.get("band"), px = mapX(p);
    const a = [px(x0), mapY(y0)], b = [px(x1), mapY(y1)];
    const ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
    return { a, b, ang, c: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2],
      wpx: model.get("width") / span() * S };
  }
  function drawBand(p) {                       // the prior as a soft blue strip and its centre line
    const { a, b, ang, c, wpx } = bandPx(p), reach = 3 * wpx;
    clipped(p, () => {
      ctx.save();
      ctx.translate(c[0], c[1]); ctx.rotate(ang);
      const grad = ctx.createLinearGradient(0, -reach, 0, reach);
      for (let k = 0; k <= 12; k++) {
        const d = 3 * ((k / 12) * 2 - 1);
        grad.addColorStop(k / 12, `rgba(${BAND},${0.42 * Math.exp(-0.5 * d * d)})`);
      }
      ctx.fillStyle = grad; ctx.fillRect(-4000, -reach, 8000, 2 * reach);
      ctx.restore();
      ctx.strokeStyle = BAND_LINE; ctx.lineWidth = 1.4; ctx.setLineDash([]);
      const dx = Math.cos(ang), dy = Math.sin(ang);
      line(a[0] - 4000 * dx, a[1] - 4000 * dy, a[0] + 4000 * dx, a[1] + 4000 * dy);
    });
  }
  function drawAnchors(p) {                    // the two ends of the centre line, to drag
    const { a, b } = bandPx(p);
    for (const q of [a, b]) {
      ctx.beginPath(); ctx.arc(q[0], q[1], 6.5, 0, 2 * Math.PI);
      ctx.fillStyle = "#fff"; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = BAND_LINE; ctx.stroke();
    }
  }
  function priorLines(p) {                     // its 1σ line: ±w across the centre line
    const { c, ang, wpx } = bandPx(p);
    const dx = Math.cos(ang), dy = Math.sin(ang);
    clipped(p, () => {
      ctx.strokeStyle = BAND_LINE; ctx.lineWidth = 1.7; ctx.setLineDash([]);
      for (const side of [-1, 1]) {
        const ox = c[0] - dy * side * wpx, oy = c[1] + dx * side * wpx;
        line(ox - 4000 * dx, oy - 4000 * dy, ox + 4000 * dx, oy + 4000 * dy);
      }
    });
  }

  function drawMarks(p, s) {
    const px = mapX(p), truth = model.get("m_true");
    clipped(p, () => {
      for (const [x, y] of model.get("stations")) marker("triangle", px(x), mapY(y), C.grey, 5);
      marker("diamond", px(s.mean[0]), mapY(s.mean[1]), "#111111", 5.2);
      marker("star", px(truth[0]), mapY(truth[1]), "#111111", 3.8);
    });
  }

  // ---- the readout: one number, before and after ------------------------------------
  function readout(after) {
    const ly = model.get("line_y");
    const [name, sub, get, fmt] = ly !== null
      ? ["NORTH OF THE STATIONS", "chance the source is there", (s) => s.north, (v) => v.toFixed(2)]
      : ["95 % REGION", "area (km²)", (s) => s.area, (v) => String(Math.round(v))];
    ctx.textAlign = "left"; ctx.font = `11px ${FONT}`; ctx.fillStyle = C.ink2;
    ctx.fillText(name, XT, T + 4);
    ctx.font = `26px ${MONO}`; ctx.fillStyle = DARK_GREEN;
    const was = fmt(get(before));
    ctx.fillText(was, XT, T + 40);
    const x = XT + ctx.measureText(was).width;
    ctx.fillStyle = C.ink2; ctx.fillText(" → ", x, T + 40);
    ctx.fillStyle = VERM; ctx.font = `bold 26px ${MONO}`;
    ctx.fillText(after ? fmt(get(after)) : "—", x + ctx.measureText(" → ").width, T + 40);
    ctx.fillStyle = C.ink2; ctx.font = `12px ${FONT}`;
    ctx.fillText(sub, XT, T + 62);
    // what the markers on the maps are
    [["star", "#111111", "true event", 3.2], ["diamond", "#111111", "mean", 4.4],
     ["triangle", C.grey, "station", 4.4]].forEach(([shape, colour, label, r], i) => {
      const y = T + 112 + i * 22;
      marker(shape, XT + 10, y - 4, colour, r);
      ctx.textAlign = "left"; ctx.fillStyle = C.ink2; ctx.font = `12px ${FONT}`;
      ctx.fillText(label, XT + 28, y);
    });
    ctx.textAlign = "left"; ctx.fillStyle = C.ink2; ctx.font = `12px ${FONT}`;
    ctx.fillText("Drag a line for the width, an anchor to move.", XT, T + S + 16);
  }

  // ---- the whole figure -----------------------------------------------------------------
  let queued = false;
  function draw() {
    queued = false;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    ctx.setLineDash([]); ctx.lineCap = "round";
    const g = bandWeights();
    const post = combine(pLike, g);
    const after = post ? summarize(post) : null;
    if (after) after.lines = contours(post, after.l68);

    title(0, [[{ t: "likelihood", colour: DARK_GREEN }, { t: "  from the network" }],
      [{ t: "new prior", colour: BAND_LINE }, { t: "  a fault, ±w across" }]]);
    clipped(0, () => { shadeMap(0, pLike, GREEN); regions(0, before.lines, DARK_GREEN); });
    drawBand(0); priorLines(0); drawAnchors(0); drawMarks(0, before); frame(0, true);
    axisLabel("x", X0[0] + S / 2, T + S + 36);
    ctx.save(); ctx.translate(X0[0] - 40, T + S / 2); ctx.rotate(-Math.PI / 2);
    axisLabel("y", 0, 0); ctx.restore();

    title(1, [[{ t: "posterior", colour: VERM }, { t: "  likelihood × new prior" }],
      [{ t: "new prior", colour: BAND_LINE }, { t: "  line at ±w" }]]);
    if (after) {
      clipped(1, () => { shadeMap(1, post, RED); regions(1, after.lines, VERM); });
      priorLines(1); drawMarks(1, after);
    } else {
      priorLines(1);
      ctx.font = `13px ${FONT}`; ctx.fillStyle = C.ink2; ctx.textAlign = "center";
      ctx.fillText("the prior misses the likelihood", X0[1] + S / 2, T + S / 2);
    }
    frame(1, false);
    axisLabel("x", X0[1] + S / 2, T + S + 36);
    readout(after);

  }
  const redraw = () => { if (!queued) { queued = true; requestAnimationFrame(draw); } };

  // ---- dragging the band -----------------------------------------------------------------
  let grab = null;
  const panelOf = (x, y) => [0, 1].find((p) => x >= X0[p] && x <= X0[p] + S && y >= T && y <= T + S);
  const toKm = (p, x, y) => [lo()[0] + (x - X0[p]) / S * span(), hi()[1] - (y - T) / S * span()];
  function hit(p, x, y) {                      // which part of the prior is under the pointer
    const { a, b, ang, c, wpx } = bandPx(p);
    const across = Math.abs((x - c[0]) * -Math.sin(ang) + (y - c[1]) * Math.cos(ang));
    if (p === 0) {                             // the anchors move it (left map only)
      if (Math.hypot(a[0] - x, a[1] - y) < 14) return { part: "start" };
      if (Math.hypot(b[0] - x, b[1] - y) < 14) return { part: "end" };
    }
    if (Math.abs(across - wpx) < 7) return { part: "width", k: 1 };   // its ±w line
    if (p === 1) return null;                  // its line only
    return across < Math.max(10, 1.2 * wpx) ? { part: "body" } : null;
  }
  const inView = (v, k) => Math.min(Math.max(v, lo()[k] + 1), hi()[k] - 1);
  canvas.addEventListener("pointerdown", (ev) => {
    const [x, y] = cv.xy(ev), p = panelOf(x, y);
    if (p === undefined) return;
    const part = hit(p, x, y);
    if (!part) return;
    canvas.setPointerCapture(ev.pointerId);
    grab = { p, ...part, from: toKm(p, x, y), band: [...model.get("band")] };
  });
  canvas.addEventListener("pointermove", (ev) => {
    const [x, y] = cv.xy(ev);
    if (!grab) {
      const p = panelOf(x, y);
      canvas.style.cursor = p !== undefined && hit(p, x, y) ? "grab" : "default";
      return;
    }
    const [kx, ky] = toKm(grab.p, x, y), [x0, y0, x1, y1] = grab.band;
    let band;
    if (grab.part === "width") {               // the distance from the centre line is k w
      const len = Math.hypot(x1 - x0, y1 - y0);
      const dist = Math.abs(((kx - x0) * -(y1 - y0) + (ky - y0) * (x1 - x0)) / len);
      model.set("width", Math.round(10 * Math.min(Math.max(dist / grab.k, WIDTHS[0]), WIDTHS[1])) / 10);
      redraw();
      return;
    }
    if (grab.part === "body") {
      const dx = Math.min(Math.max(kx - grab.from[0], Math.max(lo()[0] + 1 - x0, lo()[0] + 1 - x1)),
        Math.min(hi()[0] - 1 - x0, hi()[0] - 1 - x1));
      const dy = Math.min(Math.max(ky - grab.from[1], Math.max(lo()[1] + 1 - y0, lo()[1] + 1 - y1)),
        Math.min(hi()[1] - 1 - y0, hi()[1] - 1 - y1));
      band = [x0 + dx, y0 + dy, x1 + dx, y1 + dy];
    } else if (grab.part === "start") {
      band = [inView(kx, 0), inView(ky, 1), x1, y1];
    } else {
      band = [x0, y0, inView(kx, 0), inView(ky, 1)];
    }
    if (Math.hypot(band[2] - band[0], band[3] - band[1]) < 4) return;   // keep a direction
    model.set("band", band);
    redraw();
  });
  const release = () => { grab = null; };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);

  load();
  for (const name of ["p_likelihood", "n", "m_true"]) model.on(`change:${name}`, () => { load(); redraw(); });
  for (const name of ["lo", "hi", "band", "width", "stations", "line_y"]) model.on(`change:${name}`, redraw);
  draw();
}
export default { render };
