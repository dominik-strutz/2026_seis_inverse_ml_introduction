// Point summaries of a posterior (see summarieswidget.py); controls.js is put in front
// of this module (tokens C, FONT, SYM, makeStage, stageCanvas, elt).
// Left: the zoomed map with the density, its 68 / 95 % regions and the summaries,
// the marginal of m_x above it and of m_y beside it. Each summary sends a thin line
// from its marker to a dot on the two marginal curves. Right: one row per summary
// (name, what it minimises, position, error). Hover a row or a marker to isolate
// that summary, click a row to hide it.

const W = 1100, H = 500;
const ML = 76, MT = 112, MS = 330;       // map: left, top, side (logical px)
const TOP_H = 76, GAP = 12;              // marginal above: height, gap to the map
const RIGHT_W = 82;                      // marginal beside: width
const XL = 590;                          // left edge of the list
const ROW0 = MT + 32, PITCH = 50;        // first row, row spacing
const COL_POS = XL + 392, COL_ERR = XL + 492;
const MONO = "'DejaVu Sans Mono', Menlo, Consolas, monospace";
const MARK = 5.8, STAR = 3.8;               // marker sizes (px)
const DIM = 0.16;                        // opacity of what is not isolated

const DARK_GREEN = "#007a58";
const FILL_GREEN = "#009E73";

// text made of pieces: [{t, i: italic, b: bold, sub}], drawn from x (or ending at x)
function mathRun(ctx, segs, x, y, size, align = "left", colour = C.ink) {
  const parts = [];
  let total = 0;
  for (const s of segs) {
    const font = `${s.b ? "bold " : ""}${s.i ? "italic " : ""}${size}px ${s.i ? SYM : FONT}`;
    ctx.font = font;
    const w = ctx.measureText(s.t).width;
    let ws = 0;
    if (s.sub) { ctx.font = `italic ${size * 0.68}px ${SYM}`; ws = ctx.measureText(s.sub).width + 1; }
    parts.push({ s, font, w, ws });
    total += w + ws;
  }
  let cx = align === "right" ? x - total : align === "center" ? x - total / 2 : x;
  ctx.fillStyle = colour; ctx.textAlign = "left";
  for (const { s, font, w, ws } of parts) {
    ctx.font = font; ctx.fillText(s.t, cx, y);
    if (s.sub) { ctx.font = `italic ${size * 0.68}px ${SYM}`; ctx.fillText(s.sub, cx + w + 1, y + size * 0.3); }
    cx += w + ws;
  }
}
const M = (sub) => ({ t: "m", i: true, sub });
const P_OF = (sub) => [{ t: "p", i: true }, { t: "(" }, M(sub), { t: " | " }, { t: "d̃", i: true, b: true }, { t: ")" }];

function niceStep(max) {  // 1, 2 or 5 times a power of ten, giving 2 to 4 ticks up to max
  const e = Math.pow(10, Math.floor(Math.log10(max / 2)));
  for (const f of [1, 2, 5, 10]) if (max / (f * e) <= 4) return f * e;
  return 10 * e;
}

function render({ model, el }) {
  const st = makeStage(el, W);
  const cv = stageCanvas(st, W, H);
  const { canvas, ctx } = cv;
  st.stage.append(canvas);
  st.onScale(() => { cv.resize(); draw(); });

  let hover = null;                 // key of the isolated item
  const hidden = new Set();
  const image = new Image();
  image.onload = () => draw();
  const loadImage = () => { image.src = model.get("image"); };
  loadImage();

  // ---- geometry -------------------------------------------------------------
  const lo = () => model.get("lo"), hi = () => model.get("hi");
  const MW = () => MS * (hi()[0] - lo()[0]) / (hi()[1] - lo()[1]);
  const px = (x) => ML + (x - lo()[0]) / (hi()[0] - lo()[0]) * MW();
  const py = (y) => MT + MS - (y - lo()[1]) / (hi()[1] - lo()[1]) * MS;
  const nodes = () => {
    const n = model.get("marginal_x").length, box = model.get("box");
    return Array.from({ length: n }, (_, k) => k / (n - 1) * box);
  };
  const baseTop = MT - GAP;                      // baseline of the marginal above
  const baseRight = () => ML + MW() + 26;       // baseline of the marginal beside
  // marginals: the largest value inside the view, with room above
  function peak(values, a, b) {
    const nd = nodes();
    let m = 0;
    values.forEach((v, k) => { if (nd[k] >= a - 1 && nd[k] <= b + 1) m = Math.max(m, v); });
    return m * 1.12;
  }
  const marginalAt = (values, v) => {            // linear interpolation on the nodes
    const nd = nodes(), box = model.get("box");
    const f = Math.min(Math.max(v / box, 0), 1) * (nd.length - 1);
    const k = Math.min(Math.floor(f), nd.length - 2);
    return values[k] + (f - k) * (values[k + 1] - values[k]);
  };

  const items = () => {  // the list rows, in order
    const rows = model.get("marks").map((m) => ({ ...m, kind: "mark" }));
    rows.push({ key: "posterior", name: "posterior",
      sub: "68 % solid, 95 % dashed region", kind: "posterior" });
    const n = model.get("stations").length;
    rows.push({ key: "stations", name: `${n} stations`,
      sub: "where the arrival times are picked", kind: "stations" });
    return rows;
  };
  const rowY = (i) => ROW0 + i * PITCH;
  const alpha = (key, floor = DIM) => (hover === null || hover === key ? 1 : floor);

  // ---- marks ----------------------------------------------------------------
  function path(shape, x, y, r) {
    ctx.beginPath();
    if (shape === "diamond") {
      ctx.moveTo(x, y - r * 1.25); ctx.lineTo(x + r * 1.25, y);
      ctx.lineTo(x, y + r * 1.25); ctx.lineTo(x - r * 1.25, y);
    } else if (shape === "square") {
      ctx.rect(x - r, y - r, 2 * r, 2 * r);
    } else if (shape === "hexagon") {
      for (let k = 0; k < 6; k++) {
        const a = Math.PI / 6 + k * Math.PI / 3;
        ctx[k ? "lineTo" : "moveTo"](x + 1.15 * r * Math.cos(a), y + 1.15 * r * Math.sin(a));
      }
    } else if (shape === "star") {
      for (let k = 0; k < 10; k++) {
        const a = -Math.PI / 2 + k * Math.PI / 5, rr = k % 2 ? r * 0.88 : r * 2.1;
        ctx[k ? "lineTo" : "moveTo"](x + rr * Math.cos(a), y + rr * Math.sin(a));
      }
    } else if (shape === "triangle") {
      ctx.moveTo(x, y - r * 1.2); ctx.lineTo(x + r, y + r * 0.8); ctx.lineTo(x - r, y + r * 0.8);
    } else {
      ctx.arc(x, y, r, 0, 2 * Math.PI);
    }
    ctx.closePath();
  }
  function marker(shape, x, y, colour, r = MARK) {
    ctx.lineJoin = "round";
    if (shape === "ring") {   // open, so a marker inside it stays visible
      path("circle", x, y, r + 3);
      ctx.lineWidth = 5.5; ctx.strokeStyle = "#fff"; ctx.stroke();
      ctx.lineWidth = 2.4; ctx.strokeStyle = colour; ctx.stroke();
      return;
    }
    path(shape, x, y, r);
    ctx.lineWidth = 2.8; ctx.strokeStyle = "#fff"; ctx.stroke();
    ctx.fillStyle = colour; ctx.fill();
  }
  // a dot on a marginal curve
  function dot(x, y, colour, r = 3.4) {
    ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.lineWidth = 2.2; ctx.strokeStyle = "#fff"; ctx.stroke();
    ctx.fillStyle = colour; ctx.fill();
  }
  function tag(s, x, y, colour) {  // a name on a white chip beside a marker
    ctx.font = `13px ${FONT}`;
    const w = ctx.measureText(s).width + 14;
    const left = x + 18 + w > ML + MW() ? x - 18 - w : x + 18;
    ctx.fillStyle = "rgba(255,255,255,0.94)"; ctx.strokeStyle = colour; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.roundRect(left, y - 12, w, 24, 4); ctx.fill(); ctx.stroke();
    ctx.fillStyle = C.ink; ctx.textAlign = "left"; ctx.fillText(s, left + 7, y + 4.5);
  }

  // ---- drawing --------------------------------------------------------------
  function ticks(a, b, step) {
    const out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(v);
    return out;
  }
  function line(x0, y0, x1, y1) { ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); }
  function polyline(pts) {
    ctx.beginPath();
    pts.forEach(([x, y], k) => (k ? ctx.lineTo(px(x), py(y)) : ctx.moveTo(px(x), py(y))));
    ctx.stroke();
  }

  function drawMap() {
    const mw = MW(), box = model.get("box");
    ctx.save(); ctx.beginPath(); ctx.rect(ML, MT, mw, MS); ctx.clip();
    if (image.complete && image.naturalWidth && !hidden.has("posterior")) {  // density, cropped to the view
      const iw = image.naturalWidth, ih = image.naturalHeight;
      const sx = lo()[0] / box * iw, sw = (hi()[0] - lo()[0]) / box * iw;
      const sy = (box - hi()[1]) / box * ih, sh = (hi()[1] - lo()[1]) / box * ih;
      ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
      ctx.drawImage(image, sx, sy, sw, sh, ML, MT, mw, MS);
    }
    const c = model.get("contours");
    ctx.globalAlpha = alpha("posterior", 0.35);
    ctx.strokeStyle = DARK_GREEN; ctx.lineJoin = "round";
    ctx.lineWidth = 2.2; ctx.setLineDash([]);
    if (!hidden.has("posterior")) (c["68"] || []).forEach(polyline);
    ctx.lineWidth = 1.7; ctx.setLineDash([7, 5]);
    if (!hidden.has("posterior")) (c["95"] || []).forEach(polyline);
    ctx.setLineDash([]); ctx.globalAlpha = 1;
    // stations
    if (!hidden.has("stations")) {
      ctx.globalAlpha = alpha("stations", 0.3);
      model.get("stations").forEach(([x, y], k) => {
        marker("triangle", px(x), py(y), C.grey, 5.5);
        ctx.font = `italic 14px ${SYM}`; ctx.fillStyle = C.ink2; ctx.textAlign = "center";
        ctx.fillText(`ξ${sub(k + 1)}`, px(x), py(y) + 24);
      });
      ctx.globalAlpha = 1;
    }
    ctx.restore();
  }

  function drawFrame() {
    const mw = MW();
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1; ctx.setLineDash([]);
    ctx.strokeRect(ML + 0.5, MT + 0.5, mw, MS);
    const step = 10, tx = ticks(lo()[0], hi()[0], step), ty = ticks(lo()[1], hi()[1], step);
    ctx.font = `12.5px ${FONT}`; ctx.fillStyle = C.ink; ctx.textAlign = "center";
    for (const v of tx) {
      line(px(v), MT + MS, px(v), MT + MS - 5); line(px(v), MT, px(v), MT + 5);
      ctx.fillText(String(Math.round(v)), px(v), MT + MS + 17);
    }
    ctx.textAlign = "right";
    for (const v of ty) {
      line(ML, py(v), ML + 5, py(v)); line(ML + mw, py(v), ML + mw - 5, py(v));
      ctx.fillText(String(Math.round(v)), ML - 8, py(v) + 4.5);
    }
    mathRun(ctx, [M("x"), { t: " (km)" }], ML + mw / 2, MT + MS + 40, 14.5, "center");
    ctx.save(); ctx.translate(ML - 46, MT + MS / 2); ctx.rotate(-Math.PI / 2);
    mathRun(ctx, [M("y"), { t: " (km)" }], 0, 0, 14.5, "center");
    ctx.restore();
  }

  // the marginal above (curve of m_x) and beside (curve of m_y), with their own axes
  function drawMarginals() {
    const mw = MW(), mx = model.get("marginal_x"), my = model.get("marginal_y");
    const nd = nodes(), bR = baseRight();
    const pmx = peak(mx, lo()[0], hi()[0]), pmy = peak(my, lo()[1], hi()[1]);
    const yTop = (v) => baseTop - v / pmx * TOP_H;
    const xRight = (v) => bR + v / pmy * RIGHT_W;
    ctx.globalAlpha = hidden.has("posterior") ? 0 : alpha("posterior", 0.35);
    // above
    ctx.save(); ctx.beginPath(); ctx.rect(ML, baseTop - TOP_H - 4, mw, TOP_H + 4); ctx.clip();
    ctx.beginPath(); ctx.moveTo(px(nd[0]), baseTop);
    nd.forEach((v, k) => ctx.lineTo(px(v), yTop(mx[k])));
    ctx.lineTo(px(nd[nd.length - 1]), baseTop); ctx.closePath();
    ctx.fillStyle = "rgba(0,158,115,0.28)"; ctx.fill();
    ctx.beginPath(); nd.forEach((v, k) => ctx[k ? "lineTo" : "moveTo"](px(v), yTop(mx[k])));
    ctx.lineWidth = 2.2; ctx.strokeStyle = DARK_GREEN; ctx.lineJoin = "round"; ctx.stroke();
    ctx.restore();
    // beside
    ctx.save(); ctx.beginPath(); ctx.rect(bR, MT, RIGHT_W + 4, MS); ctx.clip();
    ctx.beginPath(); ctx.moveTo(bR, py(nd[0]));
    nd.forEach((v, k) => ctx.lineTo(xRight(my[k]), py(v)));
    ctx.lineTo(bR, py(nd[nd.length - 1])); ctx.closePath();
    ctx.fillStyle = "rgba(0,158,115,0.28)"; ctx.fill();
    ctx.beginPath(); nd.forEach((v, k) => ctx[k ? "lineTo" : "moveTo"](xRight(my[k]), py(v)));
    ctx.lineWidth = 2.2; ctx.strokeStyle = DARK_GREEN; ctx.stroke();
    ctx.restore();
    ctx.globalAlpha = 1;

    // axes: density scale (1/km) on the left of the top plot and under the right plot
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1; ctx.fillStyle = C.ink; ctx.font = `11.5px ${FONT}`;
    line(ML, baseTop, ML + mw, baseTop);            // x axis of the plot above
    line(ML, baseTop, ML, baseTop - TOP_H);         // its density axis
    for (const v of ticks(lo()[0], hi()[0], 10)) line(px(v), baseTop, px(v), baseTop - 4);
    const sx = niceStep(pmx), dec = sx < 0.1 ? 2 : 1;
    ctx.textAlign = "right";
    for (let v = 0; v <= pmx + 1e-9; v += sx) {
      line(ML, yTop(v), ML + 4, yTop(v));
      ctx.fillText(v.toFixed(v ? dec : 0), ML - 7, yTop(v) + 4);
    }
    line(bR, MT, bR, MT + MS);                      // m_y axis of the plot beside
    line(bR, MT + MS, bR + RIGHT_W, MT + MS);       // its density axis
    for (const v of ticks(lo()[1], hi()[1], 10)) line(bR, py(v), bR + 4, py(v));
    const sy = niceStep(pmy), decY = sy < 0.1 ? 2 : 1;
    ctx.textAlign = "center";
    for (let v = 0; v <= pmy + 1e-9; v += sy) {
      line(xRight(v), MT + MS, xRight(v), MT + MS - 4);
      ctx.fillText(v.toFixed(v ? decY : 0), xRight(v), MT + MS + 16);
    }
    ctx.globalAlpha = 1;
    mathRun(ctx, P_OF("x"), ML + 14, baseTop - TOP_H + 4, 14.5, "left");
    mathRun(ctx, P_OF("y"), bR + 6, MT - 22, 14.5, "left");
    mathRun(ctx, [{ t: "1/km" }], bR + RIGHT_W, MT + MS + 36, 11.5, "right", C.ink2);
    return { yTop, xRight };
  }

  function drawMarks(scales) {
    const marks = model.get("marks").filter((m) => !hidden.has(m.key));
    // the isolated one last, so it lies on top
    const order = [...marks].sort((a, b) => (a.key === hover) - (b.key === hover));
    const mw = MW(), mx = model.get("marginal_x"), my = model.get("marginal_y");
    // lines first, markers over them
    for (const m of order) {
      const on = hover === m.key, a = alpha(m.key);
      ctx.globalAlpha = a; ctx.strokeStyle = m.colour; ctx.lineWidth = on ? 2.4 : 1.4;
      const dotY = scales.yTop(marginalAt(mx, m.x)), dotX = scales.xRight(marginalAt(my, m.y));
      line(px(m.x), py(m.y), px(m.x), dotY);
      line(px(m.x), py(m.y), dotX, py(m.y));
      dot(px(m.x), dotY, m.colour); dot(dotX, py(m.y), m.colour);
    }
    for (const m of order) {
      ctx.globalAlpha = alpha(m.key);
      marker(m.shape, px(m.x), py(m.y), m.colour, m.shape === "star" ? STAR : MARK);
    }
    ctx.globalAlpha = 1;
    const h = marks.find((m) => m.key === hover);
    if (h) tag(`${h.name}  (${h.x.toFixed(1)}, ${h.y.toFixed(1)})`, px(h.x), py(h.y), h.colour);
  }

  function drawList() {
    const rows = items();
    ctx.textAlign = "left"; ctx.fillStyle = C.ink2; ctx.font = `11px ${FONT}`;
    ctx.fillText("SUMMARY", XL + 40, MT + 4);
    ctx.textAlign = "right";
    ctx.fillText("POSITION (km)", COL_POS, MT + 4);
    ctx.fillText("ERROR (km)", COL_ERR, MT + 4);
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1;
    line(XL, MT + 14, COL_ERR + 6, MT + 14);
    rows.forEach((r, i) => {
      const y = rowY(i), off = hidden.has(r.key);
      ctx.globalAlpha = (off ? 0.35 : 1) * alpha(r.key, 0.45);
      const gx = XL + 16;
      if (r.kind === "mark") marker(r.shape, gx, y, r.colour, r.shape === "star" ? STAR : MARK);
      else if (r.kind === "posterior") {
        ctx.fillStyle = "rgba(0,158,115,0.28)"; ctx.fillRect(gx - 14, y - 11, 28, 22);
        ctx.strokeStyle = DARK_GREEN; ctx.lineWidth = 2.2; line(gx - 14, y - 4, gx + 14, y - 4);
        ctx.lineWidth = 1.7; ctx.setLineDash([5, 3]); line(gx - 14, y + 5, gx + 14, y + 5);
        ctx.setLineDash([]);
      } else marker("triangle", gx, y, C.grey, 5.5);
      ctx.textAlign = "left"; ctx.fillStyle = C.ink;
      ctx.font = `${hover === r.key ? "bold " : ""}15px ${FONT}`;
      ctx.fillText(r.name, XL + 40, y - 2);
      ctx.fillStyle = C.ink2; ctx.font = `12px ${FONT}`;
      ctx.fillText(r.sub, XL + 40, y + 15);
      if (r.kind === "mark") {
        ctx.fillStyle = C.ink; ctx.font = `14px ${MONO}`; ctx.textAlign = "right";
        ctx.fillText(`${r.x.toFixed(1)}, ${r.y.toFixed(1)}`, COL_POS, y + 5);
        if (r.error !== null) {
          ctx.font = `bold 14px ${MONO}`; ctx.fillText(r.error.toFixed(1), COL_ERR, y + 5);
        }
      }
      if (off) { ctx.strokeStyle = C.ink; ctx.lineWidth = 1; line(XL + 40, y - 6, XL + 40 + ctx.measureText(r.name).width, y - 6); }
      ctx.globalAlpha = 1;
    });
    ctx.fillStyle = C.ink2; ctx.font = `12px ${FONT}`; ctx.textAlign = "left";
    ctx.fillText("Hover a row or a marker to isolate it, click a row to hide it.",
      XL, rowY(rows.length - 1) + 40);
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    ctx.setLineDash([]); ctx.lineCap = "round";
    drawMap();
    const scales = drawMarginals();
    drawFrame();
    // marks reach into the marginals, so they are drawn after everything else
    drawMarks(scales);
    drawList();
  }

  // ---- pointer --------------------------------------------------------------
  function rowAt(x, y) {
    if (x < XL - 10) return null;
    const rows = items();
    for (let i = 0; i < rows.length; i++) if (Math.abs(y - rowY(i)) < PITCH / 2) return rows[i].key;
    return null;
  }
  function markAt(x, y) {
    let best = null, bd = 18;
    for (const m of model.get("marks")) {
      if (hidden.has(m.key)) continue;
      const d = Math.hypot(px(m.x) - x, py(m.y) - y);
      if (d < bd) { bd = d; best = m.key; }
    }
    return best;
  }
  const keyAt = (x, y) => rowAt(x, y) ?? markAt(x, y);
  canvas.addEventListener("pointermove", (ev) => {
    const [x, y] = cv.xy(ev), k = keyAt(x, y);
    canvas.style.cursor = k ? "pointer" : "default";
    if (k !== hover) { hover = k; draw(); }
  });
  canvas.addEventListener("pointerleave", () => { if (hover !== null) { hover = null; draw(); } });
  canvas.addEventListener("click", (ev) => {
    const [x, y] = cv.xy(ev), k = rowAt(x, y);
    if (!k) return;
    if (hidden.has(k)) hidden.delete(k); else hidden.add(k);
    draw();
  });
  model.on("change:image", () => { loadImage(); });
  for (const name of ["lo", "hi", "marks", "contours", "marginal_x", "marginal_y", "stations"])
    model.on(`change:${name}`, () => draw());
  draw();
}
export default { render };
