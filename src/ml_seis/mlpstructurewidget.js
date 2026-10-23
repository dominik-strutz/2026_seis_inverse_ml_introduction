// A small MLP as a hoverable graph plus weight matrices (see mlpstructurewidget.py).
// controls.js is put in front of this module (tokens C, FONT, SYM, sup, sub, ...).
// Same look and linking as mlpwidget.js: tiles for neurons (hidden ones sketch the
// activation), edges and matrix cells share the weight colours, hovering one marks
// the other.

const W = 760;
const IO = 26, TILE = 24, PITCH = 34;
const X_IN = 110, X_OUT = 650;
const MAXIN = 6, MAXL = 4, MAXW = 8;  // limits of the +/- buttons
const MAT = { gap: 46, maxH: 110, maxCell: 14 };
const ACTS = {
  ReLU: (z) => Math.max(0, z),
  tanh: Math.tanh,
  sigmoid: (z) => 1 / (1 + Math.exp(-z)),
  linear: (z) => z,
};

function render({ model, el }) {
  const st = makeStage(el, W);
  const editable = !!model.get("editable");
  const top = editable ? 84 : 64;  // room for the +/- buttons above the first neurons
  let layers, cs, nCols, span, cy, bracketY, matTop, rowsMax, H;
  const act = ACTS[model.get("activation")] || ACTS.tanh;
  // the activation is drawn unclipped: zero in the middle, the largest value at the edge
  const actMax = Math.max(...Array.from({ length: 25 }, (_, k) => Math.abs(act(-2 + k / 6))));
  const colX = (c) => X_IN + (X_OUT - X_IN) * c / (nCols - 1);
  const nodeY = (c, i) => cy + (i - (cs[c] - 1) / 2) * PITCH;
  const tileOf = (c) => (c === 0 || c === nCols - 1 ? IO : TILE);
  const cellOf = (l) => Math.min(MAT.maxCell, MAT.maxH / cs[l + 1],
    (colX(l + 1) - colX(l) - 16 - MAT.gap + 38) / (cs[l] + 1.6));
  let maxAbs, maxB, geom;
  function layout() {  // everything that depends on the network's shape
    cs = [layers[0].W[0].length, ...layers.map((l) => l.b.length)];
    nCols = cs.length;
    span = (Math.max(...cs) - 1) * PITCH;
    cy = top + span / 2;
    bracketY = top + span + 44; matTop = bracketY + 40;
    rowsMax = Math.max(...cs.slice(1));
    H = Math.round(matTop + Math.min(MAT.maxH, rowsMax * MAT.maxCell) + 48);
    maxAbs = layers.map((l) => Math.max(...l.W.flat().map(Math.abs), 1e-9));
    maxB = layers.map((l) => Math.max(...l.b.map(Math.abs), 1e-9));
    geom = layers.map((l, k) => {  // the matrix block of layer k
      const cell = cellOf(k), nIn = cs[k], nOut = cs[k + 1];
      const bw = nIn * cell + 6 + cell, cx = (colX(k) + colX(k + 1)) / 2;
      const x0 = cx - bw / 2;
      return { cell, nIn, nOut, cx, x0, y0: matTop, bx: x0 + nIn * cell + 6 };
    });
  }
  layers = model.get("layers").map(([Wm, b]) => ({ W: Wm, b }));
  layout();
  const cv = stageCanvas(st, W, H);
  const { canvas, ctx } = cv;
  const bar = elt("div", "mls-bar"), holder = elt("div");
  holder.style.cssText = "position:relative;";
  bar.style.justifyContent = "center";
  st.stage.append(bar, holder);
  holder.append(canvas);
  st.onScale(() => { cv.resize(); draw(); });

  let hover = null, mouse = [0, 0];
  const inLabels = model.get("in_labels") || [];
  const inLabel = (i) =>  // more inputs than labels: carry on the numbering
    inLabels[i] ?? `${(inLabels[0] || "x_1").split("_")[0]}_${i + 1}`;

  function text(s, x, y, align = "left", colour = C.ink, size = 12, font = FONT,
    style = "") {
    ctx.font = `${style ? style + " " : ""}${size}px ${font}`;
    ctx.fillStyle = colour; ctx.textAlign = align;
    ctx.fillText(s, x, y);
  }
  function label(s, x, y, align, size = 17) {  // "m_x": italic base, small subscript
    const [base, subs] = s.split("_");
    ctx.font = `italic ${size}px ${SYM}`;
    const wb = ctx.measureText(base).width;
    ctx.font = `italic ${size * 0.65}px ${SYM}`;
    const ws = subs ? ctx.measureText(subs).width + 1 : 0;
    const x0 = align === "right" ? x - wb - ws : x;
    text(base, x0, y, "left", C.ink, size, SYM, "italic");
    if (subs) text(subs, x0 + wb + 1, y + 4, "left", C.ink, size * 0.65, SYM, "italic");
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
  function tile(x, y, s, hidden, on) {
    const x0 = x - s / 2, y0 = y - s / 2;
    ctx.fillStyle = C.bg; ctx.beginPath(); ctx.roundRect(x0, y0, s, s, 3); ctx.fill();
    if (hidden) {  // the activation, phi(z) for z in [-2, 2]
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.4; ctx.beginPath();
      for (let k = 0; k <= 24; k++) {
        const z = -2 + 4 * k / 24;
        const px = x0 + 3 + (s - 6) * k / 24;
        const py = y - act(z) / actMax * (s / 2 - 4);
        k ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      }
      ctx.stroke();
    }
    ctx.strokeStyle = on ? C.ink : C.ink2; ctx.lineWidth = on ? 2 : 1;
    ctx.beginPath(); ctx.roundRect(x0, y0, s, s, 3); ctx.stroke();
  }
  // is edge (layer c, input i, output j) marked by the hover?
  function edgeOn(c, i, j) {
    const h = hover;
    if (!h) return false;
    if (h.kind === "edge") return h.c === c && h.i === i && h.j === j;
    if (h.kind === "cell") return h.layer === c && h.col === i && h.row === j;
    if (h.kind === "bcell") return h.layer === c && h.row === j;
    return (h.col === c && h.idx === i) || (h.col === c + 1 && h.idx === j);
  }
  const rect = (g, row, col) => [g.x0 + col * g.cell, g.y0 + row * g.cell, g.cell];
  function ring(r, colour = C.ink, lw = 1.8) {
    ctx.strokeStyle = colour; ctx.lineWidth = lw;
    ctx.strokeRect(r[0] - 1, r[1] - 1, r[2] + 2, r[2] + 2);
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const heads = ["INPUT", ...Array(nCols - 2).fill("HIDDEN"), "OUTPUT"];
    heads.forEach((s, c) => {
      text(s, colX(c), 14, "center", C.ink2, 11);
      const sub2 = c === 0 || c === nCols - 1 ? `${cs[c]} value${cs[c] > 1 ? "s" : ""}`
        : `${cs[c]} neuron${cs[c] > 1 ? "s" : ""}, ${model.get("activation")}`;
      text(sub2, colX(c), 32, "center", C.ink2, 11);
    });
    // edges
    ctx.lineCap = "round";
    for (let c = 0; c < nCols - 1; c++)
      for (let i = 0; i < cs[c]; i++) for (let j = 0; j < cs[c + 1]; j++) {
        const w = layers[c].W[j][i], on = edgeOn(c, i, j);
        ctx.globalAlpha = on ? 1 : hover ? 0.12 : 0.85;
        ctx.strokeStyle = w >= 0 ? C.blue : C.orange;
        ctx.lineWidth = (0.6 + 3.6 * Math.abs(w) / maxAbs[c]) * (on ? 1.5 : 1);
        ctx.beginPath();
        ctx.moveTo(colX(c), nodeY(c, i)); ctx.lineTo(colX(c + 1), nodeY(c + 1, j));
        ctx.stroke();
      }
    ctx.globalAlpha = 1;
    // neurons
    for (let c = 0; c < nCols; c++)
      for (let i = 0; i < cs[c]; i++) {
        const on = hover?.kind === "neuron" && hover.col === c && hover.idx === i;
        tile(colX(c), nodeY(c, i), tileOf(c), c > 0 && c < nCols - 1, on);
      }
    for (let i = 0; i < cs[0]; i++)
      label(inLabel(i), colX(0) - IO / 2 - 8, nodeY(0, i) + 5, "right");
    (model.get("out_labels") || []).forEach((s, i) =>
      label(s, colX(nCols - 1) + IO / 2 + 8, nodeY(nCols - 1, i) + 5, "left"));
    // brackets: the band of edges each matrix block holds
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1.2;
    for (let c = 0; c < nCols - 1; c++) {
      const a = colX(c) + 8, b = colX(c + 1) - 8;
      ctx.beginPath(); ctx.moveTo(a, bracketY - 5); ctx.lineTo(a, bracketY);
      ctx.lineTo(b, bracketY); ctx.lineTo(b, bracketY - 5); ctx.stroke();
    }
    // matrices
    layers.forEach((l, k) => {
      const g = geom[k];
      text(`W${sup(`(${k + 1})`)}  b${sup(`(${k + 1})`)}`, g.cx, g.y0 - 10, "center",
        C.ink, 15, SYM, "bold italic");
      text(`${g.nOut}×${g.nIn} + ${g.nOut} = ${g.nOut * g.nIn + g.nOut}`, g.cx,
        g.y0 + g.nOut * g.cell + 17, "center", C.ink2, 11);
      const cell = (x, y, w, max) => {
        ctx.fillStyle = w >= 0 ? C.blue : C.orange;
        ctx.globalAlpha = 0.1 + 0.9 * Math.abs(w) / max;
        ctx.fillRect(x, y, g.cell, g.cell); ctx.globalAlpha = 1;
        ctx.strokeStyle = C.bg; ctx.lineWidth = 0.6; ctx.strokeRect(x, y, g.cell, g.cell);
      };
      for (let j = 0; j < g.nOut; j++) {
        for (let i = 0; i < g.nIn; i++)
          cell(g.x0 + i * g.cell, g.y0 + j * g.cell, l.W[j][i], maxAbs[k]);
        cell(g.bx, g.y0 + j * g.cell, l.b[j], maxB[k]);
      }
    });
    // the hover's matrix marks and value tag
    const h = hover;
    if (!h) return;
    const bRect = (g, row) => [g.bx, g.y0 + row * g.cell, g.cell];
    const ringRow = (k, row) => {
      const g = geom[k];
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.8;
      ctx.strokeRect(g.x0 - 1, g.y0 + row * g.cell - 1, g.nIn * g.cell + 2, g.cell + 2);
      ring(bRect(g, row));
    };
    const wTag = (k, j, i, x, y) =>
      tag(`W${sup(`(${k + 1})`)}${sub(j + 1)}${sub(i + 1)} = ${layers[k].W[j][i].toFixed(2)}`, x, y);
    if (h.kind === "cell") {
      ring(rect(geom[h.layer], h.row, h.col));
      wTag(h.layer, h.row, h.col, mouse[0], mouse[1] - 22);
    } else if (h.kind === "bcell") {
      ring(bRect(geom[h.layer], h.row));
      tag(`b${sup(`(${h.layer + 1})`)}${sub(h.row + 1)} = ${layers[h.layer].b[h.row].toFixed(2)}`,
        mouse[0], mouse[1] - 22);
    } else if (h.kind === "edge") {
      ring(rect(geom[h.c], h.j, h.i));
      wTag(h.c, h.j, h.i, (colX(h.c) + colX(h.c + 1)) / 2,
        (nodeY(h.c, h.i) + nodeY(h.c + 1, h.j)) / 2 - 14);
    } else if (h.kind === "neuron") {
      if (h.col > 0) ringRow(h.col - 1, h.idx);
      if (h.col < nCols - 1) {  // the matrix column that holds this neuron's outgoing edges
        const g = geom[h.col];
        ctx.strokeStyle = C.ink; ctx.lineWidth = 1.8;
        ctx.strokeRect(g.x0 + h.idx * g.cell - 1, g.y0 - 1, g.cell + 2, g.nOut * g.cell + 2);
      }
    }
  }

  const dist = (px, py, x1, y1, x2, y2) => {
    const dx = x2 - x1, dy = y2 - y1;
    const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)));
    return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
  };
  function hit(px, py) {
    for (let c = 0; c < nCols; c++) {
      const half = tileOf(c) / 2 + 3;
      for (let i = 0; i < cs[c]; i++)
        if (Math.abs(px - colX(c)) <= half && Math.abs(py - nodeY(c, i)) <= half)
          return { kind: "neuron", col: c, idx: i };
    }
    for (const [k, g] of geom.entries()) {
      if (py < g.y0 || py > g.y0 + g.nOut * g.cell) continue;
      const row = Math.floor((py - g.y0) / g.cell);
      if (px >= g.bx && px <= g.bx + g.cell) return { kind: "bcell", layer: k, row };
      if (px >= g.x0 && px < g.x0 + g.nIn * g.cell)
        return { kind: "cell", layer: k, row, col: Math.floor((px - g.x0) / g.cell) };
    }
    if (py < bracketY - 8)
      for (let c = 0; c < nCols - 1; c++)
        for (let i = 0; i < cs[c]; i++) for (let j = 0; j < cs[c + 1]; j++)
          if (dist(px, py, colX(c), nodeY(c, i), colX(c + 1), nodeY(c + 1, j)) < 4)
            return { kind: "edge", c, i, j };
    return null;
  }
  canvas.addEventListener("pointermove", (ev) => {
    const [px, py] = cv.xy(ev);
    mouse = [px, py];
    const h = hit(px, py);
    if (JSON.stringify(h) !== JSON.stringify(hover) || h?.kind === "cell"
      || h?.kind === "bcell") { hover = h; draw(); }
    canvas.style.cursor = h ? "pointer" : "default";
  });
  canvas.addEventListener("pointerleave", () => {
    if (hover) { hover = null; draw(); }
  });
  // ---- +/- buttons: depth in the bar, inputs and neurons above each column ------
  const depthPM = uiPlusMinus(() => setDepth(+1), () => setDepth(-1));
  const depthText = elt("div", "mls-head");
  const colBox = elt("div");
  if (editable) {
    const depth = elt("div", "mls-transport");
    depth.append(depthPM, depthText);
    bar.append(depth);
    holder.append(colBox);
  }
  function setSize(c, d) {
    const next = cs.slice(), v = next[c] + d;
    if (v < 1 || v > (c === 0 ? MAXIN : MAXW)) return;
    next[c] = v;
    resize(next);
  }
  function setDepth(d) {
    const hidden = cs.slice(1, -1), L = Math.min(MAXL, Math.max(0, hidden.length + d));
    if (L === hidden.length) return;
    hidden.length = Math.min(hidden.length, L);
    while (hidden.length < L) hidden.push(hidden.at(-1) ?? 4);
    resize([cs[0], ...hidden, cs.at(-1)]);
  }
  function resize(next) {  // keep the weights that still fit, draw the new ones
    const fresh = [];
    for (let k = 0; k < next.length - 1; k++) {
      const nIn = next[k], nOut = next[k + 1], old = layers[k];
      const r = () => (Math.random() * 2 - 1) / Math.sqrt(nIn);  // PyTorch's default
      fresh.push({
        W: Array.from({ length: nOut }, (_, j) =>
          Array.from({ length: nIn }, (_, i) => old?.W[j]?.[i] ?? r())),
        b: Array.from({ length: nOut }, (_, j) => old?.b[j] ?? r()),
      });
    }
    layers = fresh; hover = null;
    layout(); cv.setHeight(H); buildControls(); draw();
  }
  function buildControls() {
    if (!editable) return;
    colBox.replaceChildren();
    for (let c = 0; c < nCols - 1; c++) {
      const pm = uiPlusMinus(() => setSize(c, +1), () => setSize(c, -1));
      pm.enable(cs[c] < (c === 0 ? MAXIN : MAXW), cs[c] > 1);
      colBox.append(uiPlace(pm, colX(c), 40, "center"));
    }
    const L = cs.length - 2;
    depthPM.enable(L < MAXL, L > 0);
    depthText.textContent = `${L} HIDDEN LAYER${L === 1 ? "" : "S"}`;
  }
  buildControls();
  draw();
}
export default { render };
