// Fit a Gaussian mixture (see mixturewidget.py); controls.js is put in front of this
// module (tokens C, FONT, SYM, makeStage, stageCanvas, elt, ui*).
// Three panels in a row. 1: the target density (green) with the samples the mixture
// is fitted to (blue dots). 2: the fitted mixture of K components (vermillion density,
// one dashed 95 % ellipse each, drawn like the target's 95 % region, labelled with its weight) with the
// target's 95 % region for reference. 3: the NLL of the fitted samples and of held-out
// samples the mixture never saw, against K.
// K comes from its scrub bar, the buttons or a click on the NLL plot; the number of
// fitted samples from the second scrub bar.

const W = 1100, H = 468;
const T = 112, S = 300;                                 // panels: top, side
const X0 = [64, 408, 752];                             // panel left edges
const R95 = Math.sqrt(5.991);                          // 95 % of a 2D Gaussian: 2.45 sigma
const BLUE = C.blue, PURPLE = "#CC79A7", VERM = "#D55E00", DARK_GREEN = "#007a58";

function render({ model, el }) {
  const st = makeStage(el, W);
  const wrap = elt("div");
  wrap.style.cssText = `position:relative;width:${W}px;height:${H}px;`;
  const cv = stageCanvas(st, W, H);
  const { canvas, ctx } = cv;
  wrap.append(canvas);
  st.stage.append(wrap);
  st.onScale(() => { cv.resize(); draw(); });

  const ks = () => model.get("ks");
  const ns = () => model.get("ns");
  const idx = () => Math.max(0, ks().indexOf(model.get("k")));
  const nidx = () => Math.max(0, ns().indexOf(model.get("n")));
  const load = (src) => { const im = new Image(); im.onload = () => draw(); im.src = src; return im; };
  let target = load(model.get("image"));
  let shown = [];
  const loadModels = () => {
    shown = model.get("models").map((row) => row.map((m) => load(m.image)));
  };
  loadModels();

  // ---- controls ----------------------------------------------------------------
  const stat = uiStat(uiLabel("COMPONENTS", "K"));
  const setK = (k) => {
    const list = ks();
    k = list.reduce((a, b) => Math.abs(b - k) < Math.abs(a - k) ? b : a);
    if (k !== model.get("k")) { model.set("k", k); model.save_changes(); }
  };
  const stepK = (di) => {
    const list = ks();
    setK(list[Math.min(Math.max(idx() + di, 0), list.length - 1)]);
  };
  const pm = uiPlusMinus(() => stepK(1), () => stepK(-1));
  const scrub = uiScrub((f) => {
    const list = ks();
    setK(list[Math.round(f * (list.length - 1))]);
  });
  scrub.el.style.width = "300px";
  const statN = uiStat(uiLabel("FITTED SAMPLES"));
  const setN = (n) => {
    if (ns().includes(n) && n !== model.get("n")) { model.set("n", n); model.save_changes(); }
  };
  const pmN = uiPlusMinus(() => setN(ns()[nidx() + 1]), () => setN(ns()[nidx() - 1]));
  const scrubN = uiScrub((f) => {
    const list = ns();
    setN(list[Math.round(f * (list.length - 1))]);
  });
  scrubN.el.style.width = "300px";
  wrap.append(uiPlace(stat.el, X0[0], 6), uiPlace(pm, X0[0] + 110, 22),
    uiPlace(scrub.el, X0[0] + 190, 30),
    uiPlace(statN.el, 600, 6), uiPlace(pmN, 720, 22), uiPlace(scrubN.el, 790, 30));

  // ---- helpers -----------------------------------------------------------------
  const box = () => model.get("box");
  const line = (x0, y0, x1, y1) => { ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); };
  const lo = () => model.get("lo"), hi = () => model.get("hi");
  const span = () => hi()[0] - lo()[0];
  const mapX = (p) => (x) => X0[p] + (x - lo()[0]) / span() * S;
  const mapY = (y) => T + S - (y - lo()[1]) / span() * S;
  const tickStep = () => (span() > 60 ? 20 : span() > 30 ? 10 : 5);
  const ticksOf = (a, b) => {
    const out = [], st = tickStep();
    for (let v = Math.ceil(a / st) * st; v <= b + 1e-9; v += st) out.push(v);
    return out;
  };
  function drawImage(im, p) {  // the part of a whole-map image inside the view
    if (!(im && im.complete && im.naturalWidth)) return;
    const iw = im.naturalWidth, ih = im.naturalHeight, b = box();
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
    ctx.drawImage(im, lo()[0] / b * iw, (b - hi()[1]) / b * ih, span() / b * iw,
      span() / b * ih, X0[p], T, S, S);
  }

  function title(p, words) {  // one line per entry; an array is one line of pieces
    ctx.font = `13px ${FONT}`; ctx.textAlign = "left";
    words.forEach((w, i) => {
      let x = X0[p];
      for (const piece of [].concat(w)) {
        const t = typeof piece === "string" ? { t: piece, colour: C.ink } : piece;
        ctx.fillStyle = t.colour; ctx.fillText(t.t, x, T - 40 + 17 * i);
        x += ctx.measureText(t.t).width;
      }
    });
  }
  function frame(p, left) {
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1; ctx.setLineDash([]);
    ctx.strokeRect(X0[p] + 0.5, T + 0.5, S, S);
    ctx.font = `12px ${FONT}`; ctx.fillStyle = C.ink;
    for (const v of ticksOf(lo()[0], hi()[0])) {
      const x = mapX(p)(v);
      line(x, T + S, x, T + S - 5); line(x, T, x, T + 5);
      ctx.textAlign = "center"; ctx.fillText(String(v), x, T + S + 16);
    }
    for (const v of ticksOf(lo()[1], lo()[1] + span())) {
      const y = mapY(v);
      line(X0[p], y, X0[p] + 5, y); line(X0[p] + S, y, X0[p] + S - 5, y);
      if (left) { ctx.textAlign = "right"; ctx.fillText(String(v), X0[p] - 7, y + 4); }
    }
  }
  function axisLabel(sub, x, y) {  // m with a subscript, then (km), centred
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
  function clipped(p, fn) {
    ctx.save(); ctx.beginPath(); ctx.rect(X0[p], T, S, S); ctx.clip(); fn(); ctx.restore();
  }
  function region(p, key, width) {  // the target's 68 or 95 % region, for reference
    const px = mapX(p);
    ctx.strokeStyle = DARK_GREEN; ctx.lineWidth = width; ctx.lineJoin = "round";
    ctx.setLineDash([4, 3]);  // the same line as the ellipses
    for (const pts of (model.get("contours")[key] || [])) {
      ctx.beginPath();
      pts.forEach(([x, y], k) => (k ? ctx.lineTo(px(x), mapY(y)) : ctx.moveTo(px(x), mapY(y))));
      ctx.stroke();
    }
  }

  // ---- panels ------------------------------------------------------------------
  function panelTarget() {
    const n = model.get("n");
    title(0, [{ t: `${n} fitted samples`, colour: BLUE }, "on the target density"]);
    clipped(0, () => {
      drawImage(target, 0);
      for (const [x, y] of model.get("fitted")[nidx()]) {
        ctx.beginPath();
        ctx.arc(mapX(0)(x), mapY(y), 1.7, 0, 2 * Math.PI);
        ctx.fillStyle = BLUE; ctx.fill();
      }
    });
    frame(0, true);
    axisLabel("x", X0[0] + S / 2, T + S + 36);
    ctx.save(); ctx.translate(X0[0] - 40, T + S / 2); ctx.rotate(-Math.PI / 2);
    axisLabel("y", 0, 0); ctx.restore();
  }
  function panelMixture() {
    const k = model.get("k"), m = model.get("models")[nidx()][idx()];
    title(1, [{ t: `mixture of ${k} component${k > 1 ? "s" : ""}`, colour: VERM },
      "95 % regions: target green, components"]);
    clipped(1, () => {
      drawImage(shown[nidx()][idx()], 1);
      region(1, "95", 1.5);
      const px = mapX(1), labels = m.components.length <= 4;
      for (const c of m.components) {
        ctx.save();
        ctx.translate(px(c.x), mapY(c.y)); ctx.rotate(-c.angle * Math.PI / 180);
        ctx.beginPath();
        ctx.ellipse(0, 0, R95 * c.a / span() * S, R95 * c.b / span() * S, 0, 0, 2 * Math.PI);
        ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]); ctx.strokeStyle = VERM;
        ctx.stroke();
        ctx.restore();
        if (labels) {
          ctx.font = `12px ${FONT}`; ctx.fillStyle = VERM; ctx.textAlign = "left";
          ctx.fillText(c.w.toFixed(2), px(c.x) + 5, mapY(c.y) - 5);
        }
      }
      ctx.setLineDash([]);
    });
    frame(1, false);
  }
  function panelNll() {
    const list = ks(), nf = model.get("nll_fit")[nidx()],
      nh = model.get("nll_held")[nidx()];
    const lo = Math.min(...nf, ...nh) - 0.2, hi = Math.max(...nf, ...nh) + 0.2;
    const kk = model.get("k"), i = idx();
    title(2, [{ t: "NLL per sample", colour: C.ink },
      [{ t: `fitted ${nf[i].toFixed(2)}`, colour: BLUE }, { t: "   " },
        { t: `held out ${nh[i].toFixed(2)}`, colour: PURPLE }]]);
    const xk = (k) => X0[2] + 12 + (k - list[0]) / (list[list.length - 1] - list[0]) * (S - 24);
    const yv = (v) => T + S - (v - lo) / (hi - lo) * S;
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1;
    ctx.strokeRect(X0[2] + 0.5, T + 0.5, S, S);
    ctx.font = `12px ${FONT}`; ctx.fillStyle = C.ink; ctx.textAlign = "center";
    for (const k of list) if (true) {
      line(xk(k), T + S, xk(k), T + S - 5); ctx.fillText(String(k), xk(k), T + S + 16);
    }
    ctx.textAlign = "right";
    for (let v = Math.ceil(lo); v <= hi; v += 1) { line(X0[2], yv(v), X0[2] + 5, yv(v)); ctx.fillText(String(v), X0[2] - 7, yv(v) + 4); }
    ctx.textAlign = "center"; ctx.font = `13px ${FONT}`;
    ctx.fillText("number of components", X0[2] + S / 2, T + S + 36);
    // the gap between the curves is what the extra components learn about the fitted
    // samples alone
    ctx.beginPath();
    list.forEach((k, j) => ctx[j ? "lineTo" : "moveTo"](xk(k), yv(nh[j])));
    for (let j = list.length - 1; j >= 0; j--) ctx.lineTo(xk(list[j]), yv(nf[j]));
    ctx.closePath(); ctx.fillStyle = "rgba(204,121,167,0.14)"; ctx.fill();
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1.5; line(xk(kk), T, xk(kk), T + S);
    for (const [vals, colour] of [[nf, BLUE], [nh, PURPLE]]) {
      ctx.beginPath();
      list.forEach((k, j) => ctx[j ? "lineTo" : "moveTo"](xk(k), yv(vals[j])));
      ctx.lineWidth = 2; ctx.strokeStyle = colour; ctx.stroke();
      list.forEach((k, j) => {
        ctx.beginPath(); ctx.arc(xk(k), yv(vals[j]), k === kk ? 5 : 2.8, 0, 2 * Math.PI);
        ctx.fillStyle = colour; ctx.fill();
      });
    }
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    ctx.setLineDash([]); ctx.lineCap = "round";
    panelTarget(); panelMixture(); panelNll();
    const list = ks(), nlist = ns();
    stat.set(String(model.get("k")));
    scrub.set(idx() / (list.length - 1));
    pm.enable(idx() < list.length - 1, idx() > 0);
    statN.set(String(model.get("n")));
    scrubN.set(nidx() / (nlist.length - 1));
    pmN.enable(nidx() < nlist.length - 1, nidx() > 0);
  }

  canvas.addEventListener("click", (ev) => {
    const [x, y] = cv.xy(ev);
    if (x >= X0[2] && x <= X0[2] + S && y >= T - 10 && y <= T + S + 10) {
      const list = ks(), f = Math.min(Math.max((x - X0[2] - 12) / (S - 24), 0), 1);
      setK(list[0] + f * (list[list.length - 1] - list[0]));
    }
  });
  for (const name of ["k", "n", "lo", "hi", "n_held", "contours", "fitted",
    "nll_fit", "nll_held"])
    model.on(`change:${name}`, () => draw());
  model.on("change:models", () => { loadModels(); draw(); });
  model.on("change:image", () => { target = load(model.get("image")); });
  draw();
}
export default { render };
