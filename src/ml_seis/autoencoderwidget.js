// The Part 2c autoencoder flow (see autoencoderwidget.py); controls.js is put in front of
// this module (tokens C, FONT, SYM and the ui* controls).
//
// One row, left to right: the waveform of a test record, the encoder, the latent vector,
// the decoder, the spectrogram it rebuilds and the target it is compared with. The latent
// vector is the one thing drawn in full colour and boxed (what we keep): four strips, one
// per number, each with the latent numbers of all test records as faint dots in the colour
// of their class and the chosen record as a large marker. The decoder, the rebuild and the
// target are drawn faint and bracketed as scaffolding for training. A class and a record
// number choose the input; everything is computed in the notebook and sent once.

const W = 1300, H = 400;
const CY = 215;  // the centre line of the row
const X = {
  wave: [24, 250], enc: [296, 392], lat: [452, 700], dec: [760, 856],
  out: [916, 1076], tgt: [1124, 1284],
};
const SPEC_H = 112, WAVE_H = 112, STRIP_H = 200;
const LOCAL_CSS = `
.af-stepper { display: inline-flex; align-items: center; gap: 8px; padding: 2px 0; }
.af-stepper .mls-number { width: 3ch; text-align: center; font-size: 15px;
  -webkit-user-select: text; user-select: text; }
.af-stepper .af-of { color: ${C.ink2}; font-size: 13px;
  font-variant-numeric: tabular-nums; white-space: nowrap; }
`;
const minus = (s) => s.replace("-", "−");
const signed = (v) => (v < 0 ? "−" : "+") + Math.abs(v).toFixed(2);

function render({ model, el }) {
  const st = makeStage(el, W);
  const local = elt("style");
  local.textContent = LOCAL_CSS;
  st.stage.append(local);

  const classes = model.get("classes"), colours = model.get("colours");
  const lut = model.get("cmap"), records = model.get("records");
  const all = model.get("latent"), axes = model.get("axes");
  const D = all.values[0].length;
  // one scale for the four numbers, covering nearly all test records
  const R = (() => {
    const v = all.values.flat().map(Math.abs).sort((a, b) => a - b);
    return v[Math.floor(0.995 * (v.length - 1))] * 1.08;
  })();
  // a fixed horizontal spread of the dots of a strip, so that they do not move
  const jitter = all.values.map((_, i) => ((i * 2654435761) % 1000) / 1000 - 0.5);

  let cls = Math.max(0, records.findIndex((r) => r.length)), n = 0;
  const rec = () => records[cls][n];

  // ---- header ------------------------------------------------------------------------------
  const bar = elt("div", "mls-bar");
  const seg = uiSegmented(classes.map((c, k) => [k, c]), cls, (v) => {
    cls = v; n = Math.min(n, records[cls].length - 1); sync(); draw();
  });
  const root = elt("div", "af-stepper");
  const down = elt("button", "mls-pm", "−"), up = elt("button", "mls-pm", "+");
  const input = elt("input", "mls-number");
  input.type = "text"; input.inputMode = "numeric";
  const of = elt("span", "af-of");
  const go = (v) => {
    v = Math.min(Math.max(Math.round(v), 1), records[cls].length);
    n = v - 1; sync(); draw();
  };
  down.onclick = () => go(n); up.onclick = () => go(n + 2);  // n is 0-based, the field 1-based
  input.addEventListener("change", () => { const v = Number(input.value); if (Number.isFinite(v)) go(v); else sync(); });
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") input.blur();
    else if (ev.key === "ArrowUp") { ev.preventDefault(); go(n + 2); }
    else if (ev.key === "ArrowDown") { ev.preventDefault(); go(n); }
  });
  root.append(down, input, of, up);
  function sync() {
    input.value = String(n + 1);
    of.textContent = `/ ${records[cls].length}`;
    down.disabled = n <= 0; up.disabled = n >= records[cls].length - 1;
    seg.set(cls);
  }
  bar.append(uiField(uiLabel("CLASS"), seg), uiField(uiLabel("TEST RECORD"), root));
  st.stage.append(bar);

  const fig = stageCanvas(st, W, H);
  fig.canvas.tabIndex = 0;
  fig.canvas.style.outline = "none";
  st.stage.append(fig.canvas);
  const ctx = fig.ctx;

  // ---- drawing helpers ----------------------------------------------------------------------
  const text = (s, x, y, colour = C.ink, align = "left", px = 13, style = "") => {
    ctx.font = `${style} ${px}px ${FONT}`.trim();
    ctx.fillStyle = colour; ctx.textAlign = align; ctx.textBaseline = "alphabetic";
    ctx.fillText(s, x, y);
  };
  const cap = (s, x, y, colour = C.ink, align = "left") => {  // bold small capitals
    ctx.font = `bold 12px ${FONT}`;
    ctx.fillStyle = colour; ctx.textAlign = align; ctx.textBaseline = "alphabetic";
    if ("letterSpacing" in ctx) ctx.letterSpacing = "1px";
    ctx.fillText(s, x, y);
    if ("letterSpacing" in ctx) ctx.letterSpacing = "0px";
  };
  const italic = (s, x, y, px = 17, colour = C.ink, align = "center") => {  // a symbol
    ctx.font = `italic ${px}px ${SYM}`;
    ctx.fillStyle = colour; ctx.textAlign = align; ctx.textBaseline = "alphabetic";
    ctx.fillText(s, x, y);
  };
  const zLabel = (d, x, y, colour = C.ink) => {  // z with its subscript, centred at x
    ctx.font = `italic 18px ${SYM}`;
    const wz = ctx.measureText("z").width;
    ctx.font = `12px ${FONT}`;
    const wn = ctx.measureText(String(d + 1)).width;
    const x0 = x - (wz + wn + 1) / 2;
    ctx.fillStyle = colour; ctx.textBaseline = "alphabetic"; ctx.textAlign = "left";
    ctx.font = `italic 18px ${SYM}`; ctx.fillText("z", x0, y);
    ctx.font = `12px ${FONT}`; ctx.fillText(String(d + 1), x0 + wz + 1, y + 4);
  };
  const line = (x1, y1, x2, y2, colour = C.ink2, lw = 1.2, dash = []) => {
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
    ctx.strokeStyle = colour; ctx.lineWidth = lw; ctx.setLineDash(dash); ctx.stroke();
    ctx.setLineDash([]);
  };
  const head = (x, y, dir, colour = C.ink2) => {  // an arrow head pointing right (dir 1) or left
    ctx.beginPath(); ctx.moveTo(x - dir * 6, y - 4); ctx.lineTo(x, y); ctx.lineTo(x - dir * 6, y + 4);
    ctx.strokeStyle = colour; ctx.lineWidth = 1.2; ctx.setLineDash([]); ctx.stroke();
  };
  const arrow = (x1, x2, y, colour = C.ink2, both = false) => {
    line(x1, y, x2, y, colour); head(x2, y, 1, colour); if (both) head(x1, y, -1, colour);
  };
  const box = (x1, x2, y1, y2, colour, lw = 1.2, dash = []) => {
    ctx.beginPath(); ctx.roundRect(x1, y1, x2 - x1, y2 - y1, 4);
    ctx.strokeStyle = colour; ctx.lineWidth = lw; ctx.setLineDash(dash); ctx.stroke();
    ctx.setLineDash([]);
  };

  // ---- the parts ----------------------------------------------------------------------------
  function drawWave(r, colour) {
    const [x0, x1] = X.wave, top = CY - WAVE_H / 2, mid = CY;
    cap("INPUT", x0, top - 34);
    italic("u", x0 + 52, top - 34, 17);
    text(`${classes[cls]} test record #${rec().index}`, x0, top - 16, C.ink2, "left", 12);
    line(x0, mid, x1, mid, C.grid, 1);
    ctx.beginPath();
    r.wave.forEach((v, i) => {
      const x = x0 + (x1 - x0) * i / (r.wave.length - 1), y = mid - v * WAVE_H / 2;
      if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    });
    ctx.strokeStyle = colour; ctx.lineWidth = 1.1; ctx.stroke();
    // the time axis
    const yb = top + WAVE_H + 8;
    line(x0, yb, x1, yb, C.ink2, 1);
    [0, 5, 10, 15].forEach((t) => {
      const x = x0 + (x1 - x0) * t / axes.duration;
      line(x, yb, x, yb - 4, C.ink2, 1);
      text(String(t), x, yb + 15, C.ink2, "center", 11.5);
    });
    text("time (s)", x1, yb + 30, C.ink2, "right", 11.5);
  }

  function drawSpec(values, x0, x1, ylab) {
    const rows = values.length, cols = values[0].length, top = CY - SPEC_H / 2;
    const cw = (x1 - x0) / cols, ch = SPEC_H / rows;
    for (let b = 0; b < cols; b++) for (let f = 0; f < rows; f++) {
      const t = Math.min(Math.max((values[f][b] - sp.lo) / (sp.hi - sp.lo), 0), 1);
      ctx.fillStyle = lut[Math.round(t * (lut.length - 1))];
      ctx.fillRect(x0 + b * cw, top + SPEC_H - (f + 1) * ch, cw + 0.6, ch + 0.6);
    }
    ctx.fillStyle = "rgba(255, 255, 255, 0.3)";  // faint: scaffolding, not the product
    ctx.fillRect(x0, top, x1 - x0, SPEC_H);
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1; ctx.strokeRect(x0, top, x1 - x0, SPEC_H);
    const yb = top + SPEC_H;
    [0, 5, 10, 15].forEach((t) => {
      const x = x0 + (x1 - x0) * t / axes.duration;
      line(x, yb, x, yb + 4, C.ink2, 1);
      text(String(t), x, yb + 17, C.ink2, "center", 11.5);
    });
    text("time (s)", x1, yb + 32, C.ink2, "right", 11.5);
    if (ylab) {
      [5, 10].forEach((f) => {
        const y = top + SPEC_H * (1 - (f - axes.fmin) / (axes.fmax - axes.fmin));
        line(x0, y, x0 + 4, y, C.ink2, 1);
        text(String(f), x0 - 6, y + 4, C.ink2, "right", 11.5);
      });
      text("Hz", x0 - 6, top - 6, C.ink2, "right", 11.5);
    }
  }

  function drawLatent(r) {
    const [x0, x1] = X.lat, top = CY - STRIP_H / 2, pitch = (x1 - x0) / D;
    // the boxed vector: what we keep
    box(x0 - 10, x1 + 10, top - 66, top + STRIP_H + 64, C.orange, 1.6);
    const capText = "LATENT VECTOR";
    ctx.font = `bold 12px ${FONT}`;
    cap(capText, x0 + 6, top - 46, C.orange);
    ctx.letterSpacing = "1px";
    const wc = ctx.measureText(capText).width;
    ctx.letterSpacing = "0px";
    italic("z", x0 + 6 + wc + 8, top - 46, 17, C.orange, "left");
    text("what we keep", (x0 + x1) / 2, top - 26, C.orange, "center", 14);
    text("four numbers per record", (x0 + x1) / 2, top - 9, C.ink2, "center", 12);
    for (let d = 0; d < D; d++) {
      const cx = x0 + (d + 0.5) * pitch, yz = CY, sc = (STRIP_H / 2) / R;
      line(cx, top, cx, top + STRIP_H, C.grid, 1);
      line(cx - 22, yz, cx + 22, yz, C.ink2, 1);
      // every test record, a faint dot in the colour of its class
      all.values.forEach((v, i) => {
        const y = yz - Math.max(-R, Math.min(R, v[d])) * sc;
        ctx.globalAlpha = 0.32; ctx.fillStyle = colours[all.label[i]];
        ctx.beginPath(); ctx.arc(cx + jitter[i] * 34, y, 2.4, 0, 2 * Math.PI); ctx.fill();
      });
      ctx.globalAlpha = 1;
      // the chosen record
      const v = r.latent[d], y = yz - Math.max(-R, Math.min(R, v)) * sc;
      line(cx - 22, y, cx + 22, y, colours[cls], 1.6);
      ctx.beginPath(); ctx.arc(cx, y, 7.5, 0, 2 * Math.PI);
      ctx.fillStyle = colours[cls]; ctx.fill();
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.8; ctx.stroke();
      zLabel(d, cx, top + STRIP_H + 24);
      text(signed(v), cx, top + STRIP_H + 46, C.ink, "center", 14.5);
    }
  }

  function drawNet(x0, x1, name, sub, faint) {
    const top = CY - 46, bot = CY + 46;
    ctx.globalAlpha = faint ? 0.65 : 1;
    box(x0, x1, top, bot, faint ? C.ink2 : C.ink, 1.3, faint ? [5, 4] : []);
    text(name, (x0 + x1) / 2, CY + 1, faint ? C.ink2 : C.ink, "center", 15);
    text(sub, (x0 + x1) / 2, CY + 19, C.ink2, "center", 11.5);
    ctx.globalAlpha = 1;
  }

  // ---- draw ------------------------------------------------------------------------------------
  let sp = { lo: 0, hi: 1 };
  function draw() {
    ctx.clearRect(0, 0, W, H);
    const r = rec();
    // one colour scale per record, shared by the rebuild and the target
    const flat = r.target.flat();
    sp = { lo: Math.min(...flat), hi: Math.max(...flat) };

    drawWave(r, colours[cls]);
    arrow(X.wave[1] + 14, X.enc[0] - 8, CY);
    drawNet(X.enc[0], X.enc[1], "encoder", "the CNN of 2b", false);
    arrow(X.enc[1] + 8, X.lat[0] - 22, CY);
    drawLatent(r);
    arrow(X.lat[1] + 22, X.dec[0] - 8, CY, C.ink2);
    drawNet(X.dec[0], X.dec[1], "decoder", "", true);
    arrow(X.dec[1] + 8, X.out[0] - 22, CY, C.ink2);

    // the rebuild and the target: faint, only for training
    drawSpec(r.rebuilt, X.out[0], X.out[1], true);
    drawSpec(r.target, X.tgt[0], X.tgt[1], false);
    const top = CY - SPEC_H / 2;
    cap("REBUILT", X.out[0], top - 34, C.ink2);
    italic("ŝ", X.out[0] + 70, top - 34, 17, C.ink2, "left");
    text("spectrogram from z alone", X.out[0], top - 16, C.ink2, "left", 12);
    cap("TARGET", X.tgt[0], top - 34, C.ink2);
    italic("s", X.tgt[0] + 62, top - 34, 17, C.ink2, "left");
    text("spectrogram of the record", X.tgt[0], top - 16, C.ink2, "left", 12);
    const xm = (X.out[1] + X.tgt[0]) / 2;
    arrow(X.out[1] + 8, X.tgt[0] - 8, CY, C.ink2, true);
    text("loss", xm, CY - 9, C.ink2, "center", 12.5);
    // the target is computed from the record itself
    const yt = 30, xs = (X.wave[0] + X.wave[1]) / 2, xe = (X.tgt[0] + X.tgt[1]) / 2;
    ctx.globalAlpha = 0.7;
    const y0 = CY - WAVE_H / 2 - 50, y1 = top - 50;
    line(xs, y0, xs, yt, C.ink2, 1.2, [4, 3]);
    line(xs, yt, xe, yt, C.ink2, 1.2, [4, 3]);
    line(xe, yt, xe, y1, C.ink2, 1.2, [4, 3]);
    ctx.beginPath(); ctx.moveTo(xe - 4, y1 - 6); ctx.lineTo(xe, y1); ctx.lineTo(xe + 4, y1 - 6);
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1.2; ctx.stroke();
    ctx.globalAlpha = 1;
    text("log |STFT| of the record", (xs + xe) / 2, yt - 7, C.ink2, "center", 12);

    // the scaffolding bracket under decoder, rebuild and target
    const yb = CY + SPEC_H / 2 + 62;
    ctx.globalAlpha = 0.8;
    line(X.dec[0], yb, X.tgt[1], yb, C.ink2, 1.2);
    line(X.dec[0], yb, X.dec[0], yb - 7, C.ink2, 1.2);
    line(X.tgt[1], yb, X.tgt[1], yb - 7, C.ink2, 1.2);
    ctx.globalAlpha = 1;
    text("only needed to train the encoder", (X.dec[0] + X.tgt[1]) / 2, yb + 20, C.ink2, "center", 13.5);
  }

  fig.canvas.addEventListener("keydown", (ev) => {
    if (ev.key === "ArrowRight") go(n + 2);
    else if (ev.key === "ArrowLeft") go(n);
  });
  sync();
  draw();
}

export default { render };
