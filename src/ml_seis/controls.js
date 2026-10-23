// Shared front end of the course playgrounds. widgetkit.esm() puts this file in
// front of each widget's own module, so the names below are plain globals there.
// It holds the colour tokens and type of the course style (light; see
// styles/README.md) and one set of controls, so both playgrounds read alike: a
// header bar (transport buttons, a big counter, labelled pickers), a scrub bar,
// round +/- buttons and stat tiles. The whole figure is laid out on a fixed
// logical stage that is scaled to the cell width (makeStage).
const C = {
  bg: "#ffffff", ink: "#111111", ink2: "#5c6370", grid: "#d9dde4",
  blue: "#0072B2", orange: "#D55E00", green: "#009E73", sky: "#56B4E9",
  grey: "#5c6370",
};
const FONT = "'DejaVu Sans', Helvetica, Arial, sans-serif";
const SYM = "'STIX Two Text', 'STIXGeneral', 'Times New Roman', serif";

const SUP = { "-": "⁻", "(": "⁽", ")": "⁾", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴",
  5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
const SUB = { 0: "₀", 1: "₁", 2: "₂", 3: "₃", 4: "₄", 5: "₅", 6: "₆", 7: "₇",
  8: "₈", 9: "₉" };
const sup = (n) => String(n).split("").map((c) => SUP[c] || "").join("");
const sub = (n) => String(n).split("").map((c) => SUB[c] || "").join("");
// a stat value: plain for 0.01 <= |v| < 1000, else a x 10^e
const fmtStat = (v, digits = 2) => {
  if (!isFinite(v)) return "—";
  if (v === 0) return "0";
  v = +v.toPrecision(digits);  // round first: 9.96e-4 reads 1×10⁻³, not 10×10⁻⁴
  const e = Math.floor(Math.log10(Math.abs(v)) + 1e-9);
  if (e >= -2 && e <= 2) return String(+v.toPrecision(digits));
  return `${+(v / 10 ** e).toPrecision(digits)}×10${sup(e)}`;
};

const ICONS = {  // 24 x 24 material-style glyphs
  play: "M8 5v14l11-7z",
  pause: "M6 19h4V5H6v14zm8-14v14h4V5h-4z",
  step: "M6 18l8.5-6L6 6v12zM16 6v12h2V6h-2z",
  reset: "M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 " +
    "4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z",
};
const CHEVRON = "data:image/svg+xml;utf8," + encodeURIComponent(
  `<svg xmlns="http://www.w3.org/2000/svg" width="10" height="6">` +
  `<path d="M0 0l5 6 5-6z" fill="${C.ink2}"/></svg>`);

const KIT_CSS = `
.mls { font: 13px/1.25 ${FONT}; color: ${C.ink}; background: ${C.bg};
  -webkit-user-select: none; user-select: none; }
.mls * { box-sizing: border-box; }
.mls-bar { display: flex; align-items: flex-end; flex-wrap: wrap; gap: 10px 20px;
  padding: 4px 0 10px; }
.mls-transport { display: flex; align-items: center; gap: 8px; }
.mls-btn { width: 32px; height: 32px; border-radius: 50%; padding: 0;
  border: 1px solid ${C.grid}; background: ${C.bg}; color: ${C.ink};
  display: grid; place-items: center; cursor: pointer; }
.mls-btn:hover { border-color: ${C.ink2}; }
.mls-btn svg { width: 16px; height: 16px; fill: currentColor; }
.mls-btn.mls-play { width: 46px; height: 46px; border: none; background: ${C.ink};
  color: ${C.bg}; }
.mls-btn.mls-play:hover { background: #3a3f47; }
.mls-btn.mls-play svg { width: 22px; height: 22px; }
.mls-field { display: flex; flex-direction: column; gap: 4px; }
.mls-field.mls-off { opacity: 0.35; pointer-events: none; }
.mls-label { font-size: 10.5px; letter-spacing: 0.08em; color: ${C.ink2};
  white-space: nowrap; }
.mls-sym { font: italic 14px ${SYM}; letter-spacing: 0; color: ${C.ink}; }
.mls-value { font-size: 26px; line-height: 1; font-variant-numeric: tabular-nums;
  white-space: nowrap; }
.mls-value small { font-size: 13px; color: ${C.ink2}; margin-left: 4px; }
.mls-counter .mls-value { min-width: 7.2ch; }
.mls-select { appearance: none; -webkit-appearance: none; font: 14px ${FONT};
  color: ${C.ink}; background: transparent url("${CHEVRON}") no-repeat right 2px
  center; border: none; border-bottom: 1px solid ${C.ink2}; border-radius: 0;
  padding: 2px 18px 3px 0; cursor: pointer; }
.mls-select:focus { outline: none; border-bottom-color: ${C.ink}; }
.mls-number { font: 14px ${FONT}; color: ${C.ink}; background: transparent;
  border: none; border-bottom: 1px solid ${C.ink2}; border-radius: 0;
  padding: 2px 0 3px; font-variant-numeric: tabular-nums; }
.mls-number:focus { outline: none; border-bottom-color: ${C.ink}; }
.mls-seg { display: inline-flex; border: 1px solid ${C.grid}; border-radius: 999px;
  overflow: hidden; }
.mls-seg button { font: 13px ${FONT}; padding: 3px 11px; border: none;
  background: ${C.bg}; color: ${C.ink2}; cursor: pointer; }
.mls-seg button.mls-on { background: ${C.ink}; color: ${C.bg}; }
.mls-rule { width: 1px; align-self: stretch; background: ${C.grid}; }
.mls-status { margin-left: auto; align-self: center; color: ${C.ink2};
  font-size: 12px; min-width: 7em; text-align: right; }
.mls-progress { display: inline-block; width: 70px; height: 4px; border-radius: 2px;
  background: ${C.grid}; overflow: hidden; vertical-align: middle; }
.mls-progress > i { display: block; height: 100%; width: 0; background: ${C.orange};
  transition: width 0.12s linear; }
.mls-scrub { position: relative; height: 16px; cursor: pointer; touch-action: none; }
.mls-scrub > div { position: absolute; top: 7px; height: 3px; border-radius: 2px; }
.mls-scrub .mls-knob { top: 3px; width: 11px; height: 11px; margin-left: -5.5px;
  border-radius: 50%; background: ${C.orange}; }
.mls-pm { width: 22px; height: 22px; border-radius: 50%; padding: 0;
  border: 1px solid ${C.grid}; background: ${C.bg}; color: ${C.ink};
  font: 15px/1 ${FONT}; cursor: pointer; display: inline-grid; place-items: center; }
.mls-pm:hover:not(:disabled) { border-color: ${C.ink2}; }
.mls-pm:disabled { opacity: 0.3; cursor: default; }
.mls-head { font-size: 11px; letter-spacing: 0.1em; color: ${C.ink2};
  white-space: nowrap; }
.mls-abs { position: absolute; }
`;

function elt(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function icon(name) {
  return `<svg viewBox="0 0 24 24"><path d="${ICONS[name]}"/></svg>`;
}

// A label in small capitals with an optional italic symbol: uiLabel("STEP SIZE", "η").
function uiLabel(words, symbol = "", unit = "") {
  const e = elt("span", "mls-label", words);
  if (symbol) {
    e.append(" ");
    e.append(elt("span", "mls-sym", symbol));
  }
  if (unit) e.append(` (${unit})`);
  return e;
}
function uiField(lab, control) {
  const e = elt("div", "mls-field");
  e.append(lab, control);
  e.setOff = (off) => e.classList.toggle("mls-off", off);
  return e;
}

// Fixed logical stage of width W, scaled to the cell width. Returns the stage
// (lay the figure out in it in logical px) and onScale(cb) for canvases.
function makeStage(el, W) {
  const outer = elt("div");
  outer.style.cssText = "width:100%;overflow:hidden;position:relative;";
  const stage = elt("div", "mls");
  stage.style.cssText = `width:${W}px;transform-origin:0 0;position:relative;`;
  const style = elt("style");
  style.textContent = KIT_CSS;
  outer.append(style, stage);
  el.append(outer);
  let k = 1;
  const listeners = [];
  const fit = () => {
    const avail = outer.clientWidth || W;
    const nk = Math.min(Math.max(avail / W, 0.4), 1.35);
    stage.style.transform = `scale(${nk})`;
    stage.style.marginLeft = `${Math.max((avail - W * nk) / 2, 0)}px`;
    outer.style.height = `${stage.offsetHeight * nk}px`;
    if (Math.abs(nk - k) > 1e-3) { k = nk; listeners.forEach((cb) => cb(k)); }
  };
  new ResizeObserver(fit).observe(outer);
  new ResizeObserver(fit).observe(stage);
  return { stage, scale: () => k, onScale: (cb) => listeners.push(cb), fit };
}

// A canvas of W x H logical px that stays sharp at the stage scale; setHeight
// lets a figure grow (e.g. for a wider network) instead of shrinking its marks.
function stageCanvas(st, W, H) {
  const canvas = elt("canvas");
  canvas.style.cssText = `width:${W}px;display:block;touch-action:none;`;
  const ctx = canvas.getContext("2d");
  const resize = () => {
    const r = (window.devicePixelRatio || 1) * Math.max(st.scale(), 1);
    canvas.style.height = `${H}px`;
    canvas.width = Math.round(W * r); canvas.height = Math.round(H * r);
    ctx.setTransform(r, 0, 0, r, 0, 0);
  };
  resize();
  const setHeight = (h) => { if (h !== H) { H = h; resize(); } };
  // pointer position in logical px
  const xy = (ev) => {
    const b = canvas.getBoundingClientRect();
    return [(ev.clientX - b.left) * W / b.width, (ev.clientY - b.top) * H / b.height];
  };
  return { canvas, ctx, resize, setHeight, xy };
}

// Reset, play/pause and step, as in TensorFlow Playground.
function uiTransport({ onReset, onPlay, onStep, resetTitle }) {
  const e = elt("div", "mls-transport");
  const reset = elt("button", "mls-btn"), play = elt("button", "mls-btn mls-play"),
    step = elt("button", "mls-btn");
  reset.innerHTML = icon("reset"); reset.title = resetTitle; reset.onclick = onReset;
  play.onclick = onPlay;
  step.innerHTML = icon("step"); step.title = "one step forward"; step.onclick = onStep;
  e.append(reset, play, step);
  const setPlaying = (on) => {
    play.innerHTML = icon(on ? "pause" : "play");
    play.title = on ? "pause" : "play";
  };
  setPlaying(false);
  return { el: e, setPlaying };
}

// A big number under a small label: uiStat("EPOCH") -> set(value, of).
function uiStat(lab, cls = "") {
  const e = elt("div", `mls-field ${cls}`);
  const v = elt("div", "mls-value");
  e.append(typeof lab === "string" ? uiLabel(lab) : lab, v);
  const set = (value, small = "") => {
    v.textContent = value;
    if (small) v.append(elt("small", "", small));
  };
  return { el: e, set };
}

// A dropdown of [value, text] pairs; onChange gets the value (number if the
// option values are numbers).
function uiSelect(options, value, onChange) {
  const e = elt("select", "mls-select");
  for (const [v, t] of options) {
    const o = elt("option", "", t);
    o.value = String(v);
    e.append(o);
  }
  const numeric = typeof options[0][0] === "number";
  e.value = String(value);
  e.onchange = () => onChange(numeric ? Number(e.value) : e.value);
  e.set = (v) => { e.value = String(v); };
  return e;
}

// A pill of mutually exclusive buttons.
function uiSegmented(options, value, onChange) {
  const e = elt("div", "mls-seg");
  const buttons = options.map(([v, t]) => {
    const b = elt("button", "", t);
    b.onclick = () => { e.set(v); onChange(v); };
    b.dataset.v = String(v);
    e.append(b);
    return b;
  });
  e.set = (v) => buttons.forEach((b) => b.classList.toggle("mls-on",
    b.dataset.v === String(v)));
  e.set(value);
  return e;
}

// A number typed in freely, rounded and clamped to [min, max] when it is
// committed (Enter or leaving the field); onChange gets the number.
function uiNumber(value, { min, max }, onChange) {
  const e = elt("input", "mls-number");
  e.type = "text"; e.inputMode = "numeric"; e.value = String(value);
  e.size = String(max).length + 1;
  let last = value;
  const commit = () => {
    const v = Math.round(Number(e.value));
    const ok = Number.isFinite(v) ? Math.min(Math.max(v, min), max) : last;
    e.value = String(ok);
    if (ok !== last) { last = ok; onChange(ok); }
  };
  e.addEventListener("change", commit);
  e.addEventListener("keydown", (ev) => { if (ev.key === "Enter") e.blur(); });
  return e;
}

// Round + and - buttons.
function uiPlusMinus(onPlus, onMinus) {
  const e = elt("div");
  e.style.cssText = "display:flex;gap:6px;";
  const plus = elt("button", "mls-pm", "+"), minus = elt("button", "mls-pm", "−");
  plus.onclick = onPlus; minus.onclick = onMinus;
  e.append(plus, minus);
  e.enable = (canPlus, canMinus) => { plus.disabled = !canPlus; minus.disabled = !canMinus; };
  return e;
}

// Thin progress bar that can be dragged: onScrub(fraction in [0, 1]).
function uiScrub(onScrub) {
  const e = elt("div", "mls-scrub");
  const track = elt("div"), fill = elt("div"), knob = elt("div", "mls-knob");
  track.style.cssText = `left:0;right:0;background:${C.grid};`;
  fill.style.cssText = `left:0;width:0;background:${C.orange};`;
  e.append(track, fill, knob);
  const frac = (ev) => {
    const b = e.getBoundingClientRect();
    return Math.min(Math.max((ev.clientX - b.left) / b.width, 0), 1);
  };
  let down = false;
  e.addEventListener("pointerdown", (ev) => {
    down = true; e.setPointerCapture(ev.pointerId); onScrub(frac(ev));
  });
  e.addEventListener("pointermove", (ev) => { if (down) onScrub(frac(ev)); });
  e.addEventListener("pointerup", () => { down = false; });
  e.addEventListener("pointercancel", () => { down = false; });
  const set = (f) => {
    const p = `${Math.min(Math.max(f, 0), 1) * 100}%`;
    fill.style.width = p; knob.style.left = p;
  };
  set(0);
  return { el: e, set };
}

// Absolutely placed element on the stage (logical px).
function uiProgress() {  // a thin bar for work that Python reports on, 0 to 1
  const e = elt("span", "mls-progress"), fill = elt("i");
  e.append(fill);
  return { el: e, set: (f) => { fill.style.width = `${Math.min(Math.max(f, 0), 1) * 100}%`; } };
}
function uiPlace(e, x, y, align = "left") {
  e.classList.add("mls-abs");
  e.style.top = `${y}px`;
  if (align === "center") { e.style.left = `${x}px`; e.style.transform = "translateX(-50%)"; }
  else if (align === "right") e.style.right = `${x}px`;
  else e.style.left = `${x}px`;
  return e;
}
