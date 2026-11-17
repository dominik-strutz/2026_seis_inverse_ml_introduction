// Step through numbered samples (see stepperwidget.py); controls.js is put in front of
// this module (tokens C, FONT, KIT_CSS, elt, uiLabel). One row: LABEL  (−) [ n ] / N (+).
// The buttons and the arrow keys (up, down) move by one, a number typed in the field
// is clamped to [start, stop] when it is committed (Enter or leaving the field).
// Nothing is dragged: a slider is too coarse to land on one sample.

const STEPPER_CSS = `
.mls-stepper { display: inline-flex; align-items: center; gap: 8px; padding: 2px 0; }
.mls-stepper .mls-number { width: 4ch; text-align: center; font-size: 15px;
  -webkit-user-select: text; user-select: text; }
.mls-stepper .mls-of { color: ${C.ink2}; font-size: 13px;
  font-variant-numeric: tabular-nums; white-space: nowrap; }
`;

function render({ model, el }) {
  const style = elt("style");
  style.textContent = KIT_CSS + STEPPER_CSS;
  const root = elt("div", "mls mls-stepper");
  el.append(style, root);

  const lo = () => model.get("start"), hi = () => model.get("stop");
  const clamp = (v) => Math.min(Math.max(Math.round(v), lo()), hi());
  const set = (v) => {
    v = clamp(v);
    if (v !== model.get("value")) { model.set("value", v); model.save_changes(); }
    show();
  };

  const label = uiLabel(model.get("label").toUpperCase());
  const minus = elt("button", "mls-pm", "−"), plus = elt("button", "mls-pm", "+");
  minus.title = "previous"; plus.title = "next";
  minus.onclick = () => set(model.get("value") - 1);
  plus.onclick = () => set(model.get("value") + 1);

  const input = elt("input", "mls-number");
  input.type = "text"; input.inputMode = "numeric";
  input.addEventListener("change", () => {
    const v = Number(input.value);
    if (Number.isFinite(v)) set(v); else show();
  });
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") input.blur();
    else if (ev.key === "ArrowUp") { ev.preventDefault(); set(model.get("value") + 1); }
    else if (ev.key === "ArrowDown") { ev.preventDefault(); set(model.get("value") - 1); }
  });
  const of = elt("span", "mls-of");

  function show() {
    input.value = String(model.get("value"));
    of.textContent = `/ ${hi()}`;
    minus.disabled = model.get("value") <= lo();
    plus.disabled = model.get("value") >= hi();
  }

  root.append(label, minus, input, of, plus);
  show();
  model.on("change:value", show);
  model.on("change:stop", show);
  model.on("change:start", show);
}

export default { render };
