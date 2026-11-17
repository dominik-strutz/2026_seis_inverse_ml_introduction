// The Part 2c cluster playground (see clusterwidget.py); controls.js is put in front of
// this module (tokens C, FONT, SYM and the ui* controls).
//
// Left, three panels with the same records in the same two numbers of the latent vector:
// the clusters k-means found (colours that mean nothing, the centres as numbered discs),
// the clusters named by the labels of their five records (a hue per class, a lighter or
// darker tone of it for clusters that share a name), and the true labels. The records
// whose label was looked up are ringed. Right, one row per cluster: the true labels of
// its labelled records (the vote), the name the majority gives it, and how its test
// records fare. Below, a read-out of whatever is pointed at: a cluster (its disc in a
// panel, its row, one of its vote dots) shows the true labels of all its records, a
// record shows its label and its latent vector. A cluster in focus, or a record, is
// marked in all three panels; a click keeps a cluster in focus. The records switch
// chooses the training or the test records for all three panels (on the test records the
// middle panel is the prediction). Every number of clusters and every draw of the
// labelled records arrives at the start, so nothing waits for Python; only k and the draw
// travel back (the notebook uses them).

const W = 1490;
let H = 800;  // grows with the table: one row per cluster
const PGAP = 30, PW = (W - 64 - 10 - 2 * PGAP) / 3;  // width of a panel and the gap between panels
const PANELS = [0, 1, 2].map((i) => {
  const l = 64 + i * (PW + PGAP);
  return { l, r: l + PW };
});
const PT = 34, PB = 424;  // top and bottom of the panels
const T0 = 24;  // left edge of the table: one column, one row per cluster
const TABLE = {
  x: T0, right: T0 + 760, head: 578, row0: 618, pitch: 38,
  disc: T0 + 24, dots: T0 + 74, dotStep: 26, swatch: T0 + 232, name: T0 + 246,
  votes: T0 + 312, testN: T0 + 500, bar: T0 + 524, barW: 120, pct: T0 + 660,
};
const STRIP = { y1: 496, y2: 526, x0: 24, x1: 760 };
const LOCAL_CSS = `
.cn-stepper { display: inline-flex; align-items: center; gap: 8px; padding: 2px 0; }
.cn-stepper .mls-number { width: 3ch; text-align: center; font-size: 15px;
  -webkit-user-select: text; user-select: text; }
.cn-stepper .cn-of { color: ${C.ink2}; font-size: 13px;
  font-variant-numeric: tabular-nums; white-space: nowrap; }
.cn-stats { display: flex; gap: 26px; margin-left: auto; align-items: flex-end; }
.cn-explore { position: absolute; left: 820px; top: 566px; width: 640px; color: ${C.ink};
  font-size: 15px; line-height: 1.45; pointer-events: none; }
.cn-explore .cn-head { margin-bottom: 8px; }
.cn-explore ul { margin: 0; padding-left: 20px; }
.cn-explore li { margin: 0 0 7px; }
`;

const RECORDS = [["train", "training"], ["test", "test"]];
const KINDS = ["found", "named", "truth"];  // the three panels, left to right
const TITLES = {
  train: {
    found: "clusters, found without labels",
    named: "clusters, named by five records",
    truth: "true labels",
  },
  test: {
    found: "nearest cluster",
    named: "label of the nearest cluster",
    truth: "true labels",
  },
};
const NEUTRAL = C.grid;  // a cluster with no labelled record has no name

const alpha = (hex, a) => {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
};
const inkOn = (hex) => {  // white on dark fills, ink on light ones
  const n = parseInt(hex.slice(1), 16);
  return 0.299 * (n >> 16) + 0.587 * ((n >> 8) & 255) + 0.114 * (n & 255) < 140 ? "#ffffff" : C.ink;
};
const minus = (s) => s.replace("-", "−");

function render({ model, el }) {
  const st = makeStage(el, W);
  const local = elt("style");
  local.textContent = LOCAL_CSS;
  st.stage.append(local);

  // ---- the data, sent once ---------------------------------------------------------
  const classes = model.get("classes"), classColour = model.get("colours");
  const shades = model.get("shades");
  const sets = { train: model.get("train"), test: model.get("test") };
  const D = sets.train.latent[0].length, NC = classes.length;
  const column = (set, d) => Float32Array.from(set.latent, (row) => row[d]);
  const cols = {
    train: Array.from({ length: D }, (_, d) => column(sets.train, d)),
    test: Array.from({ length: D }, (_, d) => column(sets.test, d)),
  };
  const nTrain = sets.train.label.length, nTest = sets.test.label.length;
  const fits = new Map(model.get("fits").map((f) => [f.k, f]));
  const ks = [...fits.keys()].sort((a, b) => a - b);
  const nSamples = model.get("samples");

  // axis range of each number: the middle 99% of train and test, with some air
  const range = Array.from({ length: D }, (_, d) => {
    const v = Float32Array.from([...cols.train[d], ...cols.test[d]]).sort();
    const lo = v[Math.floor(0.005 * (v.length - 1))], hi = v[Math.ceil(0.995 * (v.length - 1))];
    return [lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo)];
  });
  const spread = Array.from({ length: D }, (_, d) => {
    const v = cols.train[d];
    const mean = v.reduce((s, x) => s + x, 0) / v.length;
    return Math.sqrt(v.reduce((s, x) => s + (x - mean) ** 2, 0) / v.length);
  });
  const byspread = [...spread.keys()].sort((a, b) => spread[b] - spread[a]);
  const baseline = Math.max(...classes.map((_, c) => sets.test.label.filter((l) => l === c).length)) / nTest;

  // ---- state -----------------------------------------------------------------------
  let records = "train";  // which records the three panels show
  let dims = [Math.min(1, D - 1), Math.min(3, D - 1)];  // by default z₂ and z₄
  let pinned = null;  // a cluster kept in focus by a click
  let pointed = null;  // the cluster under the pointer (centre, row or vote dot)
  let record = null;  // the record under the pointer: { set, i }
  let cur = null;  // everything for the current k and draw
  const paleOf = (j) => cur.f.pale[j], deepOf = (j) => cur.f.deep[j];
  const compCache = new Map();

  const kNow = () => model.get("k"), sampleNow = () => model.get("sample");

  function composition(k, f) {  // true labels of every cluster's records, per k
    if (!compCache.has(k)) {
      const count = (assign, label) => {
        const out = Array.from({ length: k }, () => new Array(NC).fill(0));
        assign.forEach((j, i) => { out[j][label[i]] += 1; });
        return out;
      };
      compCache.set(k, {
        train: count(f.assign, sets.train.label),
        test: count(f.assign_test, sets.test.label),
      });
    }
    return compCache.get(k);
  }

  function derive() {
    const k = kNow(), f = fits.get(k), s = sampleNow();
    const names = f.names[s], asked = f.asked[s], tone = f.tones[s];
    const votes = Array.from({ length: k }, () => []);  // the labelled records per cluster
    const isAsked = new Set(asked);
    asked.forEach((i) => votes[f.assign[i]].push(i));
    const comp = composition(k, f);
    const testN = new Array(k).fill(0), testRight = new Array(k).fill(0);
    let right = 0;
    f.assign_test.forEach((j, i) => {
      testN[j] += 1;
      if (names[j] === sets.test.label[i]) { testRight[j] += 1; right += 1; }
    });
    const named = new Set();
    votes.forEach((v, j) => { if (v.length) named.add(names[j]); });
    cur = { k, f, names, tone, asked, isAsked, votes, comp, testN, testRight,
      accuracy: right / nTest,
      missing: classes.filter((_, c) => !named.has(c)),
      namedCount: named.size };
  }

  // ---- header bar --------------------------------------------------------------------
  function stepper(lo, hi, get, set, shift = 0) {  // - [ n ] / N +, as Stepper
    const root = elt("div", "cn-stepper");
    const down = elt("button", "mls-pm", "−"), up = elt("button", "mls-pm", "+");
    const input = elt("input", "mls-number");
    input.type = "text"; input.inputMode = "numeric";
    const of = elt("span", "cn-of", `/ ${hi}`);
    const show = () => {
      input.value = String(get() + shift);
      down.disabled = get() + shift <= lo;
      up.disabled = get() + shift >= hi;
    };
    const go = (v) => {
      v = Math.min(Math.max(Math.round(v), lo), hi);
      if (v - shift !== get()) set(v - shift);
      show();
    };
    down.onclick = () => go(get() + shift - 1);
    up.onclick = () => go(get() + shift + 1);
    input.addEventListener("change", () => {
      const v = Number(input.value);
      if (Number.isFinite(v)) go(v); else show();
    });
    input.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") input.blur();
      else if (ev.key === "ArrowUp") { ev.preventDefault(); go(get() + shift + 1); }
      else if (ev.key === "ArrowDown") { ev.preventDefault(); go(get() + shift - 1); }
    });
    root.append(down, input, of, up);
    return { el: root, show };
  }
  const send = (name, v) => { model.set(name, v); model.save_changes(); };

  const kStep = stepper(ks[0], ks[ks.length - 1], kNow, (v) => send("k", v));
  const sStep = stepper(1, nSamples, sampleNow, (v) => send("sample", v), 1);
  const zName = (d) => "z" + sub(d + 1);
  const zOptions = Array.from({ length: D }, (_, d) => [d, zName(d)]);
  const selX = uiSelect(zOptions, dims[0], (v) => pick(0, v));
  const selY = uiSelect(zOptions, dims[1], (v) => pick(1, v));
  function pick(axis, v) {  // never the same number on both axes: the other one swaps
    if (dims[1 - axis] === v) dims[1 - axis] = dims[axis];
    dims[axis] = v;
    selX.set(dims[0]); selY.set(dims[1]);
    draw();
  }
  const seg = uiSegmented(RECORDS, records, (v) => { records = v; record = null; draw(); });
  const sStat = uiStat("LABELS USED"), nStat = uiStat("CLASSES NAMED"), aStat = uiStat("TEST RECORDS RIGHT");
  const stats = elt("div", "cn-stats");
  stats.append(sStat.el, nStat.el, aStat.el);
  const rule = () => elt("div", "mls-rule");

  const bar = elt("div", "mls-bar");
  const kField = uiField(uiLabel("CLUSTERS", "k"), kStep.el);
  kField.title = "the number of clusters k-means looks for";
  const sField = uiField(uiLabel("SAMPLE"), sStep.el);
  sField.title = "which five records of each cluster are looked up";
  const rField = uiField(uiLabel("RECORDS"), seg);
  rField.title = "the records the three panels show";
  bar.append(kField, sField, rule(),
    uiField(uiLabel("HORIZONTAL"), selX), uiField(uiLabel("VERTICAL"), selY), rule(),
    rField, stats);
  st.stage.append(bar);

  const fig = stageCanvas(st, W, H);
  fig.canvas.style.outline = "none";
  fig.canvas.tabIndex = 0;
  const wrap = elt("div");
  wrap.style.position = "relative";
  wrap.append(fig.canvas);
  // what to look for: the playground's own text, set in the free right half under the
  // panels (a figure this tall leaves no room below it on a slide)
  const explore = model.get("explore");
  if (explore.length) {
    const box = elt("div", "cn-explore");
    const head = uiLabel("THINGS TO LOOK FOR");
    head.classList.add("cn-head");
    const list = elt("ul");
    for (const [name, question] of explore) {
      const li = elt("li");
      const b = elt("b");
      b.textContent = `${name}. `;
      li.append(b, document.createTextNode(question));
      list.append(li);
    }
    box.append(head, list);
    wrap.append(box);
  }
  st.stage.append(wrap);
  const ctx = fig.ctx;

  // ---- drawing helpers ----------------------------------------------------------------
  const font = (px, style = "") => { ctx.font = `${style} ${px}px ${FONT}`.trim(); };
  const text = (s, x, y, colour = C.ink, align = "left", px = 11) => {
    font(px); ctx.fillStyle = colour; ctx.textAlign = align; ctx.textBaseline = "alphabetic";
    ctx.fillText(s, x, y);
  };
  // an italic z with its subscript; returns nothing, centred or left-aligned at x
  const zLabel = (d, x, y, align = "left", colour = C.ink, px = 17) => {
    ctx.fillStyle = colour; ctx.textBaseline = "alphabetic";
    ctx.font = `italic ${px}px ${SYM}`;
    const wz = ctx.measureText("z").width;
    ctx.font = `${px * 0.7}px ${FONT}`;
    const wn = ctx.measureText(String(d + 1)).width;
    const x0 = align === "center" ? x - (wz + wn + 1) / 2 : x;
    ctx.textAlign = "left";
    ctx.font = `italic ${px}px ${SYM}`; ctx.fillText("z", x0, y);
    ctx.font = `${px * 0.7}px ${FONT}`; ctx.fillText(String(d + 1), x0 + wz + 1, y + px * 0.25);
  };
  const niceTicks = (a, b, n = 6) => {
    const raw = (b - a) / n, mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw);
    const out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  };
  const disc = (x, y, j, r = 12, a = 1, fill = paleOf(j)) => {  // a numbered cluster disc
    ctx.globalAlpha = a;
    ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fillStyle = fill; ctx.fill();
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 0.9; ctx.stroke();
    text(String(j), x, y + r * 0.4, inkOn(fill), "center", r * 1.25);
    ctx.globalAlpha = 1;
  };
  const dot = (x, y, r, fill) => {
    ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fillStyle = fill; ctx.fill();
  };
  const ring = (x, y, r, colour, lw) => {
    ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.strokeStyle = colour; ctx.lineWidth = lw; ctx.stroke();
  };
  // an emphasised marker: a white halo under a dark ring, so that it stands out from the
  // many small dots around it
  const halo = (x, y, r, lw = 2) => {
    ring(x, y, r, "#ffffff", lw + 3);
    ring(x, y, r, C.ink, lw);
  };

  let hits = [];
  const focus = () => (pointed !== null ? pointed : pinned);
  const assignOf = (set) => (set === "test" ? cur.f.assign_test : cur.f.assign);
  const labelOf = (set, i) => sets[set].label[i];
  // the same two numbers on the same scales in every panel
  const xOf = (v, P) => P.l + (v - range[dims[0]][0]) / (range[dims[0]][1] - range[dims[0]][0]) * (P.r - P.l);
  const yOf = (v) => PB - (v - range[dims[1]][0]) / (range[dims[1]][1] - range[dims[1]][0]) * (PB - PT);

  // the colour a cluster has once it is named: the hue of its class, in its own tone
  const nameColour = (j) => (cur.tone[j] < 0 ? NEUTRAL : shades[cur.names[j]][cur.tone[j]]);
  // colour of a record in a panel
  function fillOf(kind, set, i, j) {
    if (kind === "found") return paleOf(j);
    if (kind === "truth") return classColour[labelOf(set, i)];
    return nameColour(j);
  }

  // ---- left: the three panels ------------------------------------------------------------
  function drawPanel(idx) {
    const P = PANELS[idx], kind = KINDS[idx], [a, b] = dims, set = records;
    const asg = assignOf(set), n = sets[set].label.length;
    const X = cols[set][a], Y = cols[set][b], f = focus();
    text(TITLES[set][kind], P.l, 14, C.ink, "left", 13);

    // open axes: left and bottom spines, inward ticks, labels outside (the vertical
    // scale is read off the first panel only)
    ctx.strokeStyle = C.ink2; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(P.l, PT); ctx.lineTo(P.l, PB); ctx.lineTo(P.r, PB); ctx.stroke();
    for (const v of niceTicks(...range[a], 4)) {
      const x = xOf(v, P);
      if (x < P.l + 1 || x > P.r) continue;
      ctx.beginPath(); ctx.moveTo(x, PB); ctx.lineTo(x, PB - 4); ctx.stroke();
      text(minus(String(+v.toFixed(2))), x, PB + 16, C.ink2, "center", 11.5);
    }
    for (const v of niceTicks(...range[b], 5)) {
      const y = yOf(v);
      if (y > PB - 1 || y < PT) continue;
      ctx.beginPath(); ctx.moveTo(P.l, y); ctx.lineTo(P.l + 4, y); ctx.stroke();
      if (idx === 0) text(minus(String(+v.toFixed(2))), P.l - 7, y + 4, C.ink2, "right", 11.5);
    }
    zLabel(a, (P.l + P.r) / 2, PB + 36, "center");
    if (idx === 0) {
      ctx.save();
      ctx.translate(P.l - 44, (PT + PB) / 2); ctx.rotate(-Math.PI / 2);
      zLabel(b, 0, 0, "center");
      ctx.restore();
    }

    ctx.save();
    ctx.beginPath(); ctx.rect(P.l, PT - 4, P.r - P.l + 4, PB - PT + 8); ctx.clip();
    const r0 = set === "test" ? 2.2 : 1.7;
    // records outside the cluster in focus first and faint, then the cluster itself;
    // dots of one colour are drawn together
    for (const pass of [0, 1]) {
      const groups = new Map();
      for (let i = 0; i < n; i++) {
        const j = asg[i], inFocus = f === null || j === f;
        if ((pass === 1) !== inFocus) continue;
        const fill = f !== null && inFocus && kind === "found" ? deepOf(j) : fillOf(kind, set, i, j);
        if (!groups.has(fill)) groups.set(fill, []);
        groups.get(fill).push(i);
      }
      ctx.globalAlpha = pass === 0 ? 0.1 : kind === "found" ? 0.95 : 0.8;
      const r = pass === 1 && f !== null ? r0 + 0.7 : r0;
      for (const [fill, list] of groups) {
        ctx.fillStyle = fill;
        for (const i of list) {
          ctx.beginPath(); ctx.arc(xOf(X[i], P), yOf(Y[i]), r, 0, 2 * Math.PI); ctx.fill();
        }
      }
    }
    ctx.globalAlpha = 1;
    if (set === "train") {  // the records whose label was looked up
      for (const i of cur.asked) {
        const j = asg[i];
        ctx.globalAlpha = f === null || j === f ? 1 : 0.25;
        dot(xOf(X[i], P), yOf(Y[i]), 4.4, fillOf(kind, set, i, j));
        halo(xOf(X[i], P), yOf(Y[i]), 4.4, 1.6);
      }
    } else if (kind === "named") {  // the test records that get the wrong label
      for (let i = 0; i < n; i++) {
        const j = asg[i];
        if (cur.names[j] === labelOf("test", i)) continue;
        ctx.globalAlpha = f === null || j === f ? 1 : 0.2;
        halo(xOf(X[i], P), yOf(Y[i]), 5.2, 1.6);
      }
    }
    ctx.globalAlpha = 1;
    ctx.restore();

    // the centres, numbered discs, drawn over the records (not in the truth)
    if (kind !== "truth") {
      const centres = cur.f.centres, marked = markedRow();
      for (let j = 0; j < cur.k; j++) {
        const x = xOf(centres[j][a], P), y = yOf(centres[j][b]);
        if (x < P.l - 4 || x > P.r + 4 || y < PT - 4 || y > PB + 4) continue;
        const dim = f !== null && j !== f;
        disc(x, y, j, marked === j ? 12 : 10, dim ? 0.3 : 1, kind === "found" ? paleOf(j) : nameColour(j));
        if (marked === j) halo(x, y, 12.5, 2.2);
        hits.push({ type: "centre", j, cx: x, cy: y, r: 13 });
      }
    }
    // the record under the pointer, in every panel
    if (record && record.set === set) {
      const x = xOf(X[record.i], P), y = yOf(Y[record.i]);
      if (x >= P.l && x <= P.r && y >= PT && y <= PB) {
        dot(x, y, 5, fillOf(kind, set, record.i, asg[record.i]));
        halo(x, y, 9, 2.4);
      }
    }
  }

  // the cluster whose row is marked: the one in focus, else that of the record pointed at
  function markedRow() {
    const f = focus();
    if (f !== null) return f;
    return record ? assignOf(record.set)[record.i] : null;
  }

  // ---- below: the vote ---------------------------------------------------------------------
  // one row per cluster in a single column; the figure grows taller with k
  const pitch = () => TABLE.pitch;
  const rowY = (j) => TABLE.row0 + j * pitch();
  const figHeight = () => TABLE.row0 + cur.k * pitch() + 44;
  function drawTable() {
    const T = TABLE, f = focus(), marked = markedRow();
    text("FIVE RECORDS", T.dots - 10, T.head, C.ink2, "left", 13);
    text("NAME", T.swatch - 8, T.head, C.ink2, "left", 13);
    text("VOTES", T.votes, T.head, C.ink2, "left", 13);
    text("TEST RECORDS", T.testN - 96, T.head, C.ink2, "left", 13);
    for (let j = 0; j < cur.k; j++) {
      const y = rowY(j);
      const a = f !== null && j !== f ? 0.35 : 1;
      disc(T.disc, y, j, 12, a);
      if (marked === j) halo(T.disc, y, 14.5, 2.2);
      const v = cur.votes[j];
      ctx.globalAlpha = a;
      v.forEach((i, m) => {
        const x = T.dots + m * T.dotStep;
        dot(x, y, 8, classColour[sets.train.label[i]]);
        if (record && record.set === "train" && record.i === i) halo(x, y, 11, 2.2);
        hits.push({ type: "dot", j, i, cx: x, cy: y, r: 11 });
      });
      if (v.length) {
        const c = cur.names[j];
        const won = v.filter((i) => sets.train.label[i] === c).length;
        dot(T.swatch, y, 6, nameColour(j));  // its colour in the middle panel
        text(classes[c], T.name, y + 7, classColour[c], "left", 21);
        text(`${won} of ${v.length}`, T.votes, y + 6, C.ink2, "left", 17);
      } else {
        text("no record", T.dots, y + 6, C.ink2, "left", 17);
      }
      // the test records of this cluster: how many, how many get the right label
      const n = cur.testN[j];
      if (n) {
        const share = cur.testRight[j] / n;
        text(String(n), T.testN, y + 6, C.ink2, "right", 17);
        ctx.fillStyle = alpha(classColour[cur.names[j]], 0.55);
        ctx.fillRect(T.bar, y - 5, T.barW * share, 10);
        ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
        ctx.strokeRect(T.bar + 0.5, y - 5 + 0.5, T.barW - 1, 9);
        text(`${Math.round(100 * share)}%`, T.pct, y + 6, C.ink, "left", 17);
      } else {
        text("none", T.bar, y + 6, C.ink2, "left", 17);
      }
      ctx.globalAlpha = 1;
      hits.push({ type: "row", j, x: T.x, y: y - pitch() / 2, w: T.right - T.x, h: pitch() });
    }
    // the classes
    let x = T.x + 4;
    const yL = rowY(cur.k - 1) + 40;
    classes.forEach((c, k) => {
      dot(x + 6, yL - 6, 6, classColour[k]);
      text(c, x + 18, yL, C.ink, "left", 17);
      x += 88;
    });
  }

  // ---- below: a read-out of what is pointed at -------------------------------------------
  function stack(x, y, w, h, counts, total) {  // true labels of a group, one bar
    let x0 = x;
    counts.forEach((n, c) => {
      const wd = w * n / total;
      ctx.fillStyle = classColour[c];
      ctx.fillRect(x0, y, wd, h);
      x0 += wd;
    });
  }
  function countsText(x, y, counts) {  // "TC 61   LP 38": the classes, commonest first
    let x0 = x;  // returns where the text ends
    counts.map((n, c) => [n, c]).filter(([n]) => n > 0).sort((p, q) => q[0] - p[0])
      .forEach(([n, c]) => {
        const s = `${classes[c]} ${n}`;
        text(s, x0, y, classColour[c], "left", 12.5);
        x0 += ctx.measureText(s).width + 14;
      });
    return x0;
  }
  function drawReadout() {
    const y1 = STRIP.y1, y2 = STRIP.y2, x0 = STRIP.x0, x1 = STRIP.x1, f = focus();
    if (record) {
      const { set, i } = record, j = assignOf(set)[i], lab = labelOf(set, i);
      const train = set === "train", name = cur.names[j];
      let x = x0;
      const put = (s, colour = C.ink, px = 14.5) => {
        text(s, x, y1, colour, "left", px);
        x += ctx.measureText(s).width;
      };
      put(`${train ? "training" : "test"} record #${sets[set].index[i]}`);
      put("   true label ", C.ink2); put(classes[lab], classColour[lab]);
      put("   in cluster ", C.ink2); put(String(j));
      put(", named ", C.ink2); put(classes[name], classColour[name]);
      if (train) {
        put(cur.isAsked.has(i) ? "   one of the labelled records" : "   not looked up", C.ink2);
      } else {
        put(name === lab ? "   gets the right label" : "   gets the wrong label",
          name === lab ? C.ink2 : C.ink);
      }
      text("latent  " + sets[set].latent[i].map((v, d) => `${zName(d)} ${minus(v.toFixed(2))}`).join("    "),
        x0, y2, C.ink2, "left", 12.5);
      return;
    }
    if (f === null) {
      text("point at a cluster, a vote or a record to see what it holds; click a cluster to keep it",
        x0, y1, C.ink, "left", 15);
      text(records === "train"
        ? "ringed: the records whose label was looked up, the only labels the clusters use"
        : "ringed in the middle panel: test records that get the wrong label",
      x0, y2, C.ink2, "left", 13.5);
      return;
    }
    const v = cur.votes[f], c = cur.names[f];
    let x = x0;
    disc(x + 10, y1 - 5, f, 10);
    x += 28;
    const put = (s, colour = C.ink, px = 14.5) => {
      text(s, x, y1, colour, "left", px);
      x += ctx.measureText(s).width;
    };
    put(`cluster ${f}`);
    if (v.length) {
      const won = v.filter((i) => sets.train.label[i] === c).length;
      put("   named ", C.ink2); put(classes[c], classColour[c]);
      put(`   with ${won} of ${v.length} votes`, C.ink2);
    }
    const [tr, te] = [cur.comp.train[f], cur.comp.test[f]];
    const nTr = tr.reduce((s, n) => s + n, 0), nTe = te.reduce((s, n) => s + n, 0);
    text("all its training records, by true label", x0, y2 - 14, C.ink2, "left", 11.5);
    if (nTr) { stack(x0, y2 - 8, 150, 9, tr, nTr); countsText(x0 + 160, y2, tr); }
    text("all its test records, by true label", x1, y2 - 14, C.ink2, "left", 11.5);
    if (nTe) {
      stack(x1, y2 - 8, 150, 9, te, nTe);
      const end = countsText(x1 + 160, y2, te);
      text(`${Math.round(100 * cur.testRight[f] / nTe)}% get the right label`, end + 6, y2, C.ink, "left", 12.5);
    } else {
      text("none", x1 + 160, y2, C.ink2, "left", 12.5);
    }
  }

  // ---- draw ------------------------------------------------------------------------------
  function syncHeader() {
    kStep.show(); sStep.show();
    sStat.set(String(cur.asked.length), ` of ${nTrain}`);
    nStat.set(String(cur.namedCount), ` of ${NC}` + (cur.missing.length ? `, no ${cur.missing.join(" ")}` : ""));
    aStat.set(`${Math.round(100 * cur.accuracy)}%`, ` commonest class ${Math.round(100 * baseline)}%`);
  }
  function draw() {
    H = figHeight();
    fig.setHeight(H);
    ctx.clearRect(0, 0, W, H);
    hits = [];
    for (let idx = 0; idx < PANELS.length; idx++) drawPanel(idx);
    drawTable();
    drawReadout();
  }

  function update() {  // k or the draw changed: recompute, then draw
    derive();
    syncHeader();
    draw();
  }

  // ---- pointer ---------------------------------------------------------------------------
  const inCircle = (h, [x, y]) => (x - h.cx) ** 2 + (y - h.cy) ** 2 <= h.r ** 2;
  const hitAt = (p) => hits.find((h) => (h.r !== undefined ? inCircle(h, p) :
    p[0] >= h.x && p[0] < h.x + h.w && p[1] >= h.y && p[1] < h.y + h.h));
  function nearestRecord([x, y]) {  // a record of the panel pointed at within a few px, else null
    if (y < PT || y > PB) return null;
    const P = PANELS.find((q) => x >= q.l && x <= q.r);
    if (!P) return null;
    const set = records, n = sets[set].label.length, X = cols[set][dims[0]], Y = cols[set][dims[1]];
    let best = null, bestD = 8 ** 2;
    for (let i = 0; i < n; i++) {
      const d = (xOf(X[i], P) - x) ** 2 + (yOf(Y[i]) - y) ** 2;
      if (d < bestD) { bestD = d; best = i; }
    }
    return best === null ? null : { set, i: best };
  }
  const same = (a, b) => (a === null && b === null) || (a && b && a.set === b.set && a.i === b.i);

  function target(p) {
    const h = hitAt(p);
    if (h) {
      if (h.type === "dot") return { cluster: h.j, record: { set: "train", i: h.i } };
      return { cluster: h.j, record: null };
    }
    return { cluster: null, record: nearestRecord(p) };
  }
  fig.canvas.addEventListener("pointermove", (ev) => {
    const t = target(fig.xy(ev));
    fig.canvas.style.cursor = t.cluster !== null || t.record ? "pointer" : "default";
    if (t.cluster !== pointed || !same(t.record, record)) {
      pointed = t.cluster; record = t.record; draw();
    }
  });
  fig.canvas.addEventListener("pointerleave", () => {
    if (pointed !== null || record) { pointed = null; record = null; draw(); }
  });
  fig.canvas.addEventListener("click", (ev) => {
    const t = target(fig.xy(ev));
    const j = t.cluster !== null ? t.cluster : t.record ? assignOf(t.record.set)[t.record.i] : null;
    pinned = j === null || j === pinned ? null : j;
    draw();
  });
  fig.canvas.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") { pinned = null; draw(); }
  });

  // ---- from Python -------------------------------------------------------------------------
  model.on("change:k", () => { pinned = null; pointed = null; record = null; update(); });
  model.on("change:sample", () => { record = null; update(); });
  update();
}

export default { render };
