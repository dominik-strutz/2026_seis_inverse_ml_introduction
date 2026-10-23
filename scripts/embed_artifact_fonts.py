# /// script
# requires-python = ">=3.11"
# dependencies = ["fonttools>=4.50", "brotli>=1.1"]
# ///
"""Embed Latin Modern Math and LM Roman (subset woff2, base64) in the HTML artifacts.

Artifacts are served through `mo.iframe`, so they cannot link to font files and must
work offline; the maths font therefore travels inside each file. Re-run after editing
an artifact (the embedded block is replaced, not duplicated):

    uv run scripts/embed_artifact_fonts.py [src/ml_seis/artifacts/*.html]

Needs Latin Modern installed locally (lm and lm-math font packages), or set LM_DIR
to the folder holding `lm/` and `lm-math/`.
"""

import base64
import io
import os
import re
import sys
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
LM_DIR = Path(os.environ.get("LM_DIR", "/usr/share/fonts"))
BEGIN, END = "/* ml_seis:fonts begin */", "/* ml_seis:fonts end */"

# Symbols (classes m and sym) use Latin Modern Math's own math-italic glyphs, as in
# LaTeX: ASCII and Greek letters in those elements are mapped to U+1D434 and friends,
# for static markup and for text the page writes later.
SYMBOL_CSS = """.m, .sym, svg .m, .dg .sym {
  font-family: "Latin Modern Math", "Latin Modern Roman", serif !important;
  font-style: normal !important; }"""
SYMBOL_JS = """(() => {
  const it = c => {
    const k = c.codePointAt(0);
    if (k >= 65 && k <= 90) return String.fromCodePoint(0x1D434 + k - 65);
    if (k >= 97 && k <= 122)
      return k === 104 ? "\\u210E" : String.fromCodePoint(0x1D44E + k - 97);
    if (k >= 0x391 && k <= 0x3A9) return String.fromCodePoint(0x1D6E2 + k - 0x391);
    if (k >= 0x3B1 && k <= 0x3C9) return String.fromCodePoint(0x1D6FC + k - 0x3B1);
    return c;
  };
  const fix = n => {
    const p = n.parentElement;
    if (!p || !p.closest(".m, .sym")) return;
    const u = Array.from(n.nodeValue, it).join("");
    if (u !== n.nodeValue) n.nodeValue = u;
  };
  const walk = root => {
    if (root.nodeType === 3) return fix(root);
    const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let n = w.nextNode(); n; n = w.nextNode()) fix(n);
  };
  new MutationObserver(ms => ms.forEach(m => {
    if (m.type === "characterData") fix(m.target); else m.addedNodes.forEach(walk);
  })).observe(document, { childList: true, subtree: true, characterData: true });
  document.addEventListener("DOMContentLoaded", () => walk(document.body));
})();"""

# family, file, CSS weight, CSS style
FACES = [
    ("Latin Modern Math", "lm-math/latinmodern-math.otf", 400, "normal"),
    ("Latin Modern Roman", "lm/lmroman10-regular.otf", 400, "normal"),
    ("Latin Modern Roman", "lm/lmroman10-italic.otf", 400, "italic"),
    ("Latin Modern Roman", "lm/lmroman10-bolditalic.otf", 700, "italic"),
]
# Chrome maps single-letter <mi> to the Mathematical Alphanumeric block, which never
# appears in the source: keep italic Latin and Greek, script N, bold digits for maths.
BASIC = set(range(0x20, 0x7F)) | {0xB7, 0xD7, 0x2212}
MATH_EXTRA = (
    set(range(0x1D434, 0x1D468))  # italic Latin
    | set(range(0x1D6E2, 0x1D71C))  # italic Greek
    | {0x1D4A9}  # script N
    | set(range(0x1D7CE, 0x1D7D8))  # bold digits
    | set(range(0x370, 0x400))
    | set(range(0x2200, 0x2300))  # operators
    | set(range(0x2070, 0x20A0))  # super- and subscripts
)


def subset_woff2(path, codepoints, features):
    opts = subset.Options()
    opts.flavor = "woff2"
    opts.layout_features = features
    opts.notdef_outline = True
    font = TTFont(path)
    sub = subset.Subsetter(opts)
    sub.populate(unicodes=codepoints)
    sub.subset(font)
    buf = io.BytesIO()
    font.flavor = "woff2"
    font.save(buf)
    return buf.getvalue()


def face_css(family, data, weight, style):
    b64 = base64.b64encode(data).decode()
    return (
        f'@font-face {{ font-family: "{family}"; font-weight: {weight}; '
        f"font-style: {style}; "
        f'src: url(data:font/woff2;base64,{b64}) format("woff2"); }}'
    )


def process(html_path):
    text = html_path.read_text(encoding="utf-8")
    text = re.sub(r"\s*<style id=\"ml-seis-fonts\">.*?</style>", "", text, flags=re.S)
    text = re.sub(
        r"\s*<script id=\"ml-seis-symbols\">.*?</script>", "", text, flags=re.S
    )
    used = {ord(c) for c in text} | BASIC
    css = []
    for family, name, weight, style in FACES:
        extra = MATH_EXTRA if "math" in name.lower() else set()
        data = subset_woff2(
            LM_DIR / name, sorted(used | extra), ["*"] if extra else ["kern"]
        )
        css.append(face_css(family, data, weight, style))
    block = (
        f'\n<style id="ml-seis-fonts">\n{BEGIN}\n'
        + "\n".join(css)
        + f"\n{SYMBOL_CSS}\n{END}\n</style>"
        + f'\n<script id="ml-seis-symbols">\n{SYMBOL_JS}\n</script>'
    )
    text = text.replace("</head>", block + "\n</head>", 1)
    html_path.write_text(text, encoding="utf-8")
    return len(block)


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] or sorted(
        (ROOT / "src" / "ml_seis" / "artifacts").glob("*.html")
    )
    for p in paths:
        print(f"{p.name}: +{process(p) / 1024:.0f} KB")
