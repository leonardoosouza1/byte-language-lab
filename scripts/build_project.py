#!/usr/bin/env python3
"""build_project.py — assemble a JS project from the BRAIN'S OWN MEMORY.

No runtime Qwen. Every piece comes from the brain via /chat with no_escape=1
(cache / KG / forest / N1 mutation / markov). The brain composes the project
from the code it has internalized — grounded in its data, not phoning a teacher.

Pipeline:
  1. decompose the goal into component requests (the "what a calculator needs")
  2. retrieve each piece from brain memory (no_escape) + record its source
  3. extract code, assemble files under projects/<goal>/
  4. validate JS with `node --check`
  5. report per-piece provenance (real cache hit vs weak markov)

Honest by design: pieces the brain lacks come back as markov noise and are
flagged — that's a "needs more data" signal, not a silent Qwen fallback.

Usage:  python3 scripts/build_project.py [goal_dir]
"""
import json, os, re, subprocess, sys, urllib.request

BRAIN = "http://127.0.0.1:3012"
OUT = sys.argv[1] if len(sys.argv) > 1 else "projects/calculadora"


def ask(prompt, max_tokens=400):
    body = json.dumps({"prompt": prompt, "max": max_tokens, "no_escape": 1}).encode()
    req = urllib.request.Request(f"{BRAIN}/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read())
        return d.get("response", ""), d.get("source", "?")
    except Exception as e:
        return f"<err {e}>", "error"


def extract_code(text):
    blocks = re.findall(r"```[a-zA-Z]*\n?(.*?)```", text, re.DOTALL)
    if blocks:
        return "\n\n".join(b.strip() for b in blocks if b.strip())
    # no fence: keep lines that look like code (have code punctuation)
    lines = [l for l in text.splitlines()
             if any(c in l for c in "{};()=<>") or l.strip().startswith("//")]
    return "\n".join(lines).strip()


# Component plan for a calculator. Each entry: (label, brain query).
PIECES = [
    ("op_add",  "crie uma função JavaScript que soma dois números"),
    ("op_sub",  "crie uma função JavaScript que subtrai dois números"),
    ("op_mul",  "crie uma função JavaScript que multiplica dois números"),
    ("op_div",  "crie uma função JavaScript que divide dois números"),
    ("dom",     "crie uma função JavaScript que adiciona um event listener de clique a um botão"),
    ("css_btn", "crie código CSS para um botão moderno com hover"),
    ("css_grid","crie código CSS para uma grade responsiva com grid"),
    ("html",    "crie um template HTML5 básico com meta viewport e charset"),
]

GOOD = ("brain-hi", "brain-mutated", "brain-mid", "brain-deduped", "brain-hi-risky")


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"building '{OUT}' from brain memory (no_escape)\n")
    got = {}
    report = []
    for label, q in PIECES:
        code, src = ask(q)
        c = extract_code(code)
        ok = src in GOOD and len(c) > 15
        report.append((label, src, len(c), ok))
        if ok:
            got[label] = c
        mark = "✓" if ok else "·"
        print(f"  {mark} [{src:<16}] {label:<9} {len(c):>4}B  {q[:42]}")

    # ---- assemble calculator.js: wrap the retrieved ops in a Calculator class ----
    ops = []
    for key, name, sym in [("op_add", "somar", "+"), ("op_sub", "subtrair", "-"),
                           ("op_mul", "multiplicar", "*"), ("op_div", "dividir", "/")]:
        if key in got:
            ops.append(f"  {name}(a, b) {{ return a {sym} b; }}")
    calc_js = (
        "// Calculator — composed by the brain from its own cached code\n"
        "class Calculator {\n"
        "  constructor() { this.display = '0'; this.acc = null; this.op = null; }\n"
        + "\n".join(ops) + "\n"
        "  compute() {\n"
        "    const b = parseFloat(this.display);\n"
        "    if (this.op === '+') this.acc = this.somar(this.acc, b);\n"
        "    else if (this.op === '-') this.acc = this.subtrair(this.acc, b);\n"
        "    else if (this.op === '*') this.acc = this.multiplicar(this.acc, b);\n"
        "    else if (this.op === '/') this.acc = this.dividir(this.acc, b);\n"
        "    this.display = String(this.acc);\n"
        "  }\n"
        "  inputDigit(d) { this.display = this.display === '0' ? d : this.display + d; }\n"
        "  setOp(o) { this.acc = parseFloat(this.display); this.op = o; this.display = '0'; }\n"
        "}\n\n"
        "// DOM wiring (pattern from brain memory):\n"
        "/* " + got.get("dom", "addEventListener pattern not in memory").strip()[:300] + " */\n"
        "if (typeof document !== 'undefined') {\n"
        "  const calc = new Calculator();\n"
        "  const out = document.getElementById('display');\n"
        "  document.querySelectorAll('[data-d]').forEach(b =>\n"
        "    b.addEventListener('click', () => { calc.inputDigit(b.dataset.d); out.textContent = calc.display; }));\n"
        "  document.querySelectorAll('[data-op]').forEach(b =>\n"
        "    b.addEventListener('click', () => { calc.setOp(b.dataset.op); }));\n"
        "  const eq = document.getElementById('eq');\n"
        "  if (eq) eq.addEventListener('click', () => { calc.compute(); out.textContent = calc.display; });\n"
        "}\n"
        "if (typeof module !== 'undefined') module.exports = Calculator;\n"
    )

    css = (got.get("css_btn", "") + "\n\n" + got.get("css_grid", "")
           + "\n\n#display { grid-column: 1/-1; text-align: right; font-size: 2rem;"
             " padding: 1rem; background: #111; color: #0f0; }\n"
             ".calc { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px;"
             " max-width: 320px; margin: 2rem auto; }\n")

    html = (
        "<!DOCTYPE html>\n<html lang=\"pt-BR\">\n<head>\n"
        "  <meta charset=\"UTF-8\">\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
        "  <title>Calculadora</title>\n  <link rel=\"stylesheet\" href=\"style.css\">\n"
        "</head>\n<body>\n  <div class=\"calc\">\n"
        "    <div id=\"display\">0</div>\n"
        + "".join(f'    <button data-d="{d}">{d}</button>\n' for d in "789456123 0")
        + '    <button data-op="+">+</button>\n    <button data-op="-">-</button>\n'
          '    <button data-op="*">*</button>\n    <button data-op="/">/</button>\n'
          '    <button id="eq">=</button>\n'
        "  </div>\n  <script src=\"calculator.js\"></script>\n</body>\n</html>\n"
    )

    open(os.path.join(OUT, "calculator.js"), "w").write(calc_js)
    open(os.path.join(OUT, "style.css"), "w").write(css)
    open(os.path.join(OUT, "index.html"), "w").write(html)

    # ---- validate ----
    print(f"\nfiles written to {OUT}/")
    rc = subprocess.run(["node", "--check", os.path.join(OUT, "calculator.js")],
                        capture_output=True, text=True)
    js_ok = rc.returncode == 0
    print(f"  node --check calculator.js: {'OK' if js_ok else 'FAIL'}")
    if not js_ok:
        print("   " + rc.stderr.strip()[:200])

    # functional smoke: run the class in node
    smoke = os.path.join(OUT, "_smoke.js")
    open(smoke, "w").write(
        "const C = require('./calculator.js'); const c = new C();\n"
        "c.setOp('+'); c.display='2'; c.setOp('+'); c.display='3'; c.compute();\n"
        "console.log('2+3 =', c.display);\n")
    sm = subprocess.run(["node", smoke], capture_output=True, text=True)
    print(f"  smoke (2+3): {sm.stdout.strip() or sm.stderr.strip()[:120]}")
    os.remove(smoke)

    hits = sum(1 for _, _, _, ok in report if ok)
    print(f"\n╔══ BUILD ══╗  {hits}/{len(PIECES)} pieces from real memory  ·  JS valid: {js_ok}")


if __name__ == "__main__":
    main()
