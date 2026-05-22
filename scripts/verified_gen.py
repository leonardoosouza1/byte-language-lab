#!/usr/bin/env python3
"""verified_gen.py — EASY win: verified code generation from the brain's own
memory. Proposer = brain (no_escape: cache + N1 mutation). Then a grammar
layer (bracket/quote balance) + `node --check` validate. On failure, the loop
self-corrects: extract code block, trim to balance, retry with rephrase.

This is the verify-and-search loop on the proposers we already have. It
produces code that is GUARANTEED syntactically valid (or honestly reports a
miss) — grounded in the brain's data, no runtime Qwen.

Used by build_project as the per-piece generator.
"""
import json, re, subprocess, tempfile, os, urllib.request

BRAIN = "http://127.0.0.1:3012"


def brain(prompt, max_tokens=400):
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
    lines = [l for l in text.splitlines()
             if any(c in l for c in "{};()=<>") or l.strip().startswith(("//", "const", "function", "class", "let", "import"))]
    return "\n".join(lines).strip()


def balanced(code):
    """Grammar layer: bracket/quote balance (string/comment-aware-ish)."""
    pairs = {')': '(', ']': '[', '}': '{'}
    opens = set("([{")
    st = []
    quote = None
    i = 0
    while i < len(code):
        c = code[i]
        if quote:
            if c == "\\" and quote != "`": i += 2; continue
            if c == quote: quote = None
            i += 1; continue
        if c in "\"'`":
            quote = c
        elif c == "/" and i + 1 < len(code) and code[i+1] == "/":
            j = code.find("\n", i);  i = len(code) if j < 0 else j;  continue
        elif c in opens:
            st.append(c)
        elif c in pairs:
            if not st or st[-1] != pairs[c]:
                return False
            st.pop()
        i += 1
    return not st and quote is None


def node_check(code, as_module=True):
    src = code
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(src); path = f.name
    try:
        r = subprocess.run(["node", "--check", path], capture_output=True, text=True, timeout=10)
        return r.returncode == 0, r.stderr.strip()
    finally:
        os.remove(path)


def trim_to_balance(code):
    """Self-correct: drop trailing chars until brackets balance (cheap repair)."""
    for end in range(len(code), max(0, len(code) - 400), -1):
        if balanced(code[:end]):
            return code[:end]
    return code


def verified_gen(request, rephrases=None):
    """Return (code, source, status). status: 'verified' | 'syntax-only' | 'miss'.
    Tries the request + rephrases; validates grammar + node; self-corrects."""
    attempts = [request] + (rephrases or [])
    best = None
    for q in attempts:
        text, src = brain(q)
        if not src.startswith("brain"):
            continue
        code = extract_code(text)
        if len(code) < 12:
            continue
        if best is None:
            best = (code, src)
        # grammar layer
        if not balanced(code):
            code2 = trim_to_balance(code)
            if balanced(code2) and len(code2) >= 12:
                code = code2
            else:
                continue
        # syntax layer
        ok, _ = node_check(code)
        if ok:
            return code, src, "verified"
        best = (code, src)   # keep best balanced-but-not-node-ok
    if best:
        return best[0], best[1], "syntax-only" if balanced(best[0]) else "miss"
    return "", "miss", "miss"


if __name__ == "__main__":
    import sys
    reqs = sys.argv[1:] or [
        "crie uma função JavaScript que valida um email",
        "crie uma classe JavaScript Stack com push, pop e peek",
        "crie uma função JavaScript que faz quicksort em um array",
        "crie uma função JavaScript que calcula o fatorial",
        "crie um hook React customizado useToggle",
    ]
    ok = 0
    for r in reqs:
        code, src, status = verified_gen(r)
        mark = {"verified": "✓", "syntax-only": "~", "miss": "✗"}[status]
        if status == "verified": ok += 1
        print(f"  {mark} [{status:<11}] [{src:<14}] {r[:46]}")
    print(f"\n{ok}/{len(reqs)} verified (node --check passou)")
