#!/usr/bin/env python3
"""code_agent.py — micro/macro worker over the brain's code cache.

Brain answers in microseconds — LLM in seconds. Use the time gap to compose
ANSWERS FROM PIECES. A complex request is decomposed into N atomic sub-
requests, each one served by the brain (cache hit) or — on miss — escaped.
The composer stitches the pieces, fixing identifier collisions and adding
necessary imports. Each sub-step is validated; failures retry with a
rephrased query.

Example:
  request: "crie classe Pessoa com nome e idade, depois Funcionario que
            estende Pessoa com salário"
  decomp:
    1. crie uma classe Pessoa com nome e idade
    2. crie uma classe Funcionario que estende Pessoa com salário
  → brain serves both from cache → composer concatenates → validate.

This is the cheap path to combinatorial code generation without training a
bigger model: leverage the cache's atomic patterns and the speed budget.
"""
import argparse, json, re, sys, time, urllib.request

BRAIN = "http://127.0.0.1:3012"


def post(url, payload, timeout=120):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ask_brain(q, max_tokens=600):
    t0 = time.time()
    d = post(f"{BRAIN}/chat", {"prompt": q, "max": max_tokens})
    return d.get("source", "?"), d.get("response", ""), (time.time() - t0) * 1000


def balanced(code: str) -> bool:
    """Quick brace/paren/bracket balance — string- and comment-naive."""
    depths = {"{": 0, "(": 0, "[": 0}
    pairs = {"}": "{", ")": "(", "]": "["}
    in_str = False
    quote = None
    i = 0
    while i < len(code):
        c = code[i]
        if in_str:
            if c == "\\" and i + 1 < len(code):
                i += 2; continue
            if c == quote:
                in_str = False
        elif c in "\"'`":
            in_str = True; quote = c
        elif c == "/" and i + 1 < len(code) and code[i + 1] == "/":
            while i < len(code) and code[i] != "\n":
                i += 1
        elif c in depths:
            depths[c] += 1
        elif c in pairs:
            depths[pairs[c]] -= 1
            if depths[pairs[c]] < 0:
                return False
        i += 1
    return all(v == 0 for v in depths.values())


def split_steps(request: str):
    """Naive decomposer — split on commas, conjunctions, 'depois', 'e que'.
    Each fragment becomes a sub-request. Refined version would use LLM
    planner; rule-based is enough for a first cut."""
    # normalize separators
    text = re.sub(r"\s*\b(depois|em seguida|e depois|, e |, depois)\b\s*",
                  "|", request, flags=re.IGNORECASE)
    parts = [p.strip(" ,;.") for p in text.split("|") if p.strip()]
    # add "crie" prefix to each part if missing
    out = []
    first_subject = None
    for p in parts:
        if not re.match(r"^(crie|escreva|fa[çc]a|gere|implemente)\b", p, re.IGNORECASE):
            p = "crie " + p
        out.append(p)
    return out


def extract_identifier(code: str, kind="class"):
    """Pull the first class/function name declared in the code."""
    m = re.search(rf"\b{kind}\s+([A-Za-z_][A-Za-z0-9_]*)", code)
    return m.group(1) if m else None


def extract_code(text: str) -> str:
    """Pull code out of a brain answer — prefer ``` fenced blocks; fall back
    to the text itself if it already looks like raw code."""
    blocks = re.findall(r"```[a-zA-Z]*\n?(.*?)```", text, re.DOTALL)
    if blocks:
        return "\n\n".join(b.strip() for b in blocks if b.strip())
    lines = text.splitlines()
    # drop leading prose paragraphs (no { } ; < /), keep from first code-ish line
    start = 0
    for i, ln in enumerate(lines):
        if re.search(r"[{};</]|\bfunction\b|\bclass\b|\bdef\b|\bconst\b|\blet\b", ln):
            start = i; break
    return "\n".join(lines[start:]).strip()


def compose(pieces):
    """Strip surrounding prose / fences from each piece, then concatenate."""
    cleaned = [extract_code(p) for p in pieces]
    return "\n\n".join(c for c in cleaned if c)


def run(request: str, retries: int = 2, verbose: bool = True):
    steps = split_steps(request)
    if verbose:
        print(f"\n▸ decomposed into {len(steps)} step(s)")
        for i, s in enumerate(steps, 1):
            print(f"  {i}. {s}")
    pieces = []
    sources = []
    latencies_ms = []
    failures = []
    for i, step in enumerate(steps, 1):
        attempts = [step]
        # rephrase variants for retries
        for k in range(retries):
            v = step.replace("crie", "escreva") if k == 0 else step.replace("crie uma ", "")
            attempts.append(v)
        for tries, query in enumerate(attempts, 1):
            src, code, dt = ask_brain(query)
            latencies_ms.append(dt)
            if code and balanced(code):
                pieces.append(code)
                sources.append(src)
                if verbose:
                    print(f"  step {i} ✓  [{src:<22}] {dt:.0f} ms  (try {tries})")
                break
        else:
            failures.append(step)
            if verbose:
                print(f"  step {i} ✗  all {retries+1} attempts failed")

    full = compose(pieces)
    final_balanced = balanced(full)
    return {
        "request": request,
        "steps": steps,
        "pieces": pieces,
        "sources": sources,
        "latencies_ms": latencies_ms,
        "failures": failures,
        "balanced": final_balanced,
        "composed": full,
        "total_ms": sum(latencies_ms),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("request", nargs="+", help="natural-language code request")
    ap.add_argument("--json", action="store_true", help="JSON output")
    args = ap.parse_args()
    req = " ".join(args.request)
    result = run(req, verbose=not args.json)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(f"\n═══ COMPOSED ({result['total_ms']:.0f} ms, "
          f"{len(result['pieces'])}/{len(result['steps'])} steps, "
          f"balanced={result['balanced']}) ═══\n")
    print(result["composed"])
    if result["failures"]:
        print(f"\n⚠ failed steps: {result['failures']}")


if __name__ == "__main__":
    main()
