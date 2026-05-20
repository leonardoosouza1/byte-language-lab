#!/usr/bin/env python3
"""prep_code_corpus.py — turn the ===PAIR=== code corpus into a flat training
text for train_smg_hsa (next-byte LM over a byte corpus).

Each pair becomes:  <request>\n<code>\n\n
so the SMG learns to continue a request prompt with code bytes.

Usage:  python3 prep_code_corpus.py [--in data/code_corpus.txt] [--out data/code_train.txt]
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", default="data/code_corpus.txt")
ap.add_argument("--out", default="data/code_train.txt")
ap.add_argument("--repeat", type=int, default=3,
                help="repeat the corpus N times (more gradient signal per epoch)")
args = ap.parse_args()

text = open(args.inp, encoding="utf-8", errors="replace").read()
pairs = []
for chunk in text.split("===PAIR==="):
    chunk = chunk.strip()
    if not chunk or "---CODE---" not in chunk:
        continue
    req, code = chunk.split("---CODE---", 1)
    req, code = req.strip(), code.strip()
    if len(req) < 5 or len(code) < 10:
        continue
    pairs.append(f"{req}\n{code}\n\n")

block = "".join(pairs)
with open(args.out, "w", encoding="utf-8") as f:
    for _ in range(args.repeat):
        f.write(block)

print(f"pairs: {len(pairs)}   corpus: {len(block)} bytes "
      f"×{args.repeat} = {len(block)*args.repeat} bytes → {args.out}")
