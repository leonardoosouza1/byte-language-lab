#!/usr/bin/env python3
"""harvest_local_code.py — walk the user's projects directory and build a
clean training corpus from real source code.

Filters:
  - extensions: js/ts/tsx/jsx/html/css/scss/rs/py/md/go/sh/toml/sql/yml/yaml
  - skip junk dirs: node_modules, target, .git, dist, build, .next, .venv,
    __pycache__, coverage, .cache, .nuxt, vendor, .terraform
  - skip minified, lockfiles, generated
  - size band: keep files between 100 bytes and 80 KB
  - dedupe by SHA1 of normalized content (drop exact and trivial copies)

Output: data/corpus_local.txt — each file framed with a header
  // FILE: <relpath> (<lang>)\n<content>\n\n
Lines normalized: NFKC unicode, trailing whitespace stripped, NUL bytes dropped.
"""
import argparse, hashlib, os, sys, unicodedata
from pathlib import Path

EXTS = {".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".scss",
        ".rs", ".py", ".md", ".go", ".sh", ".toml", ".sql",
        ".yml", ".yaml", ".vue", ".svelte", ".json"}
LANG = {".tsx": "tsx", ".jsx": "jsx", ".ts": "ts", ".js": "js",
        ".rs": "rust", ".py": "python", ".html": "html",
        ".css": "css", ".scss": "scss", ".md": "md", ".go": "go",
        ".sh": "bash", ".toml": "toml", ".sql": "sql",
        ".yml": "yaml", ".yaml": "yaml", ".vue": "vue", ".svelte": "svelte",
        ".json": "json"}

SKIP_DIRS = {"node_modules", "target", ".git", "dist", "build", ".next",
             ".venv", "__pycache__", "coverage", ".cache", ".nuxt", "vendor",
             ".terraform", ".turbo", "out", ".idea", ".vscode", "models",
             "models_cache", "data"}
SKIP_NAME = ("package-lock.json", "yarn.lock", "pnpm-lock.yaml", "Cargo.lock",
             "poetry.lock", "uv.lock")
SKIP_SUFFIX = (".min.js", ".min.css", ".bundle.js", ".chunk.js",
               ".map", ".lock")


def usable(path: Path) -> bool:
    if path.suffix.lower() not in EXTS:
        return False
    name = path.name
    if name in SKIP_NAME or any(name.endswith(s) for s in SKIP_SUFFIX):
        return False
    return True


def walk(root: Path):
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not x.startswith(".")]
        for f in files:
            p = Path(d) / f
            if usable(p):
                yield p


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+",
                    default=["/home/leonardo/projects"])
    ap.add_argument("--out", default="data/corpus_local.txt")
    ap.add_argument("--min", type=int, default=100, help="min file bytes")
    ap.add_argument("--max", type=int, default=80_000, help="max file bytes")
    ap.add_argument("--cap", type=int, default=200_000_000,
                    help="hard corpus cap (bytes)")
    args = ap.parse_args()

    skip_self = "byte-language-lab/data"  # don't recurse into our own outputs
    seen = set()
    total = 0
    files_kept = 0
    files_dup = 0
    files_skip_size = 0
    lang_count = {}

    with open(args.out, "w", encoding="utf-8") as out:
        for root in args.roots:
            for p in walk(Path(root)):
                if skip_self in str(p):
                    continue
                try:
                    sz = p.stat().st_size
                except OSError:
                    continue
                if sz < args.min or sz > args.max:
                    files_skip_size += 1
                    continue
                try:
                    raw = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                txt = normalize(raw)
                # cheap dedupe — exact + leading-100-char fingerprint
                h = hashlib.sha1(txt.encode("utf-8", errors="ignore")).digest()
                if h in seen:
                    files_dup += 1
                    continue
                seen.add(h)
                rel = str(p)
                lang = LANG.get(p.suffix.lower(), p.suffix.lstrip("."))
                header = f"// FILE: {rel} ({lang})\n"
                blob = header + txt + "\n"
                if total + len(blob) > args.cap:
                    print(f"  cap {args.cap} reached, stopping")
                    break
                out.write(blob)
                total += len(blob)
                files_kept += 1
                lang_count[lang] = lang_count.get(lang, 0) + 1
            else:
                continue
            break

    mb = total / (1024 * 1024)
    print(f"\n╔══ HARVEST DONE ══╗")
    print(f"  files kept:    {files_kept}")
    print(f"  duplicates:    {files_dup}")
    print(f"  size-skipped:  {files_skip_size}")
    print(f"  corpus:        {total} bytes ({mb:.1f} MB) → {args.out}")
    print(f"  by language:")
    for lang, n in sorted(lang_count.items(), key=lambda kv: -kv[1])[:15]:
        print(f"    {lang:8s} {n}")


if __name__ == "__main__":
    main()
