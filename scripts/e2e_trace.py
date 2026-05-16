#!/usr/bin/env python3
"""e2e_trace.py — view the brain's end-to-end request trace.

Each /chat request logs one line to data/e2e_trace.jsonl with a trace_id and
its routing path (brain solo vs escape→engine). This renders that timeline.

Usage:
  python3 e2e_trace.py [--n N] [--file path]
"""
import argparse, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "..", "data", "e2e_trace.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="how many recent requests")
    ap.add_argument("--file", default=DEFAULT)
    args = ap.parse_args()

    try:
        rows = [json.loads(l) for l in open(args.file) if l.strip()]
    except FileNotFoundError:
        print(f"no trace file: {args.file}")
        return

    rows = rows[-args.n:]
    print(f"\n╔══ E2E TRACE — {len(rows)} requests ══╗\n")
    n_solo = n_escape = 0
    solo_ms = escape_ms = 0
    for r in rows:
        escaped = r.get("escaped", False)
        lat = r.get("latency_ms", 0)
        if escaped:
            n_escape += 1
            escape_ms += lat
            route = f"brain → escape → engine:{r.get('engine_port')}"
            mark = "⇡"
        else:
            n_solo += 1
            solo_ms += lat
            route = "brain SOLO (cache/KG)"
            mark = "•"
        bar = "█" * min(40, lat // 100) if lat else ""
        print(f"  {mark} [{r.get('trace_id','?')[:12]}] {lat:>6}ms {bar}")
        print(f"      q: {r.get('prompt','')[:62]}")
        print(f"      → {route}  src={r.get('source','?')}")
        print()

    total = len(rows)
    if total:
        print(f"╔══ SUMMARY ══╗")
        print(f"  SOLO   {n_solo}/{total} ({100*n_solo//total}%)  "
              f"avg {solo_ms//max(n_solo,1)}ms")
        print(f"  ESCAPE {n_escape}/{total} ({100*n_escape//total}%)  "
              f"avg {escape_ms//max(n_escape,1)}ms")
        speedup = (escape_ms // max(n_escape, 1)) / max(solo_ms // max(n_solo, 1), 1)
        print(f"  brain SOLO ~{speedup:.0f}× mais rápido que escape")


if __name__ == "__main__":
    main()
