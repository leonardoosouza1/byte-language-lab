#!/usr/bin/env python3
"""Validation loop — repete /benchmark até score >= threshold ou max_rounds.

Cada round:
  1. Run benchmark com auto_fix=1 (corrige wrongs via Qwen → WAL)
  2. Aguarda daemon ingest (25s)
  3. Re-run benchmark com auto_fix=0 (mede)
  4. Se score < threshold, repete

Usage:
  python3 validation_loop.py [target_pct=80] [max_rounds=5]
"""
import json, sys, time, urllib.request

BRAIN = "http://127.0.0.1:3012"

def benchmark(auto_fix=True):
    body = json.dumps({"auto_fix": 1 if auto_fix else 0}).encode("utf-8")
    req = urllib.request.Request(f"{BRAIN}/benchmark", data=body,
        headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=300).read()
    return json.loads(r)

def main():
    target = float(sys.argv[1]) if len(sys.argv) > 1 else 80.0
    max_rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    print(f"═══ VALIDATION LOOP — target {target}%, max {max_rounds} rounds ═══")
    print()

    history = []
    for round_n in range(1, max_rounds + 1):
        print(f"┌── Round {round_n}/{max_rounds} ──┐")
        t0 = time.time()
        # 1. Run benchmark with auto_fix
        result = benchmark(auto_fix=True)
        pct = result["pct"]
        fixed = result["fixed_via_qwen"]
        print(f"  Score: {result['hits']}/{result['total']} ({pct:.0f}%) · fixed: {fixed}")
        for x in result['details']:
            mark = '✓' if x['ok'] else '✗'
            print(f"  {mark} [{x['hits']}/{x['need']}] {x['q']}")
        history.append({"round": round_n, "pct": pct, "fixed": fixed})

        if pct >= target:
            print(f"\n🎉 Target {target}% atingido em round {round_n}!")
            break

        if fixed > 0:
            print(f"  ⏳ Aguardando ingest 25s ({fixed} novas WAL records)...")
            time.sleep(25)

        # Quick check
        check = benchmark(auto_fix=False)
        print(f"  Post-ingest check: {check['pct']:.0f}%")
        print()

    print()
    print("═══ HISTÓRICO ═══")
    for h in history:
        bar = '█' * int(h['pct'] / 5)
        print(f"  Round {h['round']}: {h['pct']:>5.1f}% · fixed {h['fixed']:>3} {bar}")

    if history and history[-1]['pct'] >= target:
        print(f"\n✓ Convergiu: {history[-1]['pct']:.1f}% ≥ {target}%")
    else:
        print(f"\n⚠ Não convergiu em {max_rounds} rounds (último: {history[-1]['pct'] if history else 0:.1f}%)")

if __name__ == "__main__":
    main()
