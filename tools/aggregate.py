#!/usr/bin/env python3
"""Агрегация data/results.csv (каждый запуск - строка) по (N, стратегия, потоки):
среднее / минимум / σ времени, GFLOPS, ускорение относительно sequential.

  python3 tools/aggregate.py          # печатает таблицу в формате Markdown
"""
import csv
import os
import statistics
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "data", "results.csv")


def aggregate(path=RESULTS):
    groups = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            groups[(int(r["n"]), r["strategy"], int(r["threads"]))].append(r)

    rows = []
    for (n, strategy, threads), rs in groups.items():
        t = [float(r["time_seconds"]) for r in rs]
        checked = [r for r in rs if r["verified"] in ("OK", "FAIL")]
        rows.append({
            "n": n, "strategy": strategy, "threads": threads, "runs": len(t),
            "mean": statistics.mean(t), "min": min(t), "std": statistics.pstdev(t),
            "gflops": 2.0 * n ** 3 / statistics.mean(t) / 1e9,
            "verified": "FAIL" if any(r["verified"] == "FAIL" for r in checked) else ("OK" if checked else "NA"),
            "max_rel_err": max((float(r["max_rel_err"]) for r in checked if r["max_rel_err"]), default=0.0),
        })
    base = {r["n"]: r["mean"] for r in rows if r["threads"] == 0}
    for r in rows:
        r["speedup"] = base[r["n"]] / r["mean"] if r["n"] in base else None
    rows.sort(key=lambda r: (r["threads"], r["n"]))
    return rows


def markdown_table(rows):
    head = ("| N | Память, МБ | Операций | Стратегия | Потоки | Запусков | t средн., с | t мин., с | σ, с | "
            "Ускорение | GFLOPS | Верификация | Макс. отн. погрешность |\n" + "|---" * 13 + "|")
    lines = [head]
    for r in rows:
        n = r["n"]
        sp = "-" if r["speedup"] is None else f"{r['speedup']:.2f}"
        lines.append(f"| {n} | {3 * n * n * 8 / 1048576:.1f} | {2 * n ** 3:.3e} | {r['strategy']} | {r['threads']} | "
                     f"{r['runs']} | {r['mean']:.4f} | {r['min']:.4f} | {r['std']:.4f} | {sp} | {r['gflops']:.3f} | "
                     f"{r['verified']} | {r['max_rel_err']:.1e} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(markdown_table(aggregate()))
