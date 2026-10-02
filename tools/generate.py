#!/usr/bin/env python3
"""Генерация входных данных: N, затем матрица A (N строк), затем матрица B (N строк).
Кроме текстового файла сохраняется data/input_N.npy (для быстрой верификации).

  python3 tools/generate.py 200 data/input_200.txt --seed 1
"""
import argparse
import os
import numpy as np


def generate(n: int, path: str, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1.0, 1.0, size=(n, n))
    B = rng.uniform(-1.0, 1.0, size=(n, n))
    A, B = np.round(A, 8), np.round(B, 8)  # ровно то, что запишется в текст
    with open(path, "w") as f:
        f.write(f"{n}\n")
        np.savetxt(f, A, fmt="%.8f")
        np.savetxt(f, B, fmt="%.8f")
    np.save(os.path.splitext(path)[0] + ".npy", np.stack([A, B]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("n", type=int)
    p.add_argument("path")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    generate(a.n, a.path, a.seed)
