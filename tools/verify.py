#!/usr/bin/env python3
"""Верификация результата: сравнение C из программы с A @ B из NumPy.

  python3 tools/verify.py data/input_200.txt data/output_200_sequential.txt
Код возврата: 0 - верно, 1 - расхождение, 2 - ошибка аргументов.
Допуск 1e-5 (относительно |A|@|B|): программа выводит числа с 6 значащими цифрами.
"""
import os
import sys
from itertools import islice
import numpy as np


def load_input(path):
    npy = os.path.splitext(path)[0] + ".npy"
    if os.path.exists(npy):
        A, B = np.load(npy)
        return A, B
    with open(path) as f:
        n = int(f.readline())
        m = np.loadtxt(f, dtype=np.float64, ndmin=2)
    assert m.shape == (2 * n, n), f"{path}: ожидалось {2*n}x{n}, получено {m.shape}"
    return m[:n], m[n:]


def load_output(path):
    with open(path) as f:
        n = int(f.readline())
        c = np.loadtxt(islice(f, n), dtype=np.float64, ndmin=2)  # дальше - строки метаданных
    assert c.shape == (n, n), f"{path}: ожидалось {n}x{n}, получено {c.shape}"
    return c


def verify(input_path, output_path, rtol=1e-5):
    """Возвращает (ok, max_abs_err, max_rel_err)."""
    A, B = load_input(input_path)
    C = load_output(output_path)
    err = np.abs(C - A @ B)  # эталон: NumPy
    scale = np.abs(A) @ np.abs(B)
    max_rel = float((err / np.maximum(scale, 1e-300)).max())
    return max_rel <= rtol, float(err.max()), max_rel


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    ok, mabs, mrel = verify(*sys.argv[1:3])
    print(f"max_abs_err={mabs:.3e} max_rel_err={mrel:.3e} status={'OK' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
