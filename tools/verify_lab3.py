#!/usr/bin/env python3
"""Отложенная верификация лабы 3 (когда на кластере не было NumPy).

  python3 tools/verify_lab3.py

Берёт data/results_lab3.csv и data/lab3_out/output_N_sequential.txt.gz, привезённые с кластера.
1. Последовательный результат для каждого N сравнивается с A @ B из NumPy (tools/verify.py, допуск 1e-5).
2. Каждый MPI-элемент вычисляется тем же циклом в том же порядке суммирования, поэтому выходной файл MPI
   побитово совпадает с последовательным: если MD5 совпадает, конфигурация получает результат проверки
   последовательной версии. Если MD5 отличается, проверяется собственный сохранённый файл (или ставится FAIL).
Колонки verified и max_rel_err в CSV заполняются для первого запуска каждой конфигурации.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify import verify  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "data", "results_lab3.csv")
OUT = os.path.join(ROOT, "data", "lab3_out")


def find_output(n, tag):
    for ext in (".txt.gz", ".txt"):
        p = os.path.join(OUT, f"output_{n}_{tag}{ext}")
        if os.path.exists(p):
            return p
    return None


def main():
    with open(RES, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    cache = {}  # (n, tag) -> (status, rel)

    def check(n, tag):
        if (n, tag) not in cache:
            path = find_output(n, tag)
            if path is None:
                cache[(n, tag)] = None
            else:
                ok, _, rel = verify(os.path.join(ROOT, "data", f"input_{n}.txt"), path)
                cache[(n, tag)] = ("OK" if ok else "FAIL", f"{rel:.3e}")
                print(f"N={n} {tag}: {cache[(n, tag)][0]} (max_rel_err={rel:.3e}, {os.path.basename(path)})")
        return cache[(n, tag)]

    seq_md5 = {r["n"]: r["md5"] for r in rows if r["strategy"] == "sequential" and r["run"] == "1"}
    changed = 0
    for r in rows:
        if r["run"] != "1" or r["verified"] in ("OK", "FAIL"):
            continue
        n = r["n"]
        if r["strategy"] == "sequential":
            res = check(n, "sequential")
        elif r.get("md5") and r["md5"] == seq_md5.get(n):
            res = check(n, "sequential")
        else:
            res = check(n, f"parallel_mpi_t{r['threads']}") or ("FAIL", "")
            if res[0] == "FAIL":
                print(f"N={n} P={r['threads']}: MD5 не совпадает с последовательной версией")
        if res is None:
            print(f"N={n}: нет файла data/lab3_out/output_{n}_sequential.txt.gz - пропуск")
            continue
        r["verified"], r["max_rel_err"] = res
        changed += 1

    with open(RES, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if r["verified"] == "FAIL"]
    print(f"Обновлено строк: {changed}; FAIL: {len(bad)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
