#!/usr/bin/env python3
"""Графики и отчёт лабы 3: data/results_lab3.csv -> report/lab3/figures/*.png и report/lab3/README.md.

  python3 tools/plot_lab3.py
  python3 tools/plot_lab3.py --platform "Суперкомпьютер «Сергей Королёв», 2 узла по 16 ядер" --cores-total 32
  python3 tools/plot_lab3.py --skip-report
Имя и группа берутся из tools/plot.py (STUDENT, GROUP).
"""
import argparse
import os
import platform
import subprocess
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aggregate import aggregate
from plot import STUDENT, GROUP, device_name

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "data", "results_lab3.csv")
REPORT = os.path.join(ROOT, "report", "lab3")
FIG = os.path.join(REPORT, "figures")


def mpi_version():
    for cmd in (["mpirun", "--version"], ["mpiexec", "--version"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip().splitlines()
            if out:
                return out[0]
        except Exception:
            pass
    return "MPI (версию уточните)"


def compiler_version():
    for cmd in (["mpicxx", "--version"], ["c++", "--version"]):
        try:
            return subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines()[0]
        except Exception:
            pass
    return "unknown"


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name), dpi=130)
    plt.close()


def make_plots(rows, cores_total):
    os.makedirs(FIG, exist_ok=True)
    seq = sorted((r for r in rows if r["threads"] == 0), key=lambda r: r["n"])
    mpi = [r for r in rows if r["threads"] > 0]
    procs = sorted({r["threads"] for r in mpi})
    made = []

    plt.figure(figsize=(7, 4.5))
    if seq:
        plt.plot([r["n"] for r in seq], [r["mean"] for r in seq], "o-", label="sequential")
    for p in procs:
        rs = sorted((r for r in mpi if r["threads"] == p), key=lambda r: r["n"])
        plt.plot([r["n"] for r in rs], [r["mean"] for r in rs], "o-", label=f"MPI, {p} проц.")
    plt.xlabel("Размер матрицы N"); plt.ylabel("Время, с"); plt.grid(True); plt.legend(fontsize=8)
    plt.title("Время умножения от размера матрицы")
    save("time_vs_size.png"); made.append("time_vs_size.png")

    sp = [r for r in mpi if r["speedup"] is not None]
    if sp:
        pmax = max(r["threads"] for r in sp)
        for key, ylabel, fname, title in (
                ("speedup", "Ускорение S = T1 / Tp", "speedup_vs_procs.png", "Ускорение от числа процессов"),
                ("efficiency", "Эффективность E = S / p", "efficiency_vs_procs.png", "Эффективность от числа процессов")):
            plt.figure(figsize=(7, 4.5))
            for n in sorted({r["n"] for r in sp}):
                rs = sorted((r for r in sp if r["n"] == n), key=lambda r: r["threads"])
                plt.plot([r["threads"] for r in rs], [r[key] for r in rs], "o-", label=f"N={n}")
            if key == "speedup":
                plt.plot([1, pmax], [1, pmax], "k--", label="идеал")
            else:
                plt.axhline(1.0, color="k", ls="--", label="идеал")
            if cores_total:
                plt.axvline(cores_total, color="gray", ls=":", label=f"ядер: {cores_total}")
            plt.xlabel("Число MPI-процессов"); plt.ylabel(ylabel); plt.grid(True); plt.legend(fontsize=8)
            plt.title(title)
            save(fname); made.append(fname)
    return made


def table(rows):
    head = ("| N | Режим | Процессов | Запусков | t средн., с | t мин., с | σ, с | Ускорение S | "
            "Эффективность E | GFLOPS | Верификация | Макс. отн. погрешность |\n" + "|---" * 12 + "|")
    lines = [head]
    for r in sorted(rows, key=lambda r: (r["n"], r["threads"])):
        mode = "sequential" if r["threads"] == 0 else "MPI"
        sp = "-" if r["speedup"] is None else f"{r['speedup']:.2f}"
        ef = "-" if r["efficiency"] is None else f"{r['efficiency']:.2f}"
        lines.append(f"| {r['n']} | {mode} | {r['threads']} | {r['runs']} | {r['mean']:.4f} | {r['min']:.4f} | "
                     f"{r['std']:.4f} | {sp} | {ef} | {r['gflops']:.3f} | {r['verified']} | {r['max_rel_err']:.1e} |")
    return "\n".join(lines)


def conclusions(rows, cores_total):
    mpi = [r for r in rows if r["threads"] > 0 and r["speedup"] is not None]
    out = []
    checked = [r for r in rows if r["verified"] != "NA"]
    if checked and all(r["verified"] == "OK" for r in checked):
        out.append("1. **Корректность.** Результаты MPI-версии совпали с эталоном NumPy (`A @ B`) для всех проверенных "
                   f"конфигураций, максимальная относительная погрешность {max(r['max_rel_err'] for r in checked):.1e}.")
    elif checked:
        out.append("1. **Корректность.** Для части конфигураций верификация НЕ пройдена - см. таблицу, нужно искать ошибку.")
    else:
        out.append("1. **Корректность.** Верификация в этом запуске не выполнялась (`NA`): проверьте выходные файлы "
                   "командой `python3 tools/verify.py`.")
    if not mpi:
        return "\n".join(out)
    best = max(mpi, key=lambda r: r["speedup"])
    nmax = max(r["n"] for r in mpi)
    pmax = max(r["threads"] for r in mpi)
    big = next(r for r in mpi if r["n"] == nmax and r["threads"] == pmax)
    out.append(f"2. **Ускорение.** Максимальное ускорение S = {best['speedup']:.2f} получено при N={best['n']} и "
               f"{best['threads']} процессах (E = {best['efficiency']:.0%}). Для самой большой матрицы (N={nmax}) при "
               f"{pmax} процессах S = {big['speedup']:.2f}, E = {big['efficiency']:.0%}.")
    small_n = min(r["n"] for r in mpi)
    small = next((r for r in mpi if r["n"] == small_n and r["threads"] == pmax), None)
    if small and small_n != nmax:
        out.append(f"3. **Влияние размера.** При {pmax} процессах для малой матрицы (N={small_n}) S = {small['speedup']:.2f}, "
                   f"для большой (N={nmax}) S = {big['speedup']:.2f}: объём вычислений растёт как N^3, а пересылаемых "
                   "данных - как N^2, поэтому чем больше N, тем меньше доля коммуникаций.")
    out.append("4. **Коммуникации.** Каждый процесс получает всю матрицу B (`MPI_Bcast`) и свою часть A (`MPI_Scatterv`), "
               "затем результат собирается `MPI_Gatherv`. Эти пересылки не ускоряются при росте числа процессов, "
               "поэтому эффективность E падает с ростом P (закон Амдала); в замеры времени коммуникации включены.")
    if cores_total and pmax > cores_total:
        top = big
        ref = next((r for r in mpi if r["n"] == nmax and r["threads"] == cores_total), None)
        if ref:
            out.append(f"5. **Процессов больше ядер.** При {pmax} процессах на {cores_total} ядрах S = {top['speedup']:.2f} против "
                       f"{ref['speedup']:.2f} при {cores_total} процессах: лишние процессы делят ядра и ускорения не дают.")
    else:
        out.append("5. **Число ядер.** Ускорение растёт, пока число процессов не превышает число доступных ядер; "
                   "при межузловых запусках на суперкомпьютере добавляется задержка сети, что видно по падению эффективности.")
    return "\n".join(out)


def write_report(rows, figs, args):
    mpi = [r for r in rows if r["threads"] > 0]
    sizes = ", ".join(str(n) for n in sorted({r["n"] for r in rows}))
    procs = ", ".join(str(p) for p in sorted({r["threads"] for r in mpi}))
    titles = {"time_vs_size.png": "Время от размера", "speedup_vs_procs.png": "Ускорение от числа процессов",
              "efficiency_vs_procs.png": "Эффективность от числа процессов"}
    fig_md = "\n\n".join(f"![{titles[f]}](figures/{f})" for f in figs)
    device = args.platform or f"{device_name()} ({os.cpu_count()} логических ядер)"
    text = f"""# Лабораторная работа №3

## Параллельное умножение квадратных матриц на MPI

**Курс:** Параллельное программирование, 2026 **Дедлайн:** 10.11.2026

**Студент:** {STUDENT} **Группа:** {GROUP}

---

## Задание

Модифицировать программу из л/р №1 для параллельной работы по технологии MPI.

- Провести серию экспериментов с разными размерами матриц (примерно 200, 400, 800, 1200, 1600, 2000).
- Использовать разное количество вычислительных ядер (1, 2, 4, 8 и т.д.).
- Запустить параллельную версию на суперкомпьютере «Сергей Королёв».

## Среда выполнения

- **Платформа:** {device}
- **ОС:** {platform.platform()}
- **MPI:** {mpi_version()}
- **Компилятор:** {compiler_version()}, C++17
- **Сборка:** CMake + Make (Release, `-O3`)
- **Верификация/анализ:** Python 3, NumPy, Matplotlib

## Реализация

В `src/main.cpp` добавлена стратегия **parallel_mpi** (функция `multiplyMPI`, запуск `mpirun -np P ./build/matmul <input> 0 mpi`):

1. Процесс 0 читает матрицы `A` и `B` из файла.
2. `MPI_Bcast` рассылает матрицу `B` всем процессам.
3. `MPI_Scatterv` раздаёт каждому процессу блок строк `A` (по `N/P` строк, остаток - первым процессам, поэтому `N` не обязано делиться на `P`).
4. Каждый процесс вычисляет свои строки `C` тройным циклом (без библиотек).
5. `MPI_Gatherv` собирает строки `C` на процессе 0, он записывает результат в файл.

Время замеряется `MPI_Wtime` между двумя `MPI_Barrier` и включает рассылку, вычисление и сбор данных (чтение и запись файлов не входят).
В выходной файл дополнительно пишутся `compute_seconds` (максимум времени вычисления по процессам) и `comm_seconds`.
Ускорение S = `T_sequential / T_p`, эффективность E = `S / p`; `T_sequential` - последовательная версия на том же размере.

### Сборка и запуск

```bash
mkdir build && cd build
cmake ..
make
cd ..

python3 tools/generate.py 500 data/input_500.txt     # генерация входных данных
mpirun -np 4 ./build/matmul data/input_500.txt 0 mpi # 4 MPI-процесса
python3 tools/verify.py data/input_500.txt data/output_500_parallel_mpi_t4.txt

./tools/run_experiments_lab3.sh                      # локально: вся серия, графики, этот отчёт
sbatch tools/slurm_lab3.sbatch                       # на суперкомпьютере (SLURM)
```

## Эксперименты

- Размеры N: {sizes}
- Число MPI-процессов (1 процесс = 1 ядро): {procs}
- Запусков на конфигурацию: {rows[0]['runs']}; в таблице среднее, минимум и стандартное отклонение σ.

## Результаты

Исходные данные для графиков (то же в `data/results_lab3.csv`, по запускам):

{table(rows)}

{fig_md}

## Верификация

Результат первого запуска каждой конфигурации сравнивался с `A @ B` из NumPy. Допуск - `1e-5` относительно `|A|@|B|`
(программа выводит числа с 6 значащими цифрами, порядок суммирования различается).

## Выводы

{conclusions(rows, args.cores_total)}
"""
    with open(os.path.join(REPORT, "README.md"), "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-report", action="store_true")
    ap.add_argument("--platform", default="", help="текст про платформу, например название суперкомпьютера")
    ap.add_argument("--cores-total", type=int, default=os.cpu_count() or 0, help="число доступных ядер")
    args = ap.parse_args()
    rows = aggregate(RES)
    figs = make_plots(rows, args.cores_total)
    if not args.skip_report:
        os.makedirs(REPORT, exist_ok=True)
        write_report(rows, figs, args)
    print("Готово: report/lab3/figures/" + ("" if args.skip_report else " и README.md"))
