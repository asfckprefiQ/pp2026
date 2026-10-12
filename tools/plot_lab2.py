#!/usr/bin/env python3
"""Графики и отчёт лабы 2: data/results_lab2.csv -> report/lab2/figures/*.png и report/lab2/README.md.

  python3 tools/plot_lab2.py                 # графики + отчёт
  python3 tools/plot_lab2.py --skip-report   # только графики
Имя и группа берутся из tools/plot.py (STUDENT, GROUP).
"""
import argparse
import os
import platform
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aggregate import aggregate
from plot import STUDENT, GROUP, device_name, compiler_version

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "data", "results_lab2.csv")
REPORT = os.path.join(ROOT, "report", "lab2")
FIG = os.path.join(REPORT, "figures")
CPU = os.cpu_count() or 1


def split(rows):
    seq = sorted((r for r in rows if r["threads"] == 0), key=lambda r: r["n"])
    omp = [r for r in rows if r["threads"] > 0 and r["cores"] == 0]
    cor = [r for r in rows if r["cores"] > 0]
    return seq, omp, cor


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name), dpi=130)
    plt.close()


def make_plots(rows):
    os.makedirs(FIG, exist_ok=True)
    seq, omp, cor = split(rows)
    made = []

    plt.figure(figsize=(7, 4.5))
    if seq:
        plt.plot([r["n"] for r in seq], [r["mean"] for r in seq], "o-", label="sequential")
    for t in sorted({r["threads"] for r in omp}):
        rs = sorted((r for r in omp if r["threads"] == t), key=lambda r: r["n"])
        plt.plot([r["n"] for r in rs], [r["mean"] for r in rs], "o-", label=f"OpenMP, {t} пот.")
    plt.xlabel("Размер матрицы N"); plt.ylabel("Время, с"); plt.grid(True); plt.legend()
    plt.title("Время умножения от размера матрицы")
    save("time_vs_size.png"); made.append("time_vs_size.png")

    sp = [r for r in omp if r["speedup"] is not None]
    if sp:
        tmax = max(r["threads"] for r in sp)
        for key, ylabel, fname, title in (
                ("speedup", "Ускорение S = T1 / Tp", "speedup_vs_threads.png", "Ускорение от числа потоков"),
                ("efficiency", "Эффективность E = S / p", "efficiency_vs_threads.png", "Эффективность от числа потоков")):
            plt.figure(figsize=(7, 4.5))
            for n in sorted({r["n"] for r in sp}):
                rs = sorted((r for r in sp if r["n"] == n), key=lambda r: r["threads"])
                plt.plot([r["threads"] for r in rs], [r[key] for r in rs], "o-", label=f"N={n}")
            if key == "speedup":
                plt.plot([1, tmax], [1, tmax], "k--", label="идеал")
            else:
                plt.axhline(1.0, color="k", ls="--", label="идеал")
            plt.axvline(CPU, color="gray", ls=":", label=f"логических ядер: {CPU}")
            plt.xlabel("Число потоков"); plt.ylabel(ylabel); plt.grid(True); plt.legend(fontsize=8)
            plt.title(title)
            save(fname); made.append(fname)

    sc = [r for r in cor if r["speedup"] is not None]
    if sc:
        cmax = max(r["cores"] for r in sc)
        plt.figure(figsize=(7, 4.5))
        for n in sorted({r["n"] for r in sc}):
            rs = sorted((r for r in sc if r["n"] == n), key=lambda r: r["cores"])
            plt.plot([r["cores"] for r in rs], [r["speedup"] for r in rs], "o-", label=f"N={n}")
        plt.plot([1, cmax], [1, cmax], "k--", label="идеал")
        plt.xlabel("Число ядер (потоки = ядра)"); plt.ylabel("Ускорение S = T1 / Tp"); plt.grid(True)
        plt.legend(fontsize=8); plt.title("Ускорение от числа ядер (taskset)")
        save("speedup_vs_cores.png"); made.append("speedup_vs_cores.png")
    return made


def table(rows):
    head = ("| N | Режим | Потоки | Ядра | Запусков | t средн., с | t мин., с | σ, с | Ускорение S | "
            "Эффективность E | GFLOPS | Верификация | Макс. отн. погрешность |\n" + "|---" * 13 + "|")
    lines = [head]
    for r in sorted(rows, key=lambda r: (r["n"], r["cores"], r["threads"])):
        mode = "sequential" if r["threads"] == 0 else "OpenMP"
        sp = "-" if r["speedup"] is None else f"{r['speedup']:.2f}"
        ef = "-" if r["efficiency"] is None else f"{r['efficiency']:.2f}"
        cores = f"{r['cores']}" if r["cores"] else "все"
        lines.append(f"| {r['n']} | {mode} | {r['threads']} | {cores} | {r['runs']} | {r['mean']:.4f} | {r['min']:.4f} | "
                     f"{r['std']:.4f} | {sp} | {ef} | {r['gflops']:.3f} | {r['verified']} | {r['max_rel_err']:.1e} |")
    return "\n".join(lines)


def conclusions(rows):
    seq, omp, cor = split(rows)
    omp = [r for r in omp if r["speedup"] is not None]
    cor = [r for r in cor if r["speedup"] is not None]
    out = []
    checked = [r for r in rows if r["verified"] != "NA"]
    if checked and all(r["verified"] == "OK" for r in checked):
        out.append("1. **Корректность.** Результаты OpenMP-версии совпали с эталоном NumPy (`A @ B`) для всех проверенных "
                   f"конфигураций, максимальная относительная погрешность {max(r['max_rel_err'] for r in checked):.1e}.")
    else:
        out.append("1. **Корректность.** Для части конфигураций верификация НЕ пройдена - см. таблицу, нужно искать ошибку.")
    if omp:
        best = max(omp, key=lambda r: r["speedup"])
        nmax = max(r["n"] for r in omp)
        tmax = max(r["threads"] for r in omp)
        big = next(r for r in omp if r["n"] == nmax and r["threads"] == tmax)
        out.append(f"2. **Ускорение.** Максимальное ускорение S = {best['speedup']:.2f} получено при N={best['n']} и "
                   f"{best['threads']} потоках (эффективность E = {best['efficiency']:.0%}). Для самой большой матрицы "
                   f"(N={nmax}) при {tmax} потоках S = {big['speedup']:.2f}, E = {big['efficiency']:.0%}.")
        if tmax > 1 and big["speedup"] > 1:
            f = max(0.0, (1 / big["speedup"] - 1 / tmax) / (1 - 1 / tmax))
            out.append(f"3. **Закон Амдала.** По S(N={nmax}, p={tmax}) эффективная доля неускоряемой части оценивается "
                       f"примерно в {f:.1%}: это накладные расходы на создание потоков, неравномерность нагрузки и "
                       "ограничения памяти (матрица B обходится по столбцам, потоки конкурируют за кэш и шину).")
        small = [r for r in omp if r["n"] == min(x["n"] for x in omp) and r["threads"] == tmax]
        if small and nmax != small[0]["n"]:
            out.append(f"4. **Влияние размера.** Для малой матрицы (N={small[0]['n']}) ускорение при {tmax} потоках "
                       f"{small[0]['speedup']:.2f}, для большой (N={nmax}) - {big['speedup']:.2f}: чем больше N, тем меньше "
                       "доля накладных расходов в общем времени.")
        over = [r for r in omp if r["threads"] > CPU]
        if over:
            at_cpu = [r for r in omp if r["n"] == nmax and r["threads"] <= CPU]
            ref = max(at_cpu, key=lambda r: r["threads"]) if at_cpu else None
            top = max((r for r in over if r["n"] == nmax), key=lambda r: r["threads"], default=None)
            if ref and top:
                out.append(f"5. **Потоков больше ядер.** При {top['threads']} потоках на {CPU} логических ядрах "
                           f"S = {top['speedup']:.2f} против {ref['speedup']:.2f} при {ref['threads']} потоках: "
                           "потоков больше, чем ядер, ускорение не растёт (переключения контекста).")
        else:
            out.append(f"5. **Ядра машины.** Максимум потоков ({tmax}) не превышает число логических ядер ({CPU}). "
                       "Если процессор использует Hyper-Threading, рост ускорения после числа физических ядер "
                       "заметно слабее, чем до него.")
    if cor:
        cmax = max(r["cores"] for r in cor)
        nmax = max(r["n"] for r in cor)
        c = next((r for r in cor if r["n"] == nmax and r["cores"] == cmax), None)
        u = next((r for r in omp if r["n"] == nmax and r["threads"] == cmax), None)
        if c and u:
            out.append(f"6. **Число ядер.** При ограничении процесса {cmax} ядрами (taskset) для N={nmax} S = {c['speedup']:.2f}; "
                       f"без ограничения при том же числе потоков S = {u['speedup']:.2f}. Ускорение определяется "
                       "числом реально доступных ядер, а не только числом потоков.")
    else:
        out.append("6. **Число ядер.** Ограничить число ядер (taskset) в этой среде не удалось, поэтому эксперименты "
                   f"выполнены на фиксированном числе ядер машины ({CPU}), число потоков варьировалось.")
    return "\n".join(out)


def write_report(rows, figs):
    seq, omp, cor = split(rows)
    sizes = ", ".join(str(n) for n in sorted({r["n"] for r in rows}))
    threads = ", ".join(str(t) for t in sorted({r["threads"] for r in omp}))
    cores = ", ".join(str(c) for c in sorted({r["cores"] for r in cor})) or "не использовалось (фиксированное число ядер машины)"
    wsl = " (WSL)" if "microsoft" in platform.release().lower() else ""
    titles = {"time_vs_size.png": "Время от размера", "speedup_vs_threads.png": "Ускорение от числа потоков",
              "efficiency_vs_threads.png": "Эффективность от числа потоков", "speedup_vs_cores.png": "Ускорение от числа ядер"}
    fig_md = "\n\n".join(f"![{titles[f]}](figures/{f})" for f in figs)
    cor_md = f"\n\n### Серия по числу ядер (taskset)\n\n{table(cor)}" if cor else ""
    text = f"""# Лабораторная работа №2

## Параллельное умножение квадратных матриц на OpenMP

**Курс:** Параллельное программирование, 2026 **Дедлайн:** 15.10.2026

**Студент:** {STUDENT} **Группа:** {GROUP}

---

## Задание

Модифицировать программу из л/р №1 для параллельной работы по технологии OpenMP.

- Провести серию экспериментов с разным количеством потоков (1, 2, 4, 8 и т.д.).
- Использовать разные размеры матриц (примерно 200, 400, 800, 1200, 1600, 2000).
- Использовать разное количество вычислительных ядер при наличии технической возможности, иначе фиксированное существующее количество ядер.

## Среда выполнения

- **Устройство:** {device_name()} ({CPU} логических ядер)
- **ОС:** {platform.platform()}{wsl}
- **Компилятор:** {compiler_version()}, C++17, OpenMP
- **Сборка:** CMake + Make (Release, `-O3`)
- **Верификация/анализ:** Python 3, NumPy, Matplotlib

## Реализация

В `src/main.cpp` к стратегиям `sequential` и `parallel_threads` добавлена **parallel_openmp**:

```cpp
omp_set_dynamic(0);
omp_set_num_threads(T);
#pragma omp parallel for schedule(static)
for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {{ /* sum по k */ }}
```

Внешний цикл по строкам результирующей матрицы делится между `T` потоками статически (равными блоками строк). Каждый поток
пишет только в свои строки `C`, поэтому синхронизация не нужна. `omp_set_dynamic(0)` отключает автоподбор, чтобы число потоков
было ровно `T`. Время замеряется `std::chrono::high_resolution_clock` строго вокруг вычисления, без чтения и записи файлов.

Ускорение S = `T_sequential / T_p`, эффективность E = `S / p`, где `T_sequential` - последовательная версия на том же
размере в той же серии запусков.

### Сборка и запуск

```bash
mkdir build && cd build
cmake ..
make
cd ..

python3 tools/generate.py 500 data/input_500.txt     # генерация входных данных
./build/matmul data/input_500.txt 4 omp              # OpenMP, 4 потока
python3 tools/verify.py data/input_500.txt data/output_500_parallel_openmp_t4.txt

./tools/run_experiments_lab2.sh                      # вся серия экспериментов, графики, этот отчёт
```

## Эксперименты

- Размеры N: {sizes}
- Потоки OpenMP на всех ядрах машины: {threads}
- Число ядер (потоки = ядра, процесс привязан через `taskset`): {cores}
- Запусков на конфигурацию: {rows[0]['runs']}; в таблицах среднее, минимум и стандартное отклонение σ.

## Результаты

Исходные данные для графиков (то же в `data/results_lab2.csv`, по запускам).

### Серия по числу потоков

{table(seq + omp)}{cor_md}

{fig_md}

## Верификация

Результат первого запуска каждой конфигурации серии «по потокам» (и sequential) сравнивался с `A @ B` из NumPy.
Допуск - `1e-5` относительно `|A|@|B|` (программа выводит числа с 6 значащими цифрами, порядок суммирования различается).
В серии «по ядрам» используется тот же код, что и в серии «по потокам», поэтому там верификация не повторялась (`NA`).

## Выводы

{conclusions(rows)}
"""
    with open(os.path.join(REPORT, "README.md"), "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-report", action="store_true")
    args = ap.parse_args()
    rows = aggregate(RES)
    figs = make_plots(rows)
    if not args.skip_report:
        write_report(rows, figs)
    print("Готово: report/lab2/figures/" + ("" if args.skip_report else " и README.md"))
