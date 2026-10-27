#!/usr/bin/env python3
"""Графики и отчёт лабы 4: data/results_lab4.csv -> report/lab4/figures/*.png и report/lab4/README.md.

  python3 tools/plot_lab4.py
  python3 tools/plot_lab4.py --skip-report
  python3 tools/plot_lab4.py --os "Ubuntu 22.04 (Google Colab)" --compiler "g++ 11.4, nvcc 13.0"   # если замеры шли на другой машине
Имя и группа берутся из tools/plot.py (STUDENT, GROUP); видеокарта - из data/device_lab4.txt.
"""
import argparse
import math
import os
import platform
import subprocess
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aggregate import aggregate
from plot import STUDENT, GROUP, device_name, compiler_version

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "data", "results_lab4.csv")
DEVICE = os.path.join(ROOT, "data", "device_lab4.txt")
REPORT = os.path.join(ROOT, "report", "lab4")
FIG = os.path.join(REPORT, "figures")
KERNELS = ("naive", "tiled")
KNAME = {"naive": "naive (глобальная память)", "tiled": "tiled (shared-память)"}


def device_info():
    info = {"device": "видеокарта не определена", "sm_count": "", "cuda": "", "cpu": "", "cores": "", "os": "", "compiler": ""}
    if os.path.exists(DEVICE):
        for line in open(DEVICE, encoding="utf-8"):
            line = line.strip()
            if line.startswith("device "):
                info["device"] = line[len("device "):]
            elif line.startswith("sm_count "):
                info["sm_count"] = line.split()[1]
            elif line.startswith("cpu "):
                info["cpu"] = line[4:].strip()
            elif line.startswith("os "):
                info["os"] = line[3:].strip()
            elif line.startswith("compiler "):
                info["compiler"] = line[9:].strip()
            elif line.startswith("cores "):
                info["cores"] = line.split()[1]
            elif "release" in line:
                info["cuda"] = line
    return info


def prepare(rows):
    seq = {r["n"]: r for r in rows if r["strategy"] == "sequential"}
    cuda = [r for r in rows if r["strategy"].startswith("cuda_")]
    for r in cuda:
        r["kernel"] = r["strategy"][5:]
        r["block"] = r["threads"]
        r["grid"] = math.ceil(r["n"] / r["block"])
        base = seq.get(r["n"])
        r["speedup_total"] = base["mean"] / r["mean"] if base else None
        r["speedup_kernel"] = base["mean"] / r["kernel_mean"] if base and r["kernel_mean"] else None
        r["transfer"] = max(r["mean"] - r["kernel_mean"], 0.0) if r["kernel_mean"] is not None else None
    return seq, cuda


def best(cuda, kernel, n, key="kernel_mean"):
    rs = [r for r in cuda if r["kernel"] == kernel and r["n"] == n and r[key] is not None]
    return min(rs, key=lambda r: r[key]) if rs else None


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name), dpi=130)
    plt.close()


def make_plots(seq, cuda):
    os.makedirs(FIG, exist_ok=True)
    ns = sorted({r["n"] for r in cuda})
    blocks = sorted({r["block"] for r in cuda})
    made = []

    # 1. время от размера: sequential и лучшая конфигурация каждого ядра
    plt.figure(figsize=(7, 4.5))
    if seq:
        plt.semilogy(sorted(seq), [seq[n]["mean"] for n in sorted(seq)], "o-", label="sequential (CPU)")
    for k in KERNELS:
        pts = [(n, best(cuda, k, n, "mean")) for n in ns]
        pts = [(n, r) for n, r in pts if r]
        if pts:
            plt.semilogy([n for n, _ in pts], [r["mean"] for _, r in pts], "o-", label=f"CUDA {k}, лучший блок (с копированием)")
    plt.xlabel("Размер матрицы N"); plt.ylabel("Время, с"); plt.grid(True, which="both", alpha=0.4); plt.legend(fontsize=8)
    plt.title("Время умножения от размера матрицы")
    save("time_vs_size.png"); made.append("time_vs_size.png")

    # 2. время ядра от размера блока (линия на каждый N), отдельный график для каждого ядра
    kernels = [k for k in KERNELS if any(r["kernel"] == k for r in cuda)]
    fig, axes = plt.subplots(1, len(kernels), figsize=(6.2 * len(kernels), 4.5), squeeze=False)
    for ax, k in zip(axes[0], kernels):
        for n in ns:
            rs = sorted((r for r in cuda if r["kernel"] == k and r["n"] == n and r["kernel_mean"]), key=lambda r: r["block"])
            if rs:
                ax.plot([r["block"] for r in rs], [r["kernel_mean"] for r in rs], "o-", label=f"N={n}")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(blocks); ax.set_xticklabels([f"{b}x{b}" for b in blocks])
        ax.set_xlabel("Конфигурация блока (потоков)"); ax.set_ylabel("Время ядра, с")
        ax.set_title(f"Ядро {KNAME[k]}"); ax.grid(True, which="both", alpha=0.4); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "kernel_time_vs_block.png"), dpi=130); plt.close()
    made.append("kernel_time_vs_block.png")

    # 3. ускорение относительно sequential от размера
    plt.figure(figsize=(7, 4.5))
    for k in KERNELS:
        pts = [(n, best(cuda, k, n, "mean")) for n in ns]
        pts = [(n, r) for n, r in pts if r and r["speedup_total"]]
        if pts:
            line, = plt.semilogy([n for n, _ in pts], [r["speedup_total"] for _, r in pts], "o-", label=f"{k}: с копированием")
            pk = [(n, best(cuda, k, n)) for n in ns]
            pk = [(n, r) for n, r in pk if r and r["speedup_kernel"]]
            plt.semilogy([n for n, _ in pk], [r["speedup_kernel"] for _, r in pk], "s--", color=line.get_color(), label=f"{k}: только ядро")
    plt.axhline(1.0, color="k", ls=":", label="sequential")
    plt.xlabel("Размер матрицы N"); plt.ylabel("Ускорение S = T_seq / T_cuda"); plt.grid(True, which="both", alpha=0.4)
    plt.legend(fontsize=8); plt.title("Ускорение GPU относительно CPU (лучший блок)")
    save("speedup_vs_size.png"); made.append("speedup_vs_size.png")

    # 4. вклад копирования данных при максимальном N
    nmax = max(ns)
    rs = sorted((r for r in cuda if r["n"] == nmax and r["kernel_mean"] is not None),
                key=lambda r: (KERNELS.index(r["kernel"]) if r["kernel"] in KERNELS else 9, r["block"]))
    if rs:
        plt.figure(figsize=(7.5, 4.5))
        labels = [f"{r['kernel']}\n{r['block']}x{r['block']}" for r in rs]
        plt.bar(labels, [r["kernel_mean"] for r in rs], label="ядро")
        plt.bar(labels, [r["transfer"] for r in rs], bottom=[r["kernel_mean"] for r in rs], label="копирование CPU<->GPU")
        plt.ylabel("Время, с"); plt.title(f"Структура времени, N={nmax}"); plt.legend(); plt.grid(True, axis="y", alpha=0.4)
        plt.xticks(fontsize=7)
        save("transfer_vs_kernel.png"); made.append("transfer_vs_kernel.png")
    return made


def table(seq, cuda):
    head = ("| N | Режим | Блок | Сетка блоков | Запусков | t общее, с | t ядра, с | σ общ., с | "
            "S общее | S ядро | GFLOPS ядра | Верификация | Макс. отн. погрешность |\n" + "|---" * 13 + "|")
    lines = [head]
    for n in sorted({r["n"] for r in list(seq.values()) + cuda}):
        if n in seq:
            r = seq[n]
            lines.append(f"| {n} | sequential | - | - | {r['runs']} | {r['mean']:.4f} | - | {r['std']:.4f} | 1.00 | - | "
                         f"{r['gflops']:.3f} | {r['verified']} | {r['max_rel_err']:.1e} |")
        for r in sorted((r for r in cuda if r["n"] == n), key=lambda r: (r["kernel"], r["block"])):
            sg = "-" if r["speedup_total"] is None else f"{r['speedup_total']:.2f}"
            sk = "-" if r["speedup_kernel"] is None else f"{r['speedup_kernel']:.2f}"
            kg = "-" if r["kernel_gflops"] is None else f"{r['kernel_gflops']:.1f}"
            lines.append(f"| {n} | CUDA {r['kernel']} | {r['block']}x{r['block']} | {r['grid']}x{r['grid']} | {r['runs']} | "
                         f"{r['mean']:.4f} | {r['kernel_mean']:.5f} | {r['std']:.4f} | {sg} | {sk} | {kg} | "
                         f"{r['verified']} | {r['max_rel_err']:.1e} |")
    return "\n".join(lines)


def conclusions(seq, cuda):
    out = []
    checked = [r for r in list(seq.values()) + cuda if r["verified"] != "NA"]
    if checked and all(r["verified"] == "OK" for r in checked):
        out.append("1. **Корректность.** Результаты CUDA-версии (оба ядра, все конфигурации блоков) совпали с эталоном NumPy "
                   f"(`A @ B`), максимальная относительная погрешность {max(r['max_rel_err'] for r in checked):.1e}.")
    elif checked:
        out.append("1. **Корректность.** Для части конфигураций верификация НЕ пройдена - см. таблицу, нужно искать ошибку.")
    else:
        out.append("1. **Корректность.** Верификация в этом запуске не выполнялась (`NA`): проверьте выходные файлы `tools/verify.py`.")
    if not cuda:
        return "\n".join(out)

    ns = sorted({r["n"] for r in cuda})
    nmax, nmin = max(ns), min(ns)
    bn, bt = best(cuda, "naive", nmax), best(cuda, "tiled", nmax)
    parts = []
    for b in (bn, bt):
        if b:
            parts.append(f"{b['kernel']}: блок {b['block']}x{b['block']} (сетка {b['grid']}x{b['grid']} = {b['grid'] ** 2} блоков), "
                         f"время ядра {b['kernel_mean']:.4f} с, {b['kernel_gflops']:.1f} GFLOPS")
    out.append(f"2. **Лучшие конфигурации.** Для N={nmax}: " + "; ".join(parts) + ".")

    if bn and bt:
        ratio = bn["kernel_mean"] / bt["kernel_mean"]
        word = "быстрее" if ratio >= 1 else "медленнее"
        r = ratio if ratio >= 1 else 1 / ratio
        out.append(f"3. **naive и tiled.** При N={nmax} лучшее tiled-ядро {word} лучшего naive в {r:.2f} раза: плитки в shared-памяти "
                   "сокращают число обращений к глобальной памяти примерно в размер плитки раз, и это самый заметный фактор "
                   "оптимизации, если ядро упирается в память." if ratio >= 1 else
                   f"3. **naive и tiled.** При N={nmax} лучшее tiled-ядро {word} лучшего naive в {r:.2f} раза: выигрыша от shared-памяти "
                   "здесь нет (например, из-за кэша глобальной памяти или малого размера плитки).")

    sent = []
    for k in KERNELS:
        rs = [r for r in cuda if r["kernel"] == k and r["n"] == nmax and r["kernel_mean"]]
        if len(rs) >= 2:
            worst, top = max(rs, key=lambda r: r["kernel_mean"]), min(rs, key=lambda r: r["kernel_mean"])
            sent.append(f"{k}: от {top['kernel_mean']:.4f} с (блок {top['block']}x{top['block']}) до {worst['kernel_mean']:.4f} с "
                        f"(блок {worst['block']}x{worst['block']}), разброс в {worst['kernel_mean'] / top['kernel_mean']:.1f} раза")
    if sent:
        out.append(f"4. **Размер блока.** Время ядра при N={nmax} зависит от конфигурации сетки: " + "; ".join(sent) + ". "
                   "Блок 4x4 содержит 16 потоков, меньше размера варпа (32), поэтому часть исполнительных устройств простаивает; "
                   "очень большие блоки (32x32 = 1024 потока) ограничивают число одновременно резидентных блоков на мультипроцессоре, "
                   "оптимальная конфигурация зависит от видеокарты и определяется по таблице выше.")

    b = bt or bn
    if b and b["speedup_total"]:
        out.append(f"5. **Сравнение с CPU.** При N={nmax} лучшая CUDA-конфигурация ({b['kernel']}, {b['block']}x{b['block']}) даёт ускорение "
                   f"S = {b['speedup_total']:.1f} с учётом копирования данных и {b['speedup_kernel']:.1f} по времени одного ядра "
                   f"относительно последовательной версии на CPU ({seq[nmax]['mean']:.2f} с).")
    if b and b["transfer"] is not None:
        share = b["transfer"] / b["mean"]
        small = best(cuda, b["kernel"], nmin, "mean")
        extra = ""
        if small and small["speedup_total"] and nmin != nmax:
            if small["speedup_total"] < b["speedup_total"]:
                extra = (f" Для малых матриц (N={nmin}) ускорение с копированием заметно ниже ({small['speedup_total']:.1f} против "
                         f"{b['speedup_total']:.1f}): накладные расходы на копирование и запуск ядра занимают большую долю времени.")
            else:
                extra = f" Для малых матриц (N={nmin}) ускорение с копированием составило {small['speedup_total']:.1f}."
        out.append(f"6. **Копирование данных.** На копирование CPU<->GPU при N={nmax} приходится {share:.0%} общего времени "
                   f"({b['transfer']:.4f} с из {b['mean']:.4f} с).{extra} Объём данных растёт как N^2, а вычислений - как N^3, "
                   "поэтому с ростом N доля копирования, как правило, падает.")
    out.append("7. **Точность.** Вычисления выполнены в `double` (как в остальных лабах, для сопоставимости и проверки по NumPy). "
               "У большинства игровых и облачных видеокарт производительность в double в десятки раз ниже, чем в float, "
               "поэтому абсолютные значения GFLOPS ниже пиковых для float.")
    return "\n".join(out)


def write_report(seq, cuda, figs, args):
    info = device_info()
    ns = ", ".join(str(n) for n in sorted({r["n"] for r in cuda}))
    blocks = ", ".join(f"{b}x{b}" for b in sorted({r["block"] for r in cuda}))
    kernels = ", ".join(k for k in KERNELS if any(r["kernel"] == k for r in cuda))
    runs = cuda[0]["runs"] if cuda else "-"
    titles = {"time_vs_size.png": "Время от размера", "kernel_time_vs_block.png": "Время ядра от конфигурации блока",
              "speedup_vs_size.png": "Ускорение от размера", "transfer_vs_kernel.png": "Структура времени"}
    fig_md = "\n\n".join(f"![{titles[f]}](figures/{f})" for f in figs)
    sm = f", {info['sm_count']} мультипроцессоров" if info["sm_count"] else ""
    text = f"""# Лабораторная работа №4

## Параллельное умножение квадратных матриц на CUDA

**Курс:** Параллельное программирование, 2026 **Дедлайн:** 02.12.2026

**Студент:** {STUDENT} **Группа:** {GROUP}

---

## Задание

Модифицировать программу из л/р №1 для параллельной работы по технологии CUDA.

- Провести эксперименты с разными размерами матриц.
- Провести эксперименты с различными конфигурациями сетки блоков.

## Среда выполнения

- **GPU:** {info['device']}{sm}
- **CUDA Toolkit:** {info['cuda'] or 'уточните (nvcc --version)'}
- **CPU (база для сравнения):** {info['cpu'] or device_name()} ({info['cores'] or os.cpu_count()} логических ядер)
- **ОС:** {args.os or info["os"] or platform.platform()}
- **Компилятор:** {args.compiler or info["compiler"] or compiler_version()}, C++17
- **Сборка:** CMake + Make (Release, `-O3`)
- **Верификация/анализ:** Python 3, NumPy, Matplotlib

## Реализация

Код CUDA находится в `src/cuda_matmul.cu`, вызывается из `src/main.cpp` (стратегии `cuda_naive` и `cuda_tiled`).
Тип данных - `double`, один поток вычисляет один элемент результирующей матрицы.

1. **naive** - поток `(row, col)` считает сумму `A[row][k] * B[k][col]` по `k`, читая данные из глобальной памяти.
2. **tiled** - блок потоков по плиткам размера `BLOCK x BLOCK` загружает фрагменты `A` и `B` в разделяемую (shared) память
   (`__syncthreads()` между загрузкой и вычислением), что уменьшает число обращений к глобальной памяти.

**Конфигурация сетки.** Блок состоит из `B x B` потоков, сетка - из `ceil(N/B) x ceil(N/B)` блоков, поэтому каждый поток
получает свой элемент `C`, а для `N`, не кратных `B`, лишние потоки отсекаются проверкой границ. Исследуются
`B` = {blocks}; допустимо `B*B <= 1024` (предел числа потоков в блоке).

**Время.** Измеряется `cudaEvent`: `t общее` = копирование `A`, `B` на GPU + ядро + копирование `C` обратно;
`t ядра` = только ядро. Перед замером выполняется прогревочный запуск, инициализация контекста вынесена из замеров.
Чтение и запись файлов в время не входят. Ускорение `S = T_sequential / T_cuda`, где `T_sequential` - последовательная
версия на CPU для того же `N`. GFLOPS ядра = `2*N^3 / t ядра / 10^9`.

### Сборка и запуск

Нужны видеокарта NVIDIA, драйвер и CUDA Toolkit (`nvcc`).

```bash
mkdir build && cd build
cmake ..
make
cd ..

python3 tools/generate.py 500 data/input_500.txt     # генерация входных данных
./build/matmul data/input_500.txt 16 cuda tiled      # блок 16x16, ядро tiled
./build/matmul data/input_500.txt 16 cuda naive      # блок 16x16, ядро naive
python3 tools/verify.py data/input_500.txt data/output_500_cuda_tiled_b16.txt

./tools/run_experiments_lab4.sh                      # вся серия экспериментов, графики, этот отчёт
```

## Эксперименты

- Размеры N: {ns}
- Ядра: {kernels}
- Конфигурации блока: {blocks}
- Запусков на конфигурацию: {runs}; в таблице среднее и стандартное отклонение σ.

## Результаты

Исходные данные для графиков (то же в `data/results_lab4.csv`, по запускам). `S общее` - ускорение с учётом копирования
данных, `S ядро` - по времени одного ядра.

{table(seq, cuda)}

{fig_md}

## Верификация

Результат первого запуска каждой конфигурации сравнивался с `A @ B` из NumPy. Допуск - `1e-5` относительно `|A|@|B|`
(программа выводит числа с 6 значащими цифрами, порядок суммирования различается).

## Выводы

{conclusions(seq, cuda)}
"""
    with open(os.path.join(REPORT, "README.md"), "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-report", action="store_true")
    ap.add_argument("--os", default="", help="ОС машины, где шли замеры (например: Ubuntu 22.04, Google Colab)")
    ap.add_argument("--compiler", default="", help="компилятор машины замеров (например: g++ 11.4, nvcc 13.0)")
    args = ap.parse_args()
    seq, cuda = prepare(aggregate(RES))
    if not cuda:
        raise SystemExit("В data/results_lab4.csv нет строк CUDA: сначала запустите tools/run_experiments_lab4.sh")
    figs = make_plots(seq, cuda)
    if not args.skip_report:
        os.makedirs(REPORT, exist_ok=True)
        write_report(seq, cuda, figs, args)
    print("Готово: report/lab4/figures/" + ("" if args.skip_report else " и README.md"))
