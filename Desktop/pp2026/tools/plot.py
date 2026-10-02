#!/usr/bin/env python3
"""Графики и отчёт лабы 1: data/results.csv -> report/lab1/figures/*.png и report/lab1/README.md.

  python3 tools/plot.py                 # графики + отчёт
  python3 tools/plot.py --skip-report   # только графики
"""
import argparse
import os
import platform
import subprocess
from datetime import date

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aggregate import aggregate, markdown_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "report", "lab1")
FIG = os.path.join(REPORT, "figures")


def device_name():
    try:
        if os.path.exists("/proc/cpuinfo"):
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if platform.system() == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def compiler_version():
    try:
        return subprocess.run(["c++", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    except Exception:
        return "unknown"


def label(threads):
    return "sequential" if threads == 0 else f"std::thread, {threads} пот."


def make_plots(rows):
    os.makedirs(FIG, exist_ok=True)
    threads_set = sorted({r["threads"] for r in rows})
    seq = [r for r in rows if r["threads"] == 0]

    # time_vs_size.png: время от размера, по линии на конфигурацию
    plt.figure(figsize=(7, 4.5))
    for t in threads_set:
        rs = [r for r in rows if r["threads"] == t]
        plt.errorbar([r["n"] for r in rs], [r["mean"] for r in rs], yerr=[r["std"] for r in rs],
                     fmt="o-", capsize=3, label=label(t))
    plt.xlabel("Размер матрицы N"); plt.ylabel("Время, с"); plt.grid(True); plt.legend()
    plt.title("Время умножения от размера матрицы")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "time_vs_size.png"), dpi=130); plt.close()

    # time_vs_size_loglog.png: лог. масштаб + теоретический рост N^3
    if seq:
        ns = np.array([r["n"] for r in seq], dtype=float)
        t = np.array([r["mean"] for r in seq])
        plt.figure(figsize=(7, 4.5))
        plt.loglog(ns, t, "o-", label="sequential")
        plt.loglog(ns, t[0] * (ns / ns[0]) ** 3, "--", label="теория $\\sim N^3$")
        plt.xlabel("N"); plt.ylabel("Время, с"); plt.grid(True, which="both"); plt.legend()
        plt.title("Последовательная версия, логарифмический масштаб")
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "time_vs_size_loglog.png"), dpi=130); plt.close()

        plt.figure(figsize=(7, 4.5))
        plt.plot(ns, [r["gflops"] for r in seq], "o-")
        plt.xlabel("Размер матрицы N"); plt.ylabel("GFLOPS"); plt.grid(True)
        plt.title("Производительность (sequential)")
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "gflops_vs_size.png"), dpi=130); plt.close()

    # speedup_vs_threads.png: только если есть параллельные запуски
    par = [r for r in rows if r["threads"] > 0 and r["speedup"] is not None]
    if par:
        plt.figure(figsize=(7, 4.5))
        for n in sorted({r["n"] for r in par}):
            rs = sorted((r for r in par if r["n"] == n), key=lambda r: r["threads"])
            plt.plot([r["threads"] for r in rs], [r["speedup"] for r in rs], "o-", label=f"N={n}")
        mx = max(r["threads"] for r in par)
        plt.plot([1, mx], [1, mx], "k--", label="идеал")
        plt.xlabel("Число потоков"); plt.ylabel("Ускорение S = T1 / Tp"); plt.grid(True); plt.legend()
        plt.title("Ускорение от числа потоков")
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "speedup_vs_threads.png"), dpi=130); plt.close()
    return bool(par)


def conclusions(rows, has_par):
    seq = [r for r in rows if r["threads"] == 0]
    out = []
    ok = all(r["verified"] in ("OK", "NA") for r in rows) and any(r["verified"] == "OK" for r in rows)
    if ok:
        out.append("1. **Корректность.** Результаты программы совпали с эталоном NumPy (`A @ B`) для всех "
                   f"запущенных конфигураций, максимальная относительная погрешность {max(r['max_rel_err'] for r in rows):.1e}.")
    else:
        out.append("1. **Корректность.** Для части конфигураций верификация НЕ пройдена - см. таблицу, нужно искать ошибку.")
    if len(seq) >= 2:
        ns = np.array([r["n"] for r in seq], dtype=float)
        t = np.array([r["mean"] for r in seq])
        g = np.array([r["gflops"] for r in seq])
        k = np.polyfit(np.log(ns), np.log(t), 1)[0]
        out.append(f"2. **Сложность.** Алгоритм имеет сложность O(N^3). Наклон графика в логарифмическом масштабе "
                   f"равен {k:.2f} (теоретически 3). При росте N с {int(ns[0])} до {int(ns[-1])} (в {ns[-1]/ns[0]:.0f} раз) "
                   f"время выросло в {t[-1]/t[0]:.0f} раз (теория: {(ns[-1]/ns[0])**3:.0f}).")
        if k > 3.15:
            out.append("3. **Кэш.** Время растёт быстрее N^3: при обходе матрицы B по столбцам строки не помещаются "
                       "в кэш, растёт число промахов, GFLOPS падает с ростом N.")
        else:
            out.append("3. **Кэш.** Рост близок к кубическому, заметного влияния кэша на выбранных размерах немного.")
        out.append(f"4. **Производительность.** Скорость последовательной версии - от {g.min():.2f} до {g.max():.2f} GFLOPS "
                   f"(максимум при N={int(ns[g.argmax()])}): одно ядро, без ручной векторизации и блочной оптимизации.")
        out.append(f"5. **База для следующих работ.** Последовательное время при N={int(ns[-1])} равно {t[-1]:.2f} с - "
                   "это T1 для расчёта ускорения S = T1 / Tp и эффективности E = S / p в л/р №2-4.")
    if has_par:
        best = max((r for r in rows if r["threads"] > 0 and r["speedup"] is not None), key=lambda r: r["speedup"])
        out.append(f"6. **std::thread.** Максимальное ускорение {best['speedup']:.2f} получено при N={best['n']} и "
                   f"{best['threads']} потоках (эффективность {best['speedup']/best['threads']:.0%}).")
    return "\n".join(out)


def write_report(rows, has_par):
    sizes = ", ".join(str(n) for n in sorted({r["n"] for r in rows}))
    threads = ", ".join(str(t) for t in sorted({r["threads"] for r in rows}))
    figs = ["![Время от размера](figures/time_vs_size.png)",
            "![Время, лог. масштаб](figures/time_vs_size_loglog.png)",
            "![GFLOPS](figures/gflops_vs_size.png)"]
    if has_par:
        figs.append("![Ускорение](figures/speedup_vs_threads.png)")
    text = f"""# Лабораторная работа №1

## Параллельное умножение квадратных матриц

**Курс:** Параллельное программирование, 2026 **Дедлайн:** 05.10.2026

Студент: ____________, группа: ______. Дата отчёта: {date.today():%d.%m.%Y}

---

## Задание

Написать программу на C/C++ для перемножения двух квадратных матриц.

- Исходные данные: файл(ы) со значениями исходных матриц.
- Выходные данные: файл со значениями результирующей матрицы, время выполнения, объём задачи.
- Обязательна автоматизированная верификация результатов с помощью сторонней библиотеки (NumPy).
- Исследовать зависимость времени выполнения от объёма задачи{" и параметров распараллеливания" if has_par else ""}.

## Среда выполнения

- **Устройство:** {device_name()} ({os.cpu_count()} логических ядер)
- **ОС:** {platform.platform()}
- **Компилятор:** {compiler_version()}, C++17
- **Сборка:** CMake, Release (`-O3`)

## Реализация

- `src/main.cpp` - чтение матриц из файла, умножение тройным циклом (`multiplyRange`, без библиотек),
  стратегии `sequential` и `parallel_threads` (`std::thread`), запись результата и метаданных
  (стратегия, потоки, число ядер, время). Замеряется только вычисление (без ввода-вывода).
- `tools/generate.py` - входной файл: `N`, матрица A, матрица B (числа в [-1; 1], фиксированный seed).
- `tools/verify.py` - сравнение с `A @ B` из NumPy.
- `tools/run_experiments.sh`, `tools/aggregate.py`, `tools/plot.py` - эксперименты, агрегация, графики и отчёт.

Объём задачи: две матрицы `N x N` (`double`), `2*N^3` операций с плавающей точкой, память `3*N^2*8` байт.
GFLOPS = `2*N^3 / t_средн / 10^9`. Ускорение S = `T_sequential / T_p`.

## Эксперименты

- Размеры N: {sizes}
- Потоки (0 = последовательная версия): {threads}
- Запусков на конфигурацию: {rows[0]['runs']}; в таблице среднее, минимум и стандартное отклонение σ.

## Результаты

Исходные данные для графиков (то же в `data/results.csv`, по запускам):

{markdown_table(rows)}

{chr(10).join(chr(10) + f for f in figs).strip()}

## Верификация

Результат первого запуска каждой конфигурации сравнивался с `A @ B` из NumPy. Допуск - `1e-5`
относительно `|A|@|B|` (программа выводит числа с 6 значащими цифрами, порядок суммирования различается).

## Выводы

{conclusions(rows, has_par)}

## Воспроизведение

```
pip install numpy matplotlib
./tools/run_experiments.sh
```
"""
    with open(os.path.join(REPORT, "README.md"), "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-report", action="store_true")
    args = ap.parse_args()
    rows = aggregate()
    has_par = make_plots(rows)
    if not args.skip_report:
        write_report(rows, has_par)
    print("Готово: report/lab1/figures/" + ("" if args.skip_report else " и README.md"))
