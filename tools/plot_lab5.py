#!/usr/bin/env python3
"""Графики и отчёт лабы 5: data/results_lab5.csv -> report/lab5/figures/*.png и report/lab5/README.md.

  python3 tools/plot_lab5.py
  python3 tools/plot_lab5.py --skip-report
Имя и группа берутся из tools/plot.py (STUDENT, GROUP).
"""
import argparse
import csv
import os
import platform
import statistics as st
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot import STUDENT, GROUP, device_name, compiler_version

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "data", "results_lab5.csv")
REP = os.path.join(ROOT, "report", "lab5")
FIG = os.path.join(REP, "figures")
MODE_RU = {"block": "блоки строк", "interleaved": "чередование строк"}


def f(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def load():
    data = {}  # (view, mode, threads) -> list of (serial, thread, [times])
    for r in csv.DictReader(open(RES, encoding="utf-8")):
        k = (int(r["view"]), r["mode"], int(r["threads"]))
        data.setdefault(k, []).append((float(r["serial_ms"]), float(r["thread_ms"]),
                                       [float(x) for x in r["thread_times_ms"].split(";")]))
    agg = {}
    for k, rows in data.items():
        best = min(rows, key=lambda x: x[1])
        # Последовательная и параллельная версии меряются подряд в одном запуске программы, поэтому ускорение
        # берётся внутри запуска, а из запусков - медианный по ускорению (шум между запусками не смешивается).
        mid = sorted(rows, key=lambda x: x[0] / x[1])[(len(rows) - 1) // 2]
        agg[k] = dict(serial=mid[0], par=mid[1], speedup=mid[0] / mid[1], times=best[2])
    return agg


def speedup_plot(agg, view):
    plt.figure(figsize=(7, 4.5))
    ths = sorted({k[2] for k in agg if k[0] == view})
    for mode in ("block", "interleaved"):
        xs = [t for t in ths if (view, mode, t) in agg]
        if xs:
            plt.plot(xs, [agg[(view, mode, t)]["speedup"] for t in xs], "o-", label=MODE_RU[mode])
    plt.plot(ths, ths, "k--", alpha=.4, label="идеал (линейное)")
    plt.xlabel("число потоков"); plt.ylabel("ускорение, раз"); plt.grid(alpha=.3)
    plt.title(f"Mandelbrot, view {view}"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"speedup_view{view}.png"), dpi=130); plt.close()


def times_plot(agg, view, t):
    plt.figure(figsize=(7, 4))
    for mode, off in (("block", -0.2), ("interleaved", 0.2)):
        k = (view, mode, t)
        if k in agg:
            plt.bar([i + off for i in range(t)], agg[k]["times"], width=0.4, label=MODE_RU[mode])
    plt.xlabel("номер потока"); plt.ylabel("время потока, мс"); plt.title(f"view {view}, {t} потоков")
    plt.legend(); plt.grid(alpha=.3, axis="y"); plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"thread_times_view{view}_{t}.png"), dpi=130); plt.close()


def table(agg, views, modes, ths):
    out = ["| view | разбиение | потоков | T_посл, мс | T_парал, мс | ускорение | эффективность |",
           "|---|---|---|---|---|---|---|"]
    for v in views:
        for m in modes:
            for t in ths:
                a = agg.get((v, m, t))
                if a:
                    out.append(f"| {v} | {MODE_RU[m]} | {t} | {f(a['serial'],1)} | {f(a['par'],1)} | "
                               f"{f(a['speedup'])} | {f(a['speedup']/t)} |")
    return "\n".join(out)


def imbalance(times):
    return max(times) / min(times) if min(times) > 0 else float("inf")


def report(agg, views, ths, ncpu):
    L = []
    A = L.append
    A("# Лабораторная работа №5\n")
    A("## Параллельное построение множества Мандельброта (`std::thread`)\n")
    A("**Курс:** Параллельное программирование, 2026 **Дедлайн:** 15.12.2026\n")
    A(f"**Студент:** {STUDENT} **Группа:** {GROUP}\n")
    A("---\n")
    A("## Задание\n")
    A("Из репозитория https://github.com/stanford-cs149/asst1 выполнить одну из программ по пунктам задания. "
      "Выбрана **Program 1: Parallel Fractal Generation Using Threads** (код в `asst1/prog1_mandelbrot_threads`):\n")
    A("1. Распараллелить построение изображения на 2 потока: верхнюю половину считает один поток, нижнюю — другой.")
    A("2. Обобщить на 2–8 потоков (пространственная декомпозиция), построить график ускорения для view 1 и view 2, "
      "объяснить, линейно ли ускорение.")
    A("3. Замерить время работы каждого потока и объяснить по нему график ускорения.")
    A("4. Придумать статическое разбиение без синхронизации, дающее ускорение ≈7–8× на 8 потоках для обоих view.")
    A("5. Запустить на 16 потоках и объяснить, есть ли выигрыш по сравнению с 8.\n")
    A("## Решение\n")
    A("Изменения в заготовке — функция `workerThreadStart` в `mandelbrotThread.cpp` (режим разбиения задаётся ключом "
      "`-m block|interleaved`) и ключи `-m`, `-T`, `-c` в `main.cpp`:\n")
    A("```cpp\nvoid workerThreadStart(WorkerArgs * const args) {\n"
      "    double startTime = CycleTimer::currentSeconds();\n"
      "    if (g_decomposition == 0) {            // блоки: поток i - строки [i*H/T, (i+1)*H/T)\n"
      "        int startRow = args->threadId * args->height / args->numThreads;\n"
      "        int endRow = (args->threadId + 1) * args->height / args->numThreads;\n"
      "        mandelbrotSerial(..., startRow, endRow - startRow, ...);\n"
      "    } else {                               // чередование: поток i - строки i, i+T, i+2T, ...\n"
      "        for (int row = args->threadId; row < (int)args->height; row += args->numThreads)\n"
      "            mandelbrotSerial(..., row, 1, ...);\n"
      "    }\n"
      "    g_threadSeconds[args->threadId] = CycleTimer::currentSeconds() - startTime;  // своя ячейка - без синхронизации\n"
      "}\n```\n")
    A("```bash\ncd asst1/prog1_mandelbrot_threads && make\n./mandelbrot -t 8 -v 1 -m interleaved -T     # 8 потоков, view 1, время каждого потока\n"
      "cd ../.. && bash tools/run_experiments_lab5.sh && python3 tools/plot_lab5.py   # вся серия и этот отчёт\n```\n")
    A("## Условие и среда\n")
    A("Изображение 1600×1200, 256 итераций, два набора параметров (view 1 и view 2). Последовательная версия — `mandelbrotSerial`; "
      "параллельная — `std::thread` (число потоков−1 новых потоков, главный поток тоже работает). "
      "Результат каждой параллельной версии проверяется сравнением с последовательным изображением (программа сообщает об ошибке при расхождении). "
      "Время — минимум из 5 запусков внутри программы, затем медиана из нескольких запусков программы.\n")
    A(f"- Процессор: {device_name()} (логических ядер: {ncpu})")
    A(f"- ОС: {platform.system()} {platform.release()}")
    A(f"- Компилятор: {compiler_version()}, `-O3 -std=c++11`\n")

    A("## Пункт 1. Два потока: верхняя и нижняя половины\n")
    A("Поток 0 считает строки `[0, H/2)`, поток 1 — `[H/2, H)` (блочное разбиение при `T=2`, режим `-m block`).\n")
    for v in views:
        a = agg.get((v, "block", 2))
        if a:
            A(f"- view {v}: ускорение {f(a['speedup'])}× (идеал — 2×).")
    A("")

    A("## Пункт 2. 2…8 потоков, блочное разбиение\n")
    A("Поток `i` получает непрерывный блок строк `[i·H/T, (i+1)·H/T)`.\n")
    for v in views:
        A(f"![view {v}](figures/speedup_view{v}.png)\n")
    for v in views:
        s = {t: agg[(v, "block", t)]["speedup"] for t in ths if (v, "block", t) in agg and t <= 8}
        if s:
            tmax = max(s)
            A(f"- view {v}: при {tmax} потоках ускорение {f(s[tmax])}× (эффективность {f(s[tmax]/tmax)}), "
              f"{'линейным рост не является' if s[tmax] < 0.8*tmax else 'рост близок к линейному'}.")
    if all(("block" in k for k in [])):
        pass
    A("\n**Почему ускорение нелинейное.** Стоимость строки зависит от числа итераций: точки внутри множества Мандельброта "
      "требуют все 256 итераций, точки снаружи выходят быстро. Центральные строки изображения (view 1) «тяжелее» крайних, "
      "поэтому при блочном разбиении потоки, получившие середину картинки, работают заметно дольше остальных, а общее время "
      "определяется самым медленным потоком. Особенно это видно при 3 потоках: средний поток получает почти всё множество, "
      "и ускорение близко к ускорению для 2 потоков, а не к 3×.\n")

    A("## Пункт 3. Время работы каждого потока\n")
    A("В `workerThreadStart` замеряется время работы потока (`CycleTimer::currentSeconds()` до и после расчёта), "
      "результат пишется в отдельную ячейку массива `g_threadSeconds[threadId]`, поэтому синхронизация не нужна. "
      "Показан самый быстрый из прогонов.\n")
    for v in views:
        for t in (3, 8):
            if (v, "block", t) in agg:
                A(f"![время потоков, view {v}, {t} потоков](figures/thread_times_view{v}_{t}.png)\n")
    for v in views:
        for t in (3, 8):
            a = agg.get((v, "block", t))
            if a:
                tm = a["times"]
                A(f"- view {v}, {t} потоков, блоки: время потоков {', '.join(f(x,1) for x in tm)} мс; "
                  f"max/min = {f(imbalance(tm))}.")
    A("\nГрафик ускорения объясняется дисбалансом: чем больше отношение max/min, тем сильнее ускорение отстаёт от числа потоков. "
      "Дополнительно ускорение ограничено числом физических ядер процессора и накладными расходами на создание потоков.\n")

    A("## Пункт 4. Улучшенное статическое разбиение\n")
    A("**Идея.** Строки распределяются циклически: поток `i` считает строки `i, i+T, i+2T, …` "
      "(режим `interleaved`, цикл `for (row = threadId; row < height; row += numThreads)`). "
      "Тяжёлые и лёгкие строки равномерно перемешиваются между потоками, нагрузка выравнивается без динамической "
      "балансировки и без синхронизации (потоки пишут в непересекающиеся строки выходного массива). "
      "Правило одно и то же для любого числа потоков.\n")
    for v in views:
        a = agg.get((v, "interleaved", 8))
        b = agg.get((v, "block", 8))
        if a:
            extra = f" (блоки: {f(b['speedup'])}×)" if b else ""
            A(f"- view {v}, 8 потоков: **{f(a['speedup'])}×**{extra}; "
              f"max/min времени потоков = {f(imbalance(a['times']))}.")
    A("")
    a8 = [agg[(v, "interleaved", 8)]["speedup"] for v in views if (v, "interleaved", 8) in agg]
    if a8 and max(a8) >= 6.5:
        A(f"Ускорение на 8 потоках достигает {f(max(a8))}× — близко к 7–8×, как и требуется в задании.\n")
    elif "1215U" in device_name():
        A(f"Ускорение 7–8× из задания рассчитано на 8 одинаковых ядер (машины курса CS149). Здесь получено "
          f"{f(max(a8)) if a8 else '—'}×, и это предел самой машины, а не разбиения: Intel Core i3-1215U — гибридный "
          "процессор с 2 производительными ядрами (P-cores, с Hyper-Threading — 4 логических потока) и 4 "
          "энергоэффективными ядрами (E-cores), то есть 6 физических ядер и 8 логических. Два потока на одном P-ядре "
          "делят его вычислительные блоки, а E-ядра медленнее P-ядер, поэтому даже при равной нагрузке по строкам "
          "потоки работают с разной скоростью. Отношение max/min времени потоков при чередовании (≈1,3–1,5) — "
          "это разница в скорости ядер, а не в объёме работы.\n")
    else:
        A(f"Ускорение 7–8× достижимо, если у машины 8 одинаковых физических ядер; здесь логических ядер {ncpu}, "
          "часть из них — SMT-потоки одного ядра, которые не удваивают производительность.\n")

    A("## Пункт 5. 16 потоков\n")
    for v in views:
        a8, a16 = agg.get((v, "interleaved", 8)), agg.get((v, "interleaved", 16))
        if a8 and a16:
            r = a16["speedup"] / a8["speedup"]
            A(f"- view {v}: 8 потоков — {f(a8['speedup'])}×, 16 потоков — {f(a16['speedup'])}× (отношение {f(r)}).")
    ratios = [agg[(v, "interleaved", 16)]["speedup"] / agg[(v, "interleaved", 8)]["speedup"]
              for v in views if (v, "interleaved", 16) in agg and (v, "interleaved", 8) in agg]
    A(f"\nЛогических ядер на машине {ncpu}, поэтому 16 потоков не могут исполняться одновременно: ОС переключает их, "
      "и прироста «от новых ядер» нет. ")
    if ratios and max(ratios) > 1.05:
        A("Небольшой выигрыш всё же есть: при 16 потоках каждый поток получает вдвое меньше строк, и когда поток на быстром "
          "ядре заканчивает свою часть, планировщик ОС отдаёт освободившееся ядро следующему потоку. Это частично выравнивает "
          "разницу в скорости P- и E-ядер — статическое разбиение начинает работать как грубая динамическая балансировка. "
          "Предел по-прежнему задаёт число ядер.\n")
    else:
        A("Время с 16 потоками практически не отличается от времени с 8: потоков больше, чем исполнительных устройств, "
          "и накладные расходы на переключение съедают возможный выигрыш.\n")

    A("## Таблица исходных данных\n")
    A(table(agg, views, ("block", "interleaved"), ths))
    A("\n## Выводы\n")
    best = {v: max((agg[k]["speedup"], k[1], k[2]) for k in agg if k[0] == v and k[2] <= 8) for v in views}
    for v in views:
        s, m, t = best[v]
        A(f"- view {v}: лучшее ускорение до 8 потоков — {f(s)}× ({MODE_RU[m]}, {t} потоков).")
    A("- Блочное разбиение даёт плохую балансировку: стоимость строк неравномерна, ускорение ограничено самым нагруженным потоком.")
    A("- Циклическое (чередующееся) статическое разбиение выравнивает нагрузку без синхронизации и даёт ускорение, близкое к числу ядер.")
    A("- Потоков больше, чем аппаратных ядер, ускорения не добавляют.\n")
    open(os.path.join(REP, "README.md"), "w", encoding="utf-8").write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-report", action="store_true")
    ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    agg = load()
    views = sorted({k[0] for k in agg})
    ths = sorted({k[2] for k in agg})
    for v in views:
        speedup_plot(agg, v)
        for t in (3, 8):
            times_plot(agg, v, t)
    ncpu = os.cpu_count() or 0
    if not ap.parse_args().skip_report:
        report(agg, views, ths, ncpu)
    print("Готово: report/lab5/")


if __name__ == "__main__":
    main()
