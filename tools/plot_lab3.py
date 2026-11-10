#!/usr/bin/env python3
"""Графики и отчёт лабы 3: data/results_lab3.csv -> report/lab3/figures/*.png и report/lab3/README.md.

  python3 tools/plot_lab3.py
  python3 tools/plot_lab3.py --platform "суперкомпьютер «Сергей Королёв» (Самарский университет)"
  python3 tools/plot_lab3.py --skip-report

Сведения о платформе (процессор, MPI, компилятор, узлы) берутся из data/env_lab3.txt, который пишет
tools/run_experiments_lab3.sh на той машине, где шли замеры (в т.ч. на кластере); если файла нет - с текущей машины.
Время в таблицах - медиана по повторам; S(p) = T(1) / T(p) относительно MPI-версии на одном процессе,
E = S / p. Имя и группа берутся из tools/plot.py (STUDENT, GROUP).
"""
import argparse
import csv
import os
import statistics
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aggregate import aggregate
from plot import STUDENT, GROUP, device_name

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "data", "results_lab3.csv")
ENV = os.path.join(ROOT, "data", "env_lab3.txt")
RES_LAB2 = os.path.join(ROOT, "data", "results_lab2.csv")
REPORT = os.path.join(ROOT, "report", "lab3")
FIG = os.path.join(REPORT, "figures")


def c(x, nd=2):
    """Число с десятичной запятой."""
    return f"{x:.{nd}f}".replace(".", ",")


def ct(t):
    return c(t, 3 if t >= 0.01 else 4)


def first_line(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True)
        lines = (out.stdout + out.stderr).strip().splitlines()
        return lines[0].strip() if lines else ""
    except Exception:
        return ""


def load_env():
    env = {}
    if os.path.exists(ENV):
        with open(ENV, encoding="utf-8", errors="replace") as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    env.setdefault(k.strip(), v.strip())
    env.setdefault("cpu", device_name())
    env.setdefault("cores_per_node", str(os.cpu_count() or ""))
    env.setdefault("compiler", first_line(["mpicxx", "--version"]) or first_line(["c++", "--version"]) or "unknown")
    env.setdefault("mpi", first_line(["mpirun", "--version"]) or "MPI (версию уточните)")
    return env


def load_extra():
    """Медианы времени вычисления и число узлов по (N, P) из CSV (их нет в aggregate)."""
    comp, nodes = {}, {}
    with open(RES, newline="") as f:
        for r in csv.DictReader(f):
            key = (int(r["n"]), int(r["threads"]))
            if r.get("compute_seconds"):
                comp.setdefault(key, []).append(float(r["compute_seconds"]))
            if r.get("nodes"):
                nodes[key] = int(r["nodes"])
    return {k: statistics.median(v) for k, v in comp.items()}, nodes


class Data:
    def __init__(self, rows, comp, nodes):
        self.rows = rows
        self.seq = {r["n"]: r for r in rows if r["threads"] == 0}
        self.mpi = [r for r in rows if r["threads"] > 0]
        self.P = sorted({r["threads"] for r in self.mpi})
        self.med = {(r["n"], r["threads"]): r["median"] for r in self.mpi}
        self.N = sorted(n for n in {r["n"] for r in self.mpi} if all((n, p) in self.med for p in self.P))
        self.p0 = self.P[0]
        self.pmax = self.P[-1]
        self.runs = self.mpi[0]["runs"]
        self.comp = comp
        self.nodes = nodes

    def S(self, n, p):
        return self.med[(n, self.p0)] / self.med[(n, p)]

    def E(self, n, p):
        return self.S(n, p) / (p / self.p0)

    def gflops(self, n, p):
        return 2.0 * n ** 3 / self.med[(n, p)] / 1e9

    def comm_share(self, n, p):
        """Доля времени на обмены (Bcast + Scatterv + Gatherv + ожидание) в полном времени."""
        if (n, p) not in self.comp or (n, p) not in self.med:
            return None
        return max(0.0, 1.0 - self.comp[(n, p)] / self.med[(n, p)])

    def nodes_for(self, p):
        vals = {v for (n, q), v in self.nodes.items() if q == p}
        return max(vals) if vals else None

    def E_comp(self, n, p):
        """Эффективность только вычислительной части (без пересылок): t_выч(1) / (p * t_выч(p))."""
        if (n, self.p0) not in self.comp or (n, p) not in self.comp:
            return None
        return self.comp[(n, self.p0)] / ((p / self.p0) * self.comp[(n, p)])


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name), dpi=130)
    plt.close()


def make_plots(d, cores_per_node):
    os.makedirs(FIG, exist_ok=True)
    made = []

    plt.figure(figsize=(7, 4.5))
    seq_n = [n for n in d.N if n in d.seq]
    if seq_n:
        plt.loglog(seq_n, [d.seq[n]["median"] for n in seq_n], "ks--", label="sequential")
    for p in d.P:
        plt.loglog(d.N, [d.med[(n, p)] for n in d.N], "o-", label=f"MPI, {p} проц.")
    plt.xlabel("Размер матрицы N"); plt.ylabel("Время, с"); plt.grid(True, which="both", alpha=0.4); plt.legend(fontsize=8)
    plt.title("Время выполнения от размера матрицы")
    save("time_vs_size.png"); made.append("time_vs_size.png")

    if len(d.P) > 1:
        plt.figure(figsize=(7, 4.5))
        for n in d.N:
            plt.plot(d.P, [d.S(n, p) for p in d.P], "o-", label=f"N={n}")
        plt.plot(d.P, [p / d.p0 for p in d.P], "k--", label="идеальное линейное")
        one_node = [p for p in d.P if d.nodes_for(p) == 1]
        if one_node and (d.nodes_for(d.pmax) or 1) > 1:   # реальная граница одного узла в этом запуске
            plt.axvline(one_node[-1], color="gray", ls=":",
                        label=f"до {one_node[-1]} процессов — один узел, дальше несколько")
        elif cores_per_node and d.p0 < cores_per_node < d.pmax:
            plt.axvline(cores_per_node, color="gray", ls=":", label=f"ядер в узле: {cores_per_node}")
        plt.xlabel("Число MPI-процессов"); plt.ylabel(f"Ускорение S = T({d.p0}) / T(p)"); plt.grid(True)
        plt.legend(fontsize=8)
        plt.title("Ускорение от числа процессов")
        save("speedup_vs_procs.png"); made.append("speedup_vs_procs.png")

        plt.figure(figsize=(7, 4.5))
        for n in d.N:
            plt.plot(d.P, [d.E(n, p) for p in d.P], "o-", label=f"N={n}")
        plt.axhline(1.0, color="k", ls="--", label="идеал")
        plt.xlabel("Число MPI-процессов"); plt.ylabel("Эффективность E = S / p"); plt.grid(True); plt.legend(fontsize=8)
        plt.title("Эффективность от числа процессов")
        save("efficiency_vs_procs.png"); made.append("efficiency_vs_procs.png")

        if d.comp:
            plt.figure(figsize=(7, 4.5))
            for n in d.N:
                vals = [d.comm_share(n, p) for p in d.P]
                if all(v is not None for v in vals):
                    plt.plot(d.P, [100 * v for v in vals], "o-", label=f"N={n}")
            plt.xlabel("Число MPI-процессов"); plt.ylabel("Доля обменов, %"); plt.grid(True); plt.legend(fontsize=8)
            plt.title("Доля пересылок данных в общем времени")
            save("comm_share_vs_procs.png"); made.append("comm_share_vs_procs.png")
    return made


def wide_table(d):
    p0, others = d.P[0], d.P[1:]
    head = ["N", "посл., с", (f"{p0} процесс, с" if p0 == 1 else f"{p0} проц., с")] + [f"{p} проц., с" for p in others] \
        + [f"S ({p})" for p in others] + ([f"E ({d.pmax})"] if others else [])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for n in d.N:
        seq = ct(d.seq[n]["median"]) if n in d.seq else "-"
        cells = [str(n), seq, ct(d.med[(n, p0)])] + [ct(d.med[(n, p)]) for p in others] \
            + [c(d.S(n, p)) for p in others] + ([c(d.E(n, d.pmax))] if others else [])
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def detail_table(d):
    head = ("| N | Режим | Процессов | Узлов | Запусков | t медиана, с | t средн., с | t мин., с | σ, с | "
            "Доля обменов | GFLOPS | Верификация | Макс. отн. погрешность |\n" + "|---" * 13 + "|")
    lines = [head]
    for r in sorted(d.rows, key=lambda r: (r["n"], r["threads"])):
        p = r["threads"]
        mode = "sequential" if p == 0 else "MPI"
        nodes = d.nodes.get((r["n"], p), "-")
        share = d.comm_share(r["n"], p) if p > 0 else None
        share = "-" if share is None else f"{100 * share:.1f}%"
        lines.append(f"| {r['n']} | {mode} | {p} | {nodes} | {r['runs']} | {r['median']:.4f} | {r['mean']:.4f} | "
                     f"{r['min']:.4f} | {r['std']:.4f} | {share} | {r['gflops']:.3f} | {r['verified']} | "
                     f"{r['max_rel_err']:.1e} |")
    return "\n".join(lines)


def span(vals):
    lo, hi = min(vals), max(vals)
    return c(lo) if abs(hi - lo) < 0.005 else f"{c(lo)}–{c(hi)}"


def placement_text(d):
    parts = []
    for p in d.P:
        k = d.nodes_for(p)
        if k:
            parts.append(f"{p} — {k} {'узел' if k == 1 else ('узла' if k < 5 else 'узлов')}")
    return ("Размещение процессов по узлам: " + ", ".join(parts) + ".") if parts else ""


def text_table(d):
    if len(d.P) < 2:
        return ""
    parts = []
    mids = [p for p in d.P[1:-1]] or d.P[1:]
    near = [p for p in mids if all(d.E(n, p) >= 0.85 for n in d.N)]
    far = [p for p in mids if p not in near]
    if near:
        parts.append(f"На {', '.join(str(p) for p in near)} процессах ускорение близко к линейному на всех размерах "
                     f"(при {near[-1]} процессах S = {span([d.S(n, near[-1]) for n in d.N])}).")
    for p in far:
        parts.append(f"На {p} процессах S = {span([d.S(n, p) for n in d.N])}, эффективность "
                     f"{span([d.E(n, p) for n in d.N])}.")
    es = [d.E(n, d.pmax) for n in d.N]
    parts.append(f"На {d.pmax} процессах эффективность равна {span(es)}.")
    parts.append(bottleneck_text(d))
    nmax = d.N[-1]
    g0, g1 = d.gflops(nmax, d.p0), d.gflops(nmax, d.pmax)
    parts.append(f"Производительность при N = {nmax} {'выросла' if g1 > g0 else 'изменилась'} с ≈{c(g0, 2)} GFLOPS на {d.p0} процессе "
                 f"до ≈{c(g1, 2)} GFLOPS на {d.pmax} процессах.")
    return " ".join(parts)


def bottleneck_text(d):
    """Откуда потери эффективности: пересылки (малые N) или вычислительная часть внутри узла (большие N).
    Строится по измеренной доле обменов и по времени вычислений, без предположений о железе."""
    if not d.comp or len(d.P) < 2:
        return ""
    parts = []
    shares = {n: d.comm_share(n, d.pmax) for n in d.N}
    if any(v is None for v in shares.values()):
        return ""
    small = [n for n in d.N if shares[n] >= 0.2]
    big = [n for n in d.N if shares[n] < 0.12]
    if small:
        parts.append(f"При малых N потери дают пересылки: при N = {', '.join(str(n) for n in small)} на {d.pmax} процессах "
                     f"обмены (`MPI_Bcast` матрицы B, `MPI_Scatterv`, `MPI_Gatherv`) занимают "
                     f"{c(100 * min(shares[n] for n in small), 1)}–{c(100 * max(shares[n] for n in small), 1)} % времени, "
                     "а вычислений на процесс почти нет.")
    if not big:
        return " ".join(parts)
    parts.append(f"При N ≥ {big[0]} пересылки занимают не больше {c(100 * max(shares[n] for n in big), 1)} % времени, "
                 "и основная потеря — в самой вычислительной части.")
    single = [p for p in d.P if d.nodes_for(p) == 1 and p > d.p0]
    if single:
        ps = single[-1]
        ec = [d.E_comp(n, ps) for n in big if d.E_comp(n, ps) is not None]
        if ec:
            parts.append(f"На одном узле при {ps} процессах эффективность одних только вычислений (без пересылок) равна "
                         f"{span(ec)}: процессы одного узла делят его память и кэш, а тройной цикл ограничен скоростью "
                         "обращений к памяти, а не арифметикой.")
        g_small = d.gflops(d.N[0], d.p0)
        g_big = [d.gflops(n, d.p0) for n in d.N if n >= 1600] or [d.gflops(d.N[-1], d.p0)]
        if g_small > 1.3 * max(g_big):
            parts.append(f"Это видно и на одном процессе: {c(g_small)} GFLOPS при N = {d.N[0]}, когда матрицы почти "
                         f"целиком в кэше, и {span(g_big)} GFLOPS при N ≥ 1600.")
        steps = []
        for a, b in zip(d.P, d.P[1:]):
            ka, kb = d.nodes_for(a) or 1, d.nodes_for(b) or 1
            if a >= ps and kb > ka:
                r = [d.comp[(n, a)] / d.comp[(n, b)] for n in big if (n, a) in d.comp and (n, b) in d.comp]
                if r:
                    steps.append(f"{a}→{b} (с {ka} на {kb} узла) — в {span(r)} раза")
        if steps:
            parts.append("Когда процессы добавляются вместе с новыми узлами, каждый узел приносит свою память, и время "
                         "вычислений сокращается почти ровно пропорционально: " + ", ".join(steps) + ".")
    return " ".join(parts)


def text_fig1(d):
    if len(d.P) < 2:
        return ""
    steps = []
    for a, b in zip(d.P, d.P[1:]):
        ratios = [d.med[(n, a)] / d.med[(n, b)] for n in d.N]
        steps.append((a, b, sum(ratios) / len(ratios)))
    txt = ("В логарифмической шкале кривые для разного числа процессов идут примерно параллельно (время ~ N³). "
           "Среднее по размерам сокращение времени при увеличении числа процессов: "
           + ", ".join((f"{a}→{b} в {c(r)} раза" if r >= 1 else f"{a}→{b} замедление в {c(1 / r)} раза") for a, b, r in steps) + ".")
    if len(steps) >= 2 and steps[-1][2] < steps[0][2] - 0.2:
        txt += " Выигрыш от каждого следующего увеличения числа процессов уменьшается."
    if any(r < 1 for _, _, r in steps):
        txt += (" Замедление при большем числе процессов возможно, если процессов больше, чем доступных ядер, "
                "или если накладные расходы на пересылки превышают выигрыш от вычислений (малые N).")
    return txt


def text_fig2(d):
    if len(d.P) < 2:
        return ""
    nmax = d.N[-1]
    ps = d.P[1:]
    vals = ", ".join(f"{c(d.S(nmax, p))} на {p}" for p in ps)
    txt = f"При N = {nmax} ускорение составляет {vals} процессах (идеал — равное числу процессов)."
    eff = [d.E(nmax, p) for p in ps]
    if eff[-1] < 0.85:
        share = d.comm_share(nmax, ps[-1]) if d.comp else None
        txt += f" Отставание от идеала растёт с числом процессов (эффективность {c(eff[-1])} на {ps[-1]})"
        if share is not None and share < 0.12:
            txt += (f", хотя пересылки при N = {nmax} занимают лишь {c(100 * share, 1)} % времени: основная потеря — "
                    "в вычислительной части внутри узла (см. пояснение к таблице).")
        else:
            txt += " — это накладные расходы на пересылки данных."
    elif max(eff) > 1.05:
        txt += (" Эффективность выше 1 (сверхлинейное ускорение) объясняется кэшем: на каждый процесс приходится меньше строк A, "
                "и рабочий набор лучше помещается в кэш процессора.")
    return txt


def text_fig3(d):
    if len(d.P) < 2 or len(d.N) < 2:
        return ""
    lo, hi = d.E(d.N[0], d.pmax), d.E(d.N[-1], d.pmax)
    if hi > lo:
        return (f"Эффективность на {d.pmax} процессах растёт с размером матрицы ({c(lo)} при N = {d.N[0]} и {c(hi)} при N = {d.N[-1]}): "
                "объём вычислений растёт как N³, а пересылаемых данных — как N², поэтому доля коммуникаций падает.")
    return (f"Эффективность на {d.pmax} процессах: {c(lo)} при N = {d.N[0]} и {c(hi)} при N = {d.N[-1]}.")


def text_fig4(d):
    if not d.comp or len(d.P) < 2:
        return ""
    lo, hi = d.comm_share(d.N[-1], d.pmax), d.comm_share(d.N[0], d.pmax)
    if lo is None or hi is None:
        return ""
    return (f"Доля обменов = 1 − t_вычисл / t_общ, где t_вычисл — максимальное по процессам время умножения своих строк. "
            f"На {d.pmax} процессах она составляет {c(100 * hi, 1)}% при N = {d.N[0]} и {c(100 * lo, 1)}% при N = {d.N[-1]}: "
            "пересылается O(N²) данных, а считается O(N³), поэтому для больших матриц распараллеливание окупается лучше.")


def openmp_comparison(d):
    if not os.path.exists(RES_LAB2):
        return ""
    try:
        rows2 = aggregate(RES_LAB2)
    except Exception:
        return ""
    nmax = d.N[-1]
    cand = [r for r in rows2 if r["strategy"] == "parallel_openmp" and r["cores"] == 0 and r["n"] == nmax
            and r["speedup"] is not None and r["threads"] <= d.pmax]
    if not cand:
        return ""
    o = max(cand, key=lambda r: r["threads"])
    if (nmax, o["threads"]) not in d.med:
        return ""
    m = d.S(nmax, o["threads"])
    rel = "близко к" if abs(m - o["speedup"]) / o["speedup"] < 0.15 else ("выше, чем" if m > o["speedup"] else "ниже, чем")
    return (f" Для сравнения: ускорение MPI на {o['threads']} процессах ({c(m)}) {rel} ускорение OpenMP из лабораторной работы 2 "
            f"на {o['threads']} потоках ({c(o['speedup'])}; замеры OpenMP шли на ноутбуке, MPI — на кластере). "
            "MPI дополнительно тратит время на пересылки, которые OpenMP не нужны благодаря общей памяти, зато MPI "
            "масштабируется за пределы одного узла (распределённая память).")


def scaling_summary(d):
    """Сравнение удвоения числа процессов внутри узла и вместе с узлами (по времени вычислений, большие N)."""
    if not d.comp:
        return ""
    big = [n for n in d.N if (d.comm_share(n, d.pmax) or 1) < 0.12]
    inside, across = [], []
    for a, b in zip(d.P, d.P[1:]):
        if b != 2 * a:
            continue
        r = [d.comp[(n, a)] / d.comp[(n, b)] for n in big if (n, a) in d.comp and (n, b) in d.comp]
        if not r:
            continue
        ka, kb = d.nodes_for(a) or 1, d.nodes_for(b) or 1
        if ka == kb == 1 and a >= 4:
            inside.append((a, b, sum(r) / len(r)))
        elif kb > ka:
            across.append((a, b, sum(r) / len(r)))
    if not inside or not across:
        return ""
    ai = inside[-1]
    ac = sum(x[2] for x in across) / len(across)
    if ai[2] > ac - 0.2:
        return ""
    return (f" Для больших матриц (N ≥ {big[0]}) пересылки почти не влияют на время, а масштабирование упирается в общую память "
            f"узла: удвоение {ai[0]}→{ai[1]} процессов на одном узле ускоряет вычисления в среднем в {c(ai[2])} раза, "
            f"а удвоение вместе с числом узлов — в {c(ac)} раза.")


def text_conclusion(d, where):
    nmax = d.N[-1]
    checked = [r for r in d.rows if r["verified"] != "NA"]
    if checked and all(r["verified"] == "OK" for r in checked):
        t = ("Распределённая версия на MPI корректна: результаты всех проверенных запусков совпали с эталоном NumPy "
             f"(максимальная относительная погрешность {max(r['max_rel_err'] for r in checked):.1e}). ")
    elif checked:
        t = "Для части конфигураций верификация НЕ пройдена — см. приложение. "
    else:
        t = "Верификация ещё не выполнялась (`NA`): запустите `python3 tools/verify_lab3.py`. "
    t += f"Программа запущена на платформе: {where}. "
    if len(d.P) > 1:
        ps = d.P[1:]
        t += (f"При N = {nmax} получено ускорение "
              + ", ".join(f"{c(d.S(nmax, p))} на {p}" for p in ps)
              + f" процессах (эффективность {c(d.E(nmax, d.pmax))} на {d.pmax}).")
        t += scaling_summary(d)
        t += openmp_comparison(d)
    return t.strip()


CODE = """```cpp
// rank 0 прочитал A и B; всем процессам нужна B целиком, а A - только свои строки
MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Scatterv(A, counts, displs, MPI_DOUBLE, localA, myRows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
multiplyRows(localA, B, localC, N, myRows);          // тот же тройной цикл, что в л/р №1
MPI_Gatherv(localC, myRows * N, MPI_DOUBLE, C, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
```"""


def write_report(d, figs, args, env):
    names = {"time_vs_size.png": "Время выполнения от размера матрицы для разного числа процессов",
             "speedup_vs_procs.png": f"Ускорение S = T({d.p0}) / T(p) от числа процессов (пунктир — идеальное линейное)",
             "efficiency_vs_procs.png": "Эффективность E = S / p от числа процессов для разных размеров матрицы",
             "comm_share_vs_procs.png": "Доля времени на пересылки данных от числа процессов"}
    texts = {"time_vs_size.png": text_fig1(d), "speedup_vs_procs.png": text_fig2(d),
             "efficiency_vs_procs.png": text_fig3(d), "comm_share_vs_procs.png": text_fig4(d)}
    headings = {"time_vs_size.png": "Время от размера матрицы", "speedup_vs_procs.png": "Ускорение от числа процессов",
                "efficiency_vs_procs.png": "Эффективность от числа процессов",
                "comm_share_vs_procs.png": "Доля коммуникаций"}
    graphs = []
    for i, f in enumerate(figs, 1):
        block = f"### {headings[f]}\n\n![{names[f]}](figures/{f})\n\n*Рисунок {i} — {names[f]}*"
        if texts[f]:
            block += f"\n\n{texts[f]}"
        graphs.append(block)

    where = args.platform or (f"узел {env['host']}" if env.get("host") else "локальная машина")
    env_lines = [f"- **Платформа:** {where}"]
    if env.get("scheduler"):
        env_lines.append(f"- **Планировщик:** {env['scheduler']}; узлы задачи: {env.get('nodes', '-')}")
    one_node = [p for p in d.P if d.nodes_for(p) == 1]
    per_node = (f"; в задаче занималось по {one_node[-1]} процессов на узел"
                if one_node and (d.nodes_for(d.pmax) or 1) > 1 else "")
    env_lines.append(f"- **Процессор узла:** {env['cpu']} ({env['cores_per_node']} ядер в узле{per_node})")
    if env.get("distro") or env.get("os"):
        env_lines.append(f"- **ОС:** {env.get('distro') or ''} {('(' + env['os'] + ')') if env.get('os') else ''}".rstrip())
    env_lines.append(f"- **MPI:** {env['mpi']}; запуск: `{env.get('mpirun', 'mpirun')}`")
    env_lines.append(f"- **Компилятор:** {env['compiler']}, `mpicxx -O3`")
    env_lines.append("- **Верификация/анализ:** Python 3, NumPy, Matplotlib (на своей машине)")

    text = f"""# Лабораторная работа №3

## Параллельное умножение квадратных матриц на MPI

**Курс:** Параллельное программирование, 2026 **Дедлайн:** 10.11.2026

**Студент:** {STUDENT} **Группа:** {GROUP}

---

## Задание

Модифицировать программу из л/р №1 для параллельной работы по технологии MPI.

- Провести серию экспериментов с разными размерами матриц ({", ".join(str(n) for n in d.N)}).
- Провести серию экспериментов с разным количеством вычислительных ядер ({", ".join(str(p) for p in d.P)} процессов).
- Запустить параллельную версию на суперкомпьютере «Сергей Королёв».

## Среда выполнения

{chr(10).join(env_lines)}

## Реализация

Код — `src/matmul_mpi.cpp` (формат входа/выхода и тройной цикл те же, что в л/р №1). Строки матрицы A распределяются
между процессами (`MPI_Scatterv`), матрица B рассылается всем (`MPI_Bcast`), каждый процесс считает свои строки C,
результат собирается на нулевом процессе (`MPI_Gatherv`). Число строк на процесс отличается не более чем на одну,
поэтому N не обязано делиться на число процессов.

{CODE}

Время замеряется `MPI_Wtime` между двумя `MPI_Barrier` и включает все пересылки (чтение и запись файлов не входят).
Отдельно замеряется время чистого вычисления (максимум по процессам), из него получается доля обменов.
Программа написана на C++98 без сторонних библиотек, чтобы собираться старыми компиляторами кластера.

Верификация — `tools/verify.py` (сравнение с `A @ B` из NumPy, допуск 1e-5, первый запуск каждой конфигурации).
На кластере NumPy нет, поэтому туда передавались готовые входные матрицы, а оттуда — последовательный результат
и MD5 всех выходных матриц: каждый элемент C вычисляется одним и тем же циклом, поэтому выход MPI побитово совпадает
с последовательным, и проверка последовательного результата через NumPy распространяется на все конфигурации
(`tools/verify_lab3.py`).

### Сборка и запуск

```bash
mpicxx -O3 -o build/matmul_mpi src/matmul_mpi.cpp            # или CMake: цель matmul_mpi
mpirun -np 4 ./build/matmul_mpi data/input_200.txt           # 4 процесса
mpirun -np 1 ./build/matmul_mpi data/input_200.txt seq       # последовательная версия
python3 tools/verify.py data/input_200.txt data/output_200_parallel_mpi_t4.txt

./tools/run_experiments_lab3.sh                              # серия локально
./tools/lab3_bundle.sh                                       # архив для кластера
sbatch tools/slurm_lab3.sbatch                               # серия на «Сергее Королёве» (SLURM)
python3 tools/verify_lab3.py && python3 tools/plot_lab3.py   # проверка и этот отчёт
```

## Эксперименты

- Размеры N: {", ".join(str(n) for n in d.N)}
- Число MPI-процессов (= ядер): {", ".join(str(p) for p in d.P)}; база — последовательная версия и MPI на 1 процессе
- Запусков на конфигурацию: {d.runs}; в таблицах медиана, подробности (среднее, минимум, σ) — в приложении.

{placement_text(d)}

## Результаты

Время (медиана), ускорение S = T({d.p0}) / T(p) и эффективность E = S / p (данные — `data/results_lab3.csv`):

{wide_table(d)}

{text_table(d)}

## Графики

{chr(10).join(chr(10) + b + chr(10) for b in graphs).strip()}

## Выводы

{text_conclusion(d, where)}

## Приложение. Подробные данные

{detail_table(d)}
"""
    with open(os.path.join(REPORT, "README.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-report", action="store_true")
    ap.add_argument("--platform", default="", help="платформа замеров, например: суперкомпьютер «Сергей Королёв»")
    args = ap.parse_args()
    env = load_env()
    comp, nodes = load_extra()
    d = Data(aggregate(RES), comp, nodes)
    try:
        cpn = int(env.get("cores_per_node") or 0)
    except ValueError:
        cpn = 0
    figs = make_plots(d, cpn)
    if not args.skip_report:
        os.makedirs(REPORT, exist_ok=True)
        write_report(d, figs, args, env)
    print("Готово: report/lab3/figures/" + ("" if args.skip_report else " и README.md"))
