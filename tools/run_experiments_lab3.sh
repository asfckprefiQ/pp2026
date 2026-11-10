#!/usr/bin/env bash
# Лаба 3 (MPI): серия экспериментов по размерам матриц и числу MPI-процессов (= числу ядер).
#
#   ./tools/run_experiments_lab3.sh                                  # локально: P = 1, 2, 4, 8... до числа ядер
#   PROCS="1 2 4" SIZES="200 400" RUNS=1 ./tools/run_experiments_lab3.sh
#   sbatch tools/slurm_lab3.sbatch                                   # суперкомпьютер «Сергей Королёв» (SLURM)
#   qsub tools/pbs_lab3.pbs                                          # то же, если там Torque/PBS
#
# Переменные окружения:
#   SIZES, PROCS, RUNS        - размеры, числа процессов, повторы (по умолчанию 200..2000, 1 2 4 8.., 3)
#   MPIRUN, MPIRUN_ARGS       - команда запуска (mpirun) и её доп. ключи (например "-r ssh" для Intel MPI 4)
#   MPICXX                    - компилятор (mpicxx)
#   PYTHON                    - интерпретатор для верификации (python3); без NumPy верификация откладывается
#   KEEP=1                    - не удалять выходные матрицы
#
# Для каждого N: sequential (база), затем MPI с P процессами. Внутри задания SLURM или PBS процессы размещаются
# по первым P строкам списка слотов задания (P <= ядер узла - один узел, больше - несколько узлов).
# Результаты: data/results_lab3.csv, сведения о платформе: data/env_lab3.txt.
#
# Верификация: выход первого запуска каждой конфигурации сравнивается с A @ B из NumPy (tools/verify.py).
# Если NumPy на машине нет (так бывает на кластере), сохраняются data/lab3_out/output_N_sequential.txt.gz и
# MD5 каждой выходной матрицы. Все версии считают каждый элемент одним и тем же циклом, поэтому результат MPI
# побитово совпадает с последовательным. После копирования на свою машину: python3 tools/verify_lab3.py
set -eu
cd "$(dirname "$0")/.."

SIZES=${SIZES:-"200 400 800 1200 1600 2000"}
RUNS=${RUNS:-3}
PYTHON=${PYTHON:-python3}
MPIRUN=${MPIRUN:-mpirun}
MPIRUN_ARGS=${MPIRUN_ARGS:-}
MPICXX=${MPICXX:-mpicxx}
BIN=build/matmul_mpi
RESULTS=data/results_lab3.csv
ENVF=data/env_lab3.txt
OUTDIR=data/lab3_out

mkdir -p build data "$OUTDIR"

# Список слотов задания (имя узла на каждый процесс): PBS - $PBS_NODEFILE, SLURM - строится из списка узлов
# и SLURM_TASKS_PER_NODE (вида "8(x4)" или "8,4": число задач на каждом узле)
NODEFILE=""
SCHED=""
if [ -n "${PBS_NODEFILE:-}" ] && [ -f "${PBS_NODEFILE:-}" ]; then
    NODEFILE=$PBS_NODEFILE
    SCHED="Torque/PBS, job ${PBS_JOBID:-?}"
elif [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol > /dev/null 2>&1; then
    NODEFILE=$OUTDIR/slurm_nodes
    scontrol show hostnames "${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-}}" > "$OUTDIR/slurm_hosts"
    echo "${SLURM_TASKS_PER_NODE:-1}" | tr ',' '\n' | awk '{
        if (match($0, /\(x[0-9]+\)/)) { n = substr($0, RSTART + 2, RLENGTH - 3); c = substr($0, 1, RSTART - 1) }
        else { n = 1; c = $0 }
        for (i = 0; i < n; i++) print c }' > "$OUTDIR/slurm_counts"
    paste "$OUTDIR/slurm_hosts" "$OUTDIR/slurm_counts" | awk '{ for (i = 0; i < $2; i++) print $1 }' > "$NODEFILE"
    SCHED="SLURM, job $SLURM_JOB_ID"
fi
if [ -n "$NODEFILE" ]; then
    SLOTS=$(wc -l < "$NODEFILE" | tr -d ' ')
else
    SLOTS=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
fi

default_list() {
    local t=1 out=""
    while [ "$t" -le "$SLOTS" ]; do out="$out $t"; t=$((t * 2)); done
    [ $((t / 2)) -lt "$SLOTS" ] && out="$out $SLOTS"
    echo $out
}
PROCS=${PROCS:-$(default_list)}

# Сборка: только mpicxx, без CMake (на кластере CMake может не быть или быть старым)
if ! command -v "$MPICXX" > /dev/null 2>&1 && command -v mpiicpc > /dev/null 2>&1; then MPICXX=mpiicpc; fi
if ! command -v "$MPICXX" > /dev/null 2>&1; then
    echo "Не найден компилятор MPI ($MPICXX). Загрузите модуль: module avail 2>&1 | grep -i mpi; module load <имя>" >&2
    exit 1
fi
if [ ! -x "$BIN" ] || [ src/matmul_mpi.cpp -nt "$BIN" ]; then
    echo "Сборка: $MPICXX -O3 -o $BIN src/matmul_mpi.cpp"
    if ! $MPICXX -O3 -o "$BIN" src/matmul_mpi.cpp; then
        # «Сергей Королёв»: на вычислительных узлах нет системных заголовков (stdio.h), собирать надо на узле входа
        echo "Сборка не удалась. На кластере соберите на узле входа и поставьте задачу снова:" >&2
        echo "  module load intel/mpi5 && mkdir -p build && mpicxx -O3 -o $BIN src/matmul_mpi.cpp" >&2
        exit 1
    fi
fi

# Open MPI (новые версии) не даёт запускать больше процессов, чем ядер, без --oversubscribe (только вне заданий)
EXTRA=""
if [ -z "$NODEFILE" ] && $MPIRUN --version 2>&1 | grep -qi "open mpi" \
        && $MPIRUN --help 2>&1 | grep -q oversubscribe; then
    EXTRA="--oversubscribe"
fi

HAVE_NUMPY=0
if $PYTHON -c "import sys, numpy; sys.exit(0 if sys.version_info >= (3, 6) else 1)" 2> /dev/null; then HAVE_NUMPY=1; fi

# --- сведения о платформе (используются в отчёте: tools/plot_lab3.py) ---
{
    echo "host: $(hostname)"
    echo "date: $(date '+%Y-%m-%d %H:%M')"
    echo "os: $(uname -sr)"
    [ -f /etc/redhat-release ] && echo "distro: $(head -1 /etc/redhat-release)"
    [ -f /etc/os-release ] && echo "distro: $(. /etc/os-release; echo "$PRETTY_NAME")"
    echo "cpu: $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//;s/  */ /g')"
    echo "cores_per_node: $(grep -c ^processor /proc/cpuinfo 2>/dev/null)"
    echo "compiler: $($MPICXX --version 2>&1 | head -1)"
    echo "mpi: $( ($MPIRUN --version 2>&1 || $MPIRUN -V 2>&1) | grep -v '^$' | head -1)"
    echo "mpirun: $MPIRUN $MPIRUN_ARGS"
    if [ -n "$NODEFILE" ]; then
        echo "scheduler: $SCHED"
        echo "nodes: $(sort -u "$NODEFILE" | tr '\n' ' ')"
        echo "slots: $SLOTS"
    fi
} > "$ENVF"
echo "--- платформа ($ENVF) ---"; cat "$ENVF"; echo "-------------------------"

launch() {   # launch P команда...
    local P=$1; shift
    if [ -n "$NODEFILE" ]; then
        if [ "$P" -gt "$SLOTS" ]; then echo "P=$P больше выделенных слотов ($SLOTS)" >&2; return 1; fi
        head -n "$P" "$NODEFILE" > "$OUTDIR/hosts_$P"
        $MPIRUN $MPIRUN_ARGS -machinefile "$OUTDIR/hosts_$P" -np "$P" "$@"
    else
        $MPIRUN $MPIRUN_ARGS $EXTRA -np "$P" "$@"
    fi
}

field() { echo "$1" | sed -n "s/.* $2=\([^ ]*\).*/\1/p"; }
# MD5 только матрицы (первые N+1 строк), без метаданных (время, число процессов), которые у запусков разные
md5of() { head -n $(($(head -n 1 "$1") + 1)) "$1" | { if command -v md5sum > /dev/null; then md5sum | cut -c1-32; else md5 -q; fi; }; }

echo "n,strategy,threads,cores,run,time_seconds,verified,max_rel_err,compute_seconds,nodes,md5" > "$RESULTS"
echo "Слотов: $SLOTS; процессы MPI: $PROCS; размеры: $SIZES; повторов: $RUNS; запуск: $MPIRUN $MPIRUN_ARGS $EXTRA; numpy: $HAVE_NUMPY"

# run_cfg N P   (P=0 - sequential, запускается как 1 процесс)
run_cfg() {
    local N=$1 P=$2 IN=data/input_${1}.txt NP MODE TAG STRAT OUT DEST LINE TIME COMP NODES VERIFIED REL VOUT SUM SUM1="" r
    if [ "$P" -eq 0 ]; then NP=1; MODE=seq; STRAT=sequential; TAG=sequential
    else NP=$P; MODE=mpi; STRAT=parallel_mpi; TAG=parallel_mpi_t$P; fi
    OUT=$OUTDIR/output_${N}_${TAG}.txt
    for r in $(seq 1 "$RUNS"); do
        if [ "$r" -eq 1 ]; then DEST=$OUT; else DEST=none; fi   # матрица пишется только в первом запуске
        LINE=$(launch "$NP" "$BIN" "$IN" "$MODE" "$DEST" | grep '^RESULT') || { echo "Запуск N=$N P=$P завершился с ошибкой"; exit 1; }
        TIME=$(field "$LINE" time); COMP=$(field "$LINE" compute); NODES=$(field "$LINE" nodes)
        VERIFIED=NA; REL=""; SUM=""
        if [ "$r" -eq 1 ]; then
            SUM=$(md5of "$OUT")
            if [ "$HAVE_NUMPY" -eq 1 ]; then
                if VOUT=$($PYTHON tools/verify.py "$IN" "$OUT"); then VERIFIED=OK; else VERIFIED=FAIL; fi
                echo "    верификация: $VOUT"
                REL=$(echo "$VOUT" | sed -n 's/.*max_rel_err=\([^ ]*\).*/\1/p')
            fi
        fi
        echo "$N,$STRAT,$P,$P,$r,$TIME,$VERIFIED,$REL,$COMP,$NODES,$SUM" >> "$RESULTS"
        if [ "$r" -eq 1 ]; then SUM1=$SUM; fi
    done
    echo "  N=$N P=$P ($TAG, узлов: $NODES): последний запуск ${TIME}s"
    # Без NumPy сохраняем последовательный результат (и MPI-результат, если он вдруг отличается) для проверки позже.
    if [ "${KEEP:-0}" = "1" ]; then
        :
    elif [ "$HAVE_NUMPY" -eq 0 ] && { [ "$P" -eq 0 ] || [ "$SUM1" != "$(md5of_seq "$N")" ]; }; then
        gzip -f "$OUT"
    else
        rm -f "$OUT"
    fi
}

md5of_seq() { grep "^$1,sequential,0,0,1," "$RESULTS" | cut -d, -f11; }

for N in $SIZES; do
    IN=data/input_${N}.txt
    if [ ! -f "$IN" ]; then
        if [ "$HAVE_NUMPY" -eq 1 ]; then
            $PYTHON tools/generate.py "$N" "$IN" --seed "$N"
        else
            echo "Нет $IN и нет NumPy для генерации: распакуйте lab3_bundle.tar.gz (tools/lab3_bundle.sh) или скопируйте файл в data/"
            exit 1
        fi
    fi
    run_cfg "$N" 0
    for P in $PROCS; do run_cfg "$N" "$P"; done
done

if [ "$HAVE_NUMPY" -eq 1 ] && $PYTHON -c "import matplotlib" 2> /dev/null; then
    $PYTHON tools/plot_lab3.py
    echo "Готово: $RESULTS, report/lab3/README.md, report/lab3/figures/"
else
    echo "Готово: $RESULTS, $ENVF, $OUTDIR/*.gz"
    echo "Скопируйте их на свою машину и выполните: python3 tools/verify_lab3.py && python3 tools/plot_lab3.py"
fi
