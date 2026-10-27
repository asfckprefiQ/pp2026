#!/usr/bin/env bash
# Лаба 3 (MPI): серия экспериментов по размерам матриц и числу MPI-процессов (= число ядер).
#
#   ./tools/run_experiments_lab3.sh                                   # локально: P = 1, 2, 4, 8... до числа ядер
#   PROCS="1 2 4" SIZES="200 400" RUNS=1 ./tools/run_experiments_lab3.sh
#   LAUNCHER="srun -n" PROCS="1 2 4 8 16 32" ./tools/run_experiments_lab3.sh   # суперкомпьютер (SLURM)
#   KEEP=1  - не удалять выходные матрицы;  PYTHON=python  - другой интерпретатор
#
# Для каждого N: sequential (база T1), затем MPI с P процессами. Результаты: data/results_lab3.csv
# (колонки threads и cores = число процессов P). Если NumPy недоступен, верификация пропускается (NA).
set -euo pipefail
cd "$(dirname "$0")/.."

SIZES=${SIZES:-"200 400 800 1200 1600 2000"}
RUNS=${RUNS:-3}
PYTHON=${PYTHON:-python3}
LAUNCHER=${LAUNCHER:-"mpirun -np"}
RESULTS=data/results_lab3.csv
NPROC=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)

default_list() {
    local t=1 out=""
    while [ "$t" -le "$NPROC" ]; do out="$out $t"; t=$((t * 2)); done
    [ $((t / 2)) -lt "$NPROC" ] && out="$out $NPROC"
    echo $out
}
PROCS=${PROCS:-$(default_list)}

# Open MPI не даёт запускать больше процессов, чем слотов, без --oversubscribe
EXTRA=""
if [ "$LAUNCHER" = "mpirun -np" ] && mpirun --version 2>&1 | grep -qi "open mpi"; then EXTRA="--oversubscribe"; fi

HAVE_NUMPY=0
if $PYTHON -c "import numpy" 2> /dev/null; then HAVE_NUMPY=1; fi

mkdir -p data
if command -v cmake > /dev/null; then
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null
    cmake --build build > /dev/null
else
    mkdir -p build
    mpicxx -O3 -std=c++17 -pthread -fopenmp -DUSE_MPI -o build/matmul src/main.cpp
fi
BIN=""
for c in build/matmul build/matmul.exe build/Release/matmul.exe; do
    [ -f "$c" ] && BIN=$c && break
done
[ -n "$BIN" ] || { echo "Не найден исполняемый файл matmul в build/"; exit 1; }

echo "n,strategy,threads,cores,run,time_seconds,verified,max_rel_err" > "$RESULTS"
echo "Ядер: $NPROC; процессы MPI: $PROCS; запуск: $LAUNCHER $EXTRA; numpy: $HAVE_NUMPY"

# run_cfg N P  (P=0 - sequential)
run_cfg() {
    local N=$1 P=$2 IN=data/input_${1}.txt STRATEGY OUT TIME VERIFIED REL VOUT
    if [ "$P" -eq 0 ]; then
        STRATEGY=sequential;   OUT=data/output_${N}_sequential.txt
    else
        STRATEGY=parallel_mpi; OUT=data/output_${N}_parallel_mpi_t${P}.txt
    fi
    for r in $(seq 1 "$RUNS"); do
        if [ "$P" -eq 0 ]; then "$BIN" "$IN" 0 > /dev/null
        else $LAUNCHER "$P" $EXTRA "$BIN" "$IN" 0 mpi > /dev/null; fi
        TIME=$(awk '/^time_seconds/ {print $2}' "$OUT")
        VERIFIED=NA; REL=""
        if [ "$r" -eq 1 ] && [ "$HAVE_NUMPY" -eq 1 ]; then
            if VOUT=$($PYTHON tools/verify.py "$IN" "$OUT"); then VERIFIED=OK; else VERIFIED=FAIL; fi
            echo "    верификация: $VOUT"
            REL=$(echo "$VOUT" | sed -n 's/.*max_rel_err=\([^ ]*\).*/\1/p')
        fi
        echo "$N,$STRATEGY,$P,$P,$r,$TIME,$VERIFIED,$REL" >> "$RESULTS"
    done
    echo "  N=$N strategy=$STRATEGY P=$P: последний запуск ${TIME}s"
    if [ "${KEEP:-0}" != "1" ]; then rm -f "$OUT"; fi
}

for N in $SIZES; do
    IN=data/input_${N}.txt
    if [ ! -f "$IN" ]; then
        [ "$HAVE_NUMPY" -eq 1 ] || { echo "Нет $IN и нет NumPy для генерации: создайте файл на другой машине и скопируйте в data/"; exit 1; }
        $PYTHON tools/generate.py "$N" "$IN" --seed "$N"
    fi
    run_cfg "$N" 0
    for P in $PROCS; do run_cfg "$N" "$P"; done
done

if [ "$HAVE_NUMPY" -eq 1 ] && $PYTHON -c "import matplotlib" 2> /dev/null; then
    $PYTHON tools/plot_lab3.py
    echo "Готово: data/results_lab3.csv, report/lab3/README.md, report/lab3/figures/"
else
    echo "Готово: data/results_lab3.csv. Графики/отчёт: скопируйте файл на машину с NumPy и Matplotlib и выполните python3 tools/plot_lab3.py"
fi
