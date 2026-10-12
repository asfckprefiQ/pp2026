#!/usr/bin/env bash
# Лаба 2 (OpenMP): серия экспериментов по размерам матриц, числу потоков и числу ядер.
#
#   ./tools/run_experiments_lab2.sh
#   THREADS="1 2 4 8" CORES="1 2 4" SIZES="200 400" RUNS=1 ./tools/run_experiments_lab2.sh
#   KEEP=1 PYTHON=python ./tools/run_experiments_lab2.sh
#
# Серии (для каждого N):
#   1) sequential - базовая линия T1;
#   2) OpenMP с T потоками на всех ядрах машины (THREADS);
#   3) OpenMP, потоки = ядра = C, процесс привязан к ядрам 0..C-1 через taskset (CORES).
#      Если taskset недоступен (macOS), серия 3 пропускается: используется фиксированное число ядер машины.
# Результаты: data/results_lab2.csv
set -euo pipefail
cd "$(dirname "$0")/.."

SIZES=${SIZES:-"200 400 800 1200 1600 2000"}
RUNS=${RUNS:-3}
PYTHON=${PYTHON:-python3}
RESULTS=data/results_lab2.csv
NPROC=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)

# по умолчанию: 1, 2, 4, 8 ... до числа логических ядер (и само число ядер, если оно не степень двойки)
default_list() {
    local t=1 out=""
    while [ "$t" -le "$NPROC" ]; do out="$out $t"; t=$((t * 2)); done
    [ $((t / 2)) -lt "$NPROC" ] && out="$out $NPROC"
    echo $out
}
THREADS=${THREADS:-$(default_list)}
CORES=${CORES:-$(default_list)}

HAVE_TASKSET=0
if command -v taskset > /dev/null && taskset -c 0 true 2> /dev/null; then HAVE_TASKSET=1; fi

mkdir -p data
if command -v cmake > /dev/null; then
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null
    cmake --build build > /dev/null
else   # запасной вариант без CMake
    mkdir -p build
    g++ -O3 -std=c++17 -pthread -fopenmp -o build/matmul src/main.cpp
fi
BIN=""
for c in build/matmul build/matmul.exe build/Release/matmul.exe; do
    [ -f "$c" ] && BIN=$c && break
done
[ -n "$BIN" ] || { echo "Не найден исполняемый файл matmul в build/"; exit 1; }

echo "n,strategy,threads,cores,run,time_seconds,verified,max_rel_err" > "$RESULTS"
echo "Логических ядер: $NPROC; потоки: $THREADS; ядра: $CORES; taskset: $HAVE_TASKSET"

# run_cfg N T backend cores verify
run_cfg() {
    local N=$1 T=$2 BACKEND=$3 C=$4 DO_VERIFY=$5
    local IN=data/input_${N}.txt STRATEGY OUT PREFIX="" TIME VERIFIED REL VOUT
    if [ "$T" -eq 0 ]; then
        STRATEGY=sequential;      OUT=data/output_${N}_sequential.txt
    else
        STRATEGY=parallel_openmp; OUT=data/output_${N}_parallel_openmp_t${T}.txt
    fi
    if [ "$C" -gt 0 ]; then PREFIX="taskset -c 0-$((C - 1))"; fi

    for r in $(seq 1 "$RUNS"); do
        $PREFIX "$BIN" "$IN" "$T" "$BACKEND" > /dev/null
        TIME=$(awk '/^time_seconds/ {print $2}' "$OUT")
        VERIFIED=NA; REL=""
        if [ "$r" -eq 1 ] && [ "$DO_VERIFY" -eq 1 ]; then
            if VOUT=$($PYTHON tools/verify.py "$IN" "$OUT"); then VERIFIED=OK; else VERIFIED=FAIL; fi
            echo "    верификация: $VOUT"
            REL=$(echo "$VOUT" | sed -n 's/.*max_rel_err=\([^ ]*\).*/\1/p')
        fi
        echo "$N,$STRATEGY,$T,$C,$r,$TIME,$VERIFIED,$REL" >> "$RESULTS"
    done
    echo "  N=$N strategy=$STRATEGY threads=$T cores=$C: последний запуск ${TIME}s"
    if [ "${KEEP:-0}" != "1" ]; then rm -f "$OUT"; fi
}

for N in $SIZES; do
    IN=data/input_${N}.txt
    [ -f "$IN" ] || $PYTHON tools/generate.py "$N" "$IN" --seed "$N"

    run_cfg "$N" 0 std 0 1
    for T in $THREADS; do
        run_cfg "$N" "$T" omp 0 1
    done
    if [ "$HAVE_TASKSET" -eq 1 ]; then
        for C in $CORES; do
            if [ "$C" -le "$NPROC" ]; then run_cfg "$N" "$C" omp "$C" 0; fi
        done
    fi
done

$PYTHON tools/plot_lab2.py
echo "Готово: data/results_lab2.csv, report/lab2/README.md, report/lab2/figures/"
