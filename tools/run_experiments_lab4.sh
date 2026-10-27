#!/usr/bin/env bash
# Лаба 4 (CUDA): серия экспериментов по размерам матриц и конфигурациям сетки блоков.
#
#   ./tools/run_experiments_lab4.sh
#   SIZES="200 400" BLOCKS="8 16" RUNS=1 ./tools/run_experiments_lab4.sh
#   KERNELS="tiled" ./tools/run_experiments_lab4.sh      # только ядро с shared-памятью
#   BIN=build/matmul KEEP=1 PYTHON=python ./tools/run_experiments_lab4.sh
#
# Для каждого N: sequential (база T1 на CPU), затем для каждого ядра (naive, tiled) и каждой стороны блока B
# запускается CUDA-версия с блоком B x B потоков и сеткой ceil(N/B) x ceil(N/B) блоков.
# Результаты: data/results_lab4.csv (колонка threads = сторона блока B).
# Нужны: видеокарта NVIDIA, CUDA Toolkit (nvcc), g++, CMake (или только nvcc).
set -euo pipefail
cd "$(dirname "$0")/.."

SIZES=${SIZES:-"200 400 800 1200 1600 2000"}
BLOCKS=${BLOCKS:-"4 8 16 32"}
KERNELS=${KERNELS:-"naive tiled"}
RUNS=${RUNS:-3}
SEQ_RUNS=${SEQ_RUNS:-$RUNS}
PYTHON=${PYTHON:-python3}
RESULTS=data/results_lab4.csv

HAVE_NUMPY=0
if $PYTHON -c "import numpy" 2> /dev/null; then HAVE_NUMPY=1; fi

mkdir -p data
if command -v nvidia-smi > /dev/null; then nvidia-smi -L || true; else echo "nvidia-smi не найден: видеокарта NVIDIA/драйвер недоступны?"; fi

if [ -z "${BIN:-}" ]; then
    if command -v cmake > /dev/null; then
        cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null
        cmake --build build > /dev/null
    elif command -v nvcc > /dev/null; then   # запасной вариант без CMake
        mkdir -p build
        nvcc -O3 -std=c++17 -Xcompiler -fopenmp -DUSE_CUDA -Isrc -o build/matmul src/main.cpp src/cuda_matmul.cu
    else
        echo "Нет ни cmake, ни nvcc: установите CUDA Toolkit"; exit 1
    fi
    BIN=""
    for c in build/matmul build/matmul.exe build/Release/matmul.exe; do
        [ -f "$c" ] && BIN=$c && break
    done
fi
[ -n "$BIN" ] && [ -f "$BIN" ] || { echo "Не найден исполняемый файл matmul"; exit 1; }

echo "n,strategy,threads,cores,run,time_seconds,kernel_seconds,verified,max_rel_err" > "$RESULTS"
echo "Размеры: $SIZES; блоки: $BLOCKS; ядра: $KERNELS; запусков: $RUNS"

# run_cfg N STRATEGY B RUNS    (B=0 - sequential)
run_cfg() {
    local N=$1 STRATEGY=$2 B=$3 NRUNS=$4 IN=data/input_${1}.txt OUT TIME KTIME VERIFIED REL VOUT K
    if [ "$B" -eq 0 ]; then
        OUT=data/output_${N}_sequential.txt
    else
        K=${STRATEGY#cuda_}
        OUT=data/output_${N}_cuda_${K}_b${B}.txt
    fi
    for r in $(seq 1 "$NRUNS"); do
        if [ "$B" -eq 0 ]; then "$BIN" "$IN" 0 > /dev/null
        else "$BIN" "$IN" "$B" cuda "$K" > /dev/null; fi
        TIME=$(awk '/^time_seconds/ {print $2}' "$OUT")
        KTIME=$(awk '/^kernel_seconds/ {print $2}' "$OUT")
        VERIFIED=NA; REL=""
        if [ "$r" -eq 1 ] && [ "$HAVE_NUMPY" -eq 1 ]; then
            if VOUT=$($PYTHON tools/verify.py "$IN" "$OUT"); then VERIFIED=OK; else VERIFIED=FAIL; fi
            echo "    верификация: $VOUT"
            REL=$(echo "$VOUT" | sed -n 's/.*max_rel_err=\([^ ]*\).*/\1/p')
        fi
        echo "$N,$STRATEGY,$B,0,$r,$TIME,$KTIME,$VERIFIED,$REL" >> "$RESULTS"
        if [ "$B" -gt 0 ] && [ ! -f data/device_lab4.txt ]; then   # данные о видеокарте для отчёта
            { grep -E '^(device|sm_count) ' "$OUT" || true; nvcc --version 2> /dev/null | grep release || true
              echo "cpu $(grep -m1 'model name' /proc/cpuinfo 2> /dev/null | sed 's/.*: //' || true)"
              echo "cores $(nproc 2> /dev/null || echo ?)"; } > data/device_lab4.txt
        fi
    done
    echo "  N=$N $STRATEGY block=$B: последний запуск total=${TIME}s kernel=${KTIME:--}s"
    if [ "${KEEP:-0}" != "1" ]; then rm -f "$OUT"; fi
}

for N in $SIZES; do
    IN=data/input_${N}.txt
    if [ ! -f "$IN" ]; then
        [ "$HAVE_NUMPY" -eq 1 ] || { echo "Нет $IN и нет NumPy для генерации: создайте файл на другой машине и скопируйте в data/"; exit 1; }
        $PYTHON tools/generate.py "$N" "$IN" --seed "$N"
    fi
    run_cfg "$N" sequential 0 "$SEQ_RUNS"
    for K in $KERNELS; do
        for B in $BLOCKS; do
            run_cfg "$N" "cuda_$K" "$B" "$RUNS"
        done
    done
done

if [ "$HAVE_NUMPY" -eq 1 ] && $PYTHON -c "import matplotlib" 2> /dev/null; then
    $PYTHON tools/plot_lab4.py
    echo "Готово: data/results_lab4.csv, report/lab4/README.md, report/lab4/figures/"
else
    echo "Готово: data/results_lab4.csv. Графики/отчёт: python3 tools/plot_lab4.py на машине с NumPy и Matplotlib"
fi
