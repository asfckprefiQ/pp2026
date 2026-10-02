#!/usr/bin/env bash
# Лаба 1: сборка, серия экспериментов, верификация, графики и отчёт.
#
#   ./tools/run_experiments.sh                      # последовательная версия, N = 200..2000
#   THREADS="0 1 2 4 8" ./tools/run_experiments.sh  # + std::thread (0 = последовательная)
#   SIZES="200 400" RUNS=5 KEEP=1 PYTHON=python ./tools/run_experiments.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SIZES=${SIZES:-"200 400 800 1200 1600 2000"}
THREADS=${THREADS:-"0"}
RUNS=${RUNS:-3}
PYTHON=${PYTHON:-python3}
RESULTS=data/results.csv

mkdir -p data
if command -v cmake > /dev/null; then
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null
    cmake --build build > /dev/null
else   # запасной вариант без CMake
    mkdir -p build
    g++ -O3 -std=c++17 -pthread -o build/matmul src/main.cpp
fi
BIN=""
for c in build/matmul build/matmul.exe build/Release/matmul.exe; do
    [ -f "$c" ] && BIN=$c && break
done
[ -n "$BIN" ] || { echo "Не найден исполняемый файл matmul в build/"; exit 1; }

echo "n,strategy,threads,run,time_seconds,verified,max_rel_err" > "$RESULTS"

for N in $SIZES; do
    IN=data/input_${N}.txt
    [ -f "$IN" ] || $PYTHON tools/generate.py "$N" "$IN" --seed "$N"

    for T in $THREADS; do
        if [ "$T" -eq 0 ]; then
            STRATEGY=sequential;       OUT=data/output_${N}_sequential.txt
        else
            STRATEGY=parallel_threads; OUT=data/output_${N}_parallel_threads_t${T}.txt
        fi

        for r in $(seq 1 "$RUNS"); do
            "$BIN" "$IN" "$T"
            TIME=$(awk '/^time_seconds/ {print $2}' "$OUT")
            VERIFIED=NA; REL=""
            if [ "$r" -eq 1 ]; then   # проверяем первый запуск каждой конфигурации
                if VOUT=$($PYTHON tools/verify.py "$IN" "$OUT"); then VERIFIED=OK; else VERIFIED=FAIL; fi
                echo "  верификация N=$N threads=$T: $VOUT"
                REL=$(echo "$VOUT" | sed -n 's/.*max_rel_err=\([^ ]*\).*/\1/p')
            fi
            echo "$N,$STRATEGY,$T,$r,$TIME,$VERIFIED,$REL" >> "$RESULTS"
        done
        [ "${KEEP:-0}" = "1" ] || rm -f "$OUT"
    done
done

$PYTHON tools/plot.py
echo "Готово: data/results.csv, report/lab1/README.md, report/lab1/figures/"
