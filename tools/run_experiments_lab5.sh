#!/usr/bin/env bash
# Лаба 5: Program 1 (Mandelbrot) из stanford-cs149/asst1. Запуск из корня репозитория.
set -euo pipefail
cd "$(dirname "$0")/.."
DIR=asst1/prog1_mandelbrot_threads
OUT=data/results_lab5.csv
WORK=data/lab5_work
VIEWS="${VIEWS:-1 2}"
MODES="${MODES:-block interleaved}"
THREADS="${THREADS:-1 2 3 4 5 6 7 8 16}"
RUNS="${RUNS:-3}"
mkdir -p "$WORK" data
(cd "$DIR" && make -s)
echo "view,mode,threads,run,serial_ms,thread_ms,speedup,thread_times_ms" > "$OUT"
for v in $VIEWS; do for m in $MODES; do for t in $THREADS; do for r in $(seq 1 "$RUNS"); do
  line=$(cd "$WORK" && "../../$DIR/mandelbrot" -t "$t" -v "$v" -m "$m" -c | grep '^CSV,')
  IFS=, read -r _ vv mm tt s th sp tms <<< "$line"
  echo "$vv,$mm,$tt,$r,$s,$th,$sp,$tms" >> "$OUT"
  echo "view=$v mode=$m threads=$t run=$r speedup=$sp"
done; done; done; done
echo "Готово: $OUT"
