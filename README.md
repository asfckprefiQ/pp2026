# Лабораторные работы — Параллельное программирование, 2026

Тема — параллельное умножение квадратных матриц. Каждая лабораторная работа добавляет новый способ
распараллеливания одной и той же задачи; подробности, код, эксперименты и выводы — в отчёте
соответствующей лабы.

- [Лабораторная работа 1](report/lab1/README.md) — последовательная версия и параллельная версия на `std::thread`.
- [Лабораторная работа 2](report/lab2/README.md) — параллельная версия на OpenMP.
- [Лабораторная работа 4](report/lab4/README.md) — параллельная версия на CUDA.

## Структура репозитория

- `src/main.cpp` — реализация всех стратегий (sequential, std::thread, OpenMP, CUDA)
- `src/cuda_matmul.cu`, `src/cuda_matmul.h` — CUDA-ядра (naive и tiled) и их вызов с хоста
- `tools/` — генерация данных, верификация, агрегация, графики
- `data/` — входные данные (не хранятся в git) и `results.csv` (лаба 1), `results_lab2.csv` (лаба 2), `results_lab4.csv` и `device_lab4.txt` (лаба 4)
- `report/`
  - `lab1/` — отчёт и графики лабы 1
  - `lab2/` — отчёт и графики лабы 2
  - `lab4/` — отчёт и графики лабы 4
- `CMakeLists.txt`

## Сборка и запуск

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
python3 tools/generate.py 200 data/input_200.txt --seed 1
./build/matmul data/input_200.txt          # последовательная версия
./build/matmul data/input_200.txt 4        # 4 потока std::thread
./build/matmul data/input_200.txt 4 omp    # 4 потока OpenMP
./build/matmul data/input_200.txt 16 cuda tiled   # CUDA: блок 16x16 потоков, ядро с shared-памятью
./build/matmul data/input_200.txt 16 cuda naive   # CUDA: блок 16x16 потоков, наивное ядро
python3 tools/verify.py data/input_200.txt data/output_200_sequential.txt
```

Для CUDA нужны видеокарта NVIDIA и CUDA Toolkit (`nvcc`). Без них CMake пропускает CUDA-часть, а лабы 1–2 собираются как обычно.
Для CUDA второй аргумент — сторона блока потоков, четвёртый — ядро (`naive` или `tiled`); имя выходного файла:
`data/output_200_cuda_tiled_b16.txt`.

Полная серия экспериментов (проверка, графики, отчёт лабы 1):

```
pip install numpy matplotlib
./tools/run_experiments.sh
THREADS="0 1 2 4 8" ./tools/run_experiments.sh   # вместе с std::thread
```

Лаба 2 (OpenMP; размеры 200–2000, потоки 1/2/4/8…, серия по числу ядер через `taskset`):

```
./tools/run_experiments_lab2.sh
```

Лаба 4 (CUDA; размеры 200–2000, ядра naive/tiled, конфигурации блока 4x4, 8x8, 16x16, 32x32):

```
./tools/run_experiments_lab4.sh
SIZES="200 400" BLOCKS="8 16" RUNS=1 ./tools/run_experiments_lab4.sh   # быстрый пробный прогон
python3 tools/plot_lab4.py --os "Ubuntu 24.04 (Google Colab)" --compiler "g++ 11.4, nvcc 13.0"   # пересобрать отчёт, если замеры шли на другой машине
```

Если своей видеокарты нет, серию можно выполнить в Google Colab (среда выполнения с GPU T4), затем скачать
`data/results_lab4.csv`, `data/device_lab4.txt` и пересобрать отчёт командой `plot_lab4.py` выше.
