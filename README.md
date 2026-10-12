# Лабораторные работы — Параллельное программирование, 2026

Тема — параллельное умножение квадратных матриц. Каждая лабораторная работа добавляет новый способ
распараллеливания одной и той же задачи; подробности, код, эксперименты и выводы — в отчёте
соответствующей лабы.

- [Лабораторная работа 1](report/lab1/README.md) — последовательная версия и параллельная версия на `std::thread`.
- [Лабораторная работа 2](report/lab2/README.md) — параллельная версия на OpenMP.

## Структура репозитория

- `src/main.cpp` — реализация всех стратегий (sequential, std::thread, OpenMP)
- `tools/` — генерация данных, верификация, агрегация, графики
- `data/` — входные данные (не хранятся в git) и `results.csv` (лаба 1), `results_lab2.csv` (лаба 2)
- `report/`
  - `lab1/` — отчёт и графики лабы 1
  - `lab2/` — отчёт и графики лабы 2
- `CMakeLists.txt`

## Сборка и запуск

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
python3 tools/generate.py 200 data/input_200.txt --seed 1
./build/matmul data/input_200.txt          # последовательная версия
./build/matmul data/input_200.txt 4        # 4 потока std::thread
./build/matmul data/input_200.txt 4 omp    # 4 потока OpenMP
python3 tools/verify.py data/input_200.txt data/output_200_sequential.txt
```

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
