# Лабораторные работы — Параллельное программирование, 2026

**Студент:** Храмов Кирилл **Группа:** 6212-100503D

Тема лаб 1–4 — параллельное умножение квадратных матриц. Каждая лабораторная работа добавляет новый способ
распараллеливания одной и той же задачи. Лаба 5 — программа из Stanford CS149. Подробности, код, эксперименты
и выводы — в отчёте соответствующей лабы.

- [Лабораторная работа 1](report/lab1/README.md) — последовательная версия и параллельная версия на `std::thread`.
- [Лабораторная работа 2](report/lab2/README.md) — параллельная версия на OpenMP.
- [Лабораторная работа 3](report/lab3/README.md) — параллельная версия на MPI, запуск на суперкомпьютере «Сергей Королёв».
- [Лабораторная работа 4](report/lab4/README.md) — параллельная версия на CUDA.
- [Лабораторная работа 5](report/lab5/README.md) — параллельное построение множества Мандельброта на `std::thread` (Program 1 из [stanford-cs149/asst1](https://github.com/stanford-cs149/asst1)).

## Структура репозитория

- `src/main.cpp` — последовательная версия, `std::thread`, OpenMP и CUDA (лабы 1, 2, 4)
- `src/matmul_mpi.cpp` — MPI-версия (лаба 3), переносимый C++98 для старых компиляторов кластера
- `src/cuda_matmul.cu`, `src/cuda_matmul.h` — CUDA-ядра (naive и tiled) и их вызов с хоста
- `asst1/prog1_mandelbrot_threads` — Program 1 из asst1 (лаба 5)
- `tools/` — генерация данных, верификация, серии экспериментов, графики и отчёты, скрипты заданий для кластера (SLURM и PBS)
- `data/` — результаты замеров: `results.csv` (лаба 1), `results_lab2.csv`, `results_lab3.csv` и `env_lab3.txt`,
  `results_lab4.csv` и `device_lab4.txt`, `results_lab5.csv`; входные матрицы `input_*.txt` в git не хранятся
  (генерируются `tools/generate.py`)
- `report/labN/` — отчёт и графики лабы N
- `CMakeLists.txt`

## Сборка и запуск

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
python3 tools/generate.py 200 data/input_200.txt --seed 1
./build/matmul data/input_200.txt                  # последовательная версия
./build/matmul data/input_200.txt 4                # 4 потока std::thread
./build/matmul data/input_200.txt 4 omp            # 4 потока OpenMP
mpirun -np 4 ./build/matmul_mpi data/input_200.txt # 4 процесса MPI (если найден MPI)
./build/matmul data/input_200.txt 16 cuda tiled    # CUDA: блок 16x16 потоков, ядро с shared-памятью
./build/matmul data/input_200.txt 16 cuda naive    # CUDA: блок 16x16 потоков, наивное ядро
python3 tools/verify.py data/input_200.txt data/output_200_sequential.txt
```

MPI и CUDA необязательны: без них CMake пропускает соответствующую часть, остальные лабы собираются как обычно.
Для CUDA второй аргумент — сторона блока потоков, четвёртый — ядро (`naive` или `tiled`); имя выходного файла:
`data/output_200_cuda_tiled_b16.txt`.

### Серии экспериментов

Нужны `pip install numpy matplotlib`. Каждый скрипт проверяет результаты через NumPy, строит графики и пересобирает отчёт.

```bash
./tools/run_experiments.sh          # лаба 1 (THREADS="0 1 2 4 8" — вместе с std::thread)
./tools/run_experiments_lab2.sh     # лаба 2: OpenMP, потоки 1/2/4/8, ядра через taskset
./tools/run_experiments_lab3.sh     # лаба 3: MPI локально (PROCS="1 2 4" SIZES="200 400" RUNS=1 — пробный прогон)
./tools/run_experiments_lab4.sh     # лаба 4: CUDA, ядра naive/tiled, блоки 4x4..32x32
bash tools/run_experiments_lab5.sh && python3 tools/plot_lab5.py   # лаба 5
```

Лаба 4 без своей видеокарты выполнялась в Google Colab (GPU T4): скачать `data/results_lab4.csv`, `data/device_lab4.txt`
и пересобрать отчёт `python3 tools/plot_lab4.py --os "Ubuntu 24.04 (Google Colab)" --compiler "g++ 11.4, nvcc 13.0"`.

### Лаба 3 на суперкомпьютере «Сергей Королёв»

Вход — `sk.ssau.ru` (только из сети университета или через его VPN), задания ставятся через SLURM (`sbatch`);
для Torque/PBS (`qsub`) есть `tools/pbs_lab3*.pbs`. NumPy на кластере не требуется: входные матрицы
готовятся здесь, а проверка выполняется после копирования результатов обратно. Пошаговая инструкция —
[docs/lab3_cluster.md](docs/lab3_cluster.md).

```bash
./tools/lab3_bundle.sh                                    # lab3_bundle.tar.gz: исходник, скрипты, входные матрицы
scp lab3_bundle.tar.gz <логин>@sk.ssau.ru:
# на кластере:
mkdir -p pp2026 && tar xzf lab3_bundle.tar.gz -C pp2026 && cd pp2026
sbatch tools/slurm_lab3_test.sbatch                       # пробная задача на 1-2 минуты
sbatch tools/slurm_lab3.sbatch                            # полная серия: 1..32 процесса, N = 200..2000
# обратно на свою машину:
scp -r <логин>@sk.ssau.ru:pp2026/data/results_lab3.csv <логин>@sk.ssau.ru:pp2026/data/env_lab3.txt data/
scp -r <логин>@sk.ssau.ru:pp2026/data/lab3_out data/
python3 tools/verify_lab3.py
python3 tools/plot_lab3.py --platform "суперкомпьютер «Сергей Королёв» (Самарский университет)"
```
