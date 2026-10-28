# asst1 (лаба 5)

Здесь лежит Program 1 (параллельный Mandelbrot на `std::thread`) из учебного репозитория
https://github.com/stanford-cs149/asst1 (Stanford CS149). Исходный код-заготовку (`common/`,
`mandelbrotSerial.cpp`, `main.cpp`, `Makefile`) взят оттуда; изменения сделаны в `mandelbrotThread.cpp`
(разбиение работы, замер времени потоков) и `main.cpp` (ключи `-m`, `-T`, `-c`, `#include <cstring>`).

Сборка: `cd asst1/prog1_mandelbrot_threads && make`
Запуск: `./mandelbrot -t 8 -v 1 -m interleaved -T`  (`-m block|interleaved`, `-v 1|2`)
Эксперименты и отчёт: `bash tools/run_experiments_lab5.sh && python3 tools/plot_lab5.py` (из корня репозитория).
