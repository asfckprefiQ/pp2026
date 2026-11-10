#!/bin/bash
# Лаба 3: полная серия на суперкомпьютере «Сергей Королёв» (SLURM).
# Сначала собрать на узле входа (на вычислительных узлах нет заголовков stdio.h и т.п.):
#   module load intel/mpi5 && mkdir -p build && mpicxx -O3 -o build/matmul_mpi src/matmul_mpi.cpp
#   sbatch tools/slurm_lab3.sbatch                (из корня распакованного архива, см. tools/lab3_bundle.sh)
#   squeue -u $USER                                - состояние;  вывод - в файле matmul_mpi-<id>.out
#   scancel <id>                                   - снять задачу
#
# 4 узла x 8 ядер = 32 процесса: P = 1, 2, 4, 8 - на одном узле, 16 - на двух, 32 - на четырёх.
# Если многоузловой запуск не работает или узлов не дают, только один узел (P = 1, 2, 4, 8):
#   sbatch --nodes=1 tools/slurm_lab3.sbatch
# Перед запуском проверьте: sinfo (имя раздела и число ядер на узле), module avail 2>&1 | grep -i mpi.
#SBATCH --job-name=matmul_mpi
#SBATCH --partition=batch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --time=02:00:00
#SBATCH --output=matmul_mpi-%j.out

cd "${SLURM_SUBMIT_DIR:-.}"
. tools/cluster_env_lab3.sh

echo "Узлы задачи: $SLURM_JOB_NODELIST, задач на узел: $SLURM_TASKS_PER_NODE"
export SIZES=${SIZES:-"200 400 800 1200 1600 2000"}
export RUNS=${RUNS:-3}
# PROCS не задан - берётся 1 2 4 8 ... до числа выделенных слотов
bash tools/run_experiments_lab3.sh
