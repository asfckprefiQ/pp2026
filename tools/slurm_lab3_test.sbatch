#!/bin/bash
# Лаба 3: короткая пробная задача (1-2 минуты) - проверить сборку, mpirun и размещение процессов
# на двух узлах перед полной серией.
# Сначала собрать на узле входа (на вычислительных узлах нет заголовков stdio.h и т.п.):
#   module load intel/mpi5 && mkdir -p build && mpicxx -O3 -o build/matmul_mpi src/matmul_mpi.cpp
#   sbatch tools/slurm_lab3_test.sbatch   ->   cat matmul_test-<id>.out
#SBATCH --job-name=matmul_test
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:10:00
#SBATCH --output=matmul_test-%j.out

cd "${SLURM_SUBMIT_DIR:-.}"
. tools/cluster_env_lab3.sh

echo "Узлы задачи: $SLURM_JOB_NODELIST, задач на узел: $SLURM_TASKS_PER_NODE"
export PROCS="1 2 8 16" SIZES="200 400" RUNS=1
bash tools/run_experiments_lab3.sh
cat data/results_lab3.csv
