#!/usr/bin/env bash
# Лаба 3: архив для переноса на суперкомпьютер - исходник MPI-версии, скрипты и входные матрицы.
#   ./tools/lab3_bundle.sh                       -> lab3_bundle.tar.gz в корне репозитория
#   scp lab3_bundle.tar.gz <логин>@sk.ssau.ru:
#   (на кластере) mkdir -p pp2026 && tar xzf lab3_bundle.tar.gz -C pp2026 && cd pp2026 && sbatch tools/slurm_lab3.sbatch
# Входные матрицы генерируются здесь (NumPy), потому что на кластере NumPy может не оказаться.
set -eu
cd "$(dirname "$0")/.."
SIZES=${SIZES:-"200 400 800 1200 1600 2000"}
PYTHON=${PYTHON:-python3}

FILES="src/matmul_mpi.cpp tools/run_experiments_lab3.sh tools/cluster_env_lab3.sh tools/slurm_lab3.sbatch tools/slurm_lab3_test.sbatch tools/pbs_lab3.pbs tools/pbs_lab3_test.pbs tools/verify.py tools/generate.py"
for N in $SIZES; do
    IN=data/input_${N}.txt
    [ -f "$IN" ] || $PYTHON tools/generate.py "$N" "$IN" --seed "$N"
    FILES="$FILES $IN"
done
tar czf lab3_bundle.tar.gz $FILES
ls -lh lab3_bundle.tar.gz
