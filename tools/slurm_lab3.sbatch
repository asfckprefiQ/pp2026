#!/bin/bash
# Лаба 3: запуск на суперкомпьютере «Сергей Королёв» (SLURM).
#   sbatch tools/slurm_lab3.sbatch
# Перед первым запуском проверьте у себя на кластере (названия отличаются!):
#   module avail         - как называются модули MPI/GCC/CMake/Python
#   sinfo                - партиции;  squeue -u $USER - очередь
# Затем поправьте строки #SBATCH и module load ниже под ваш кластер.
#SBATCH --job-name=matmul-mpi
#SBATCH --output=lab3_%j.log
#SBATCH --ntasks=32              # максимум процессов в серии (должен быть >= max(PROCS))
#SBATCH --time=02:00:00
##SBATCH --partition=<партиция>  # раскомментируйте и впишите, если требуется
##SBATCH --nodes=2               # при необходимости

# module load gcc cmake openmpi python   # <- впишите реальные имена модулей из `module avail`

cd "$SLURM_SUBMIT_DIR"
export PYTHON=${PYTHON:-python3}
export LAUNCHER="srun -n"        # если srun не настроен для MPI, используйте LAUNCHER="mpirun -np"
export PROCS=${PROCS:-"1 2 4 8 16 32"}
export RUNS=${RUNS:-3}

bash tools/run_experiments_lab3.sh
