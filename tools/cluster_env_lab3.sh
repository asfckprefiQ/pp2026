# Лаба 3: окружение MPI на «Сергее Королёве» (подключается через "." из slurm_*.sbatch и pbs_*.pbs).
# Загружает первый найденный модуль MPI. Нужный модуль можно задать явно: MPI_MODULE=intel/mpi5 sbatch ...
# Список модулей на кластере: module avail 2>&1 | grep -i mpi
[ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh   # команда module в пакетном режиме

if command -v module > /dev/null 2>&1 || type module > /dev/null 2>&1; then
    for m in ${MPI_MODULE:-intel/mpi5 intel/mpi4 impi/4 impi/3 openmpi}; do
        # старый module (Tcl 3.x) возвращает 0 даже для несуществующего модуля - сначала ищем его в module avail
        module avail "$m" 2>&1 | grep -q "$m" || continue
        if module load "$m" > /dev/null 2>&1 && command -v mpirun > /dev/null 2>&1; then
            echo "Модуль MPI: $m"
            break
        fi
    done
fi

if mpirun -V 2>&1 | grep -qi intel || mpirun --version 2>&1 | grep -qi intel; then
    export MPIRUN_ARGS=${MPIRUN_ARGS:-"-r ssh"}       # Intel MPI: запуск процессов на узлах через ssh
    export I_MPI_FALLBACK_DEVICE=${I_MPI_FALLBACK_DEVICE:-enable}
fi
echo "mpirun: $(command -v mpirun || echo 'не найден')"
