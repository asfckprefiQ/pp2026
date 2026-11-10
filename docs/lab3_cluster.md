# Лаба 3 на суперкомпьютере «Сергей Королёв»: пошаговая инструкция

Источники: инструкции центра [hpc.ssau.ru](https://hpc.ssau.ru/node/19) (вход — [node/20](https://hpc.ssau.ru/node/20),
PuTTY — [node/23](https://hpc.ssau.ru/node/23), MPI — [node/36](https://hpc.ssau.ru/node/36), правила —
[node/16](https://hpc.ssau.ru/node/16)), сообщение одногруппника и рабочий пример
[Dauxdu/pp-2026](https://github.com/Dauxdu/pp-2026) (`scripts/slurm/job.sh`).

Важно: инструкции на hpc.ssau.ru писались в 2011–2015 годах и описывают Torque/PBS (`qsub`). В примере
одногруппника, запущенном на кластере в этом семестре, используется **SLURM** (`sbatch`, раздел `batch`,
модуль `intel/mpi5`, узлы 2 × Intel Xeon X5647 = 8 ядер, CentOS 7). Поэтому основной вариант ниже — SLURM,
а скрипты для PBS оставлены на случай, если `sbatch` на кластере не окажется.

Что понадобится: логин и пароль от личного кабинета (ЛК) университета, Windows 11 (в ней уже есть `ssh` и `scp`),
WSL с Python, NumPy и matplotlib (уже настроен для лаб 1–5).

---

## Шаг 1. VPN университета

Вход на кластер разрешён только из сети университета или с IP-адресов из заявки ([node/20](https://hpc.ssau.ru/node/20)).
Из дома нужен VPN. Перед подключением выключите другие VPN (Cloudflare WARP, Proton и т. п.): они перехватывают маршруты.

### Вариант А (рекомендуется): Ideco Client

1. Откройте <https://vpn.ssau.ru/cabinet/>, войдите логином и паролем от ЛК.
2. Скачайте в кабинете Ideco Client для Windows и установите его.
3. Подключитесь в клиенте с логином и паролем от ЛК (адрес сервера — `vpn.ssau.ru`, если клиент его спросит).

### Вариант Б: встроенный L2TP-VPN Windows (скрипт `Ideco_NGFW_VPN_L2TP.ps1`)

Это запасной вариант, если Ideco Client не ставится или не подключается. Что делает скрипт:

- перезапускает себя от имени администратора;
- **включает правила брандмауэра «Общий доступ к файлам и принтерам»** (входящие SMB и т. п.). Для VPN они не нужны.
  Сейчас на этом компьютере они выключены;
- создаёт 25 L2TP/IPsec-подключений: по одному на каждый адрес шлюза университета. Параметры: общий ключ из скрипта,
  MS-CHAPv2, раздельное туннелирование (через VPN идёт только трафик в сеть университета).

Из дома работает только подключение к публичному адресу шлюза. `vpn.ssau.ru` = `91.222.130.250`, поэтому нужно
подключение **«Ideco NGFW L2TP VPN to 91.222.130.250»**. Адреса `10.*`, `172.16.*`, `192.168.*` доступны только
внутри сети университета.

Запуск (PowerShell, затем подтвердить запрос администратора):

```powershell
powershell -ExecutionPolicy Bypass -File "$env:USERPROFILE\Downloads\Ideco_NGFW_VPN_L2TP.ps1"
```

Подключение: «Параметры → Сеть и Интернет → VPN → Ideco NGFW L2TP VPN to 91.222.130.250 → Подключиться», логин и пароль от ЛК.

Уборка после скрипта (PowerShell от администратора) — удалить лишние 24 подключения и выключить общий доступ к файлам:

```powershell
Get-VpnConnection | Where-Object { $_.Name -like "Ideco NGFW*" -and $_.ServerAddress -ne "91.222.130.250" } | Remove-VpnConnection -Force
```

```powershell
Disable-NetFirewallRule -Group "@FirewallAPI.dll,-28502"
```

### Проверка

```powershell
ssh -o ConnectTimeout=10 <логин>@sk.ssau.ru
```

Запрос пароля значит, что сеть в порядке. Ошибки `Connection closed by remote host` или `kex_exchange_identification`
значат, что сервер отклоняет ваш IP, то есть VPN не подключён. Без VPN порт 22 открывается, но соединение сразу сбрасывается.

---

## Шаг 2. Вход на кластер

**Встроенный ssh (проще всего).** PowerShell или WSL:

```powershell
ssh <логин>@sk.ssau.ru
```

На вопрос о ключе сервера ответьте `yes`. Логин и пароль — от ЛК (по сообщению одногруппника).

**PuTTY (как в [node/23](https://hpc.ssau.ru/node/23)).** Host Name: `sk.ssau.ru`, порт 22. Window → Translation →
кодировка UTF-8. Session → в поле Saved Sessions ввести `sk` → Save → Open. При первом входе принять ключ сервера.
Файлы по [node/20](https://hpc.ssau.ru/node/20) передаются по SFTP, например WinSCP: хост `sk.ssau.ru`, протокол SFTP,
тот же логин.

---

## Шаг 3. Ключ SSH (чтобы не вводить пароль и чтобы Claude мог выполнять команды сам)

Правила центра рекомендуют вход по ключу ([node/16](https://hpc.ssau.ru/node/16)). PowerShell:

```powershell
ssh-keygen -t ed25519
```

На все вопросы — Enter. Затем скопировать открытый ключ на кластер (последний раз вводится пароль):

```powershell
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh <логин>@sk.ssau.ru "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

Короткое имя `sk`: создайте файл `C:\Users\kirill\.ssh\config` (без расширения) с текстом

```
Host sk
    HostName sk.ssau.ru
    User <логин>
```

Проверка: `ssh sk hostname` выводит имя узла без запроса пароля. Если старый сервер не принимает ed25519,
повторите с `ssh-keygen -t rsa -b 4096` и `id_rsa.pub`.

После этого шаги 4–6 можно поручить Claude: он выполнит их через `ssh sk ...`, пока VPN подключён.

---

## Шаг 4. Архив с программой и входными матрицами

На кластере может не быть NumPy, поэтому входные матрицы готовятся здесь. WSL:

```bash
cd /mnt/c/Users/kirill/Desktop/pp2026 && ./tools/lab3_bundle.sh
```

Получится `lab3_bundle.tar.gz` (~83 МБ): `src/matmul_mpi.cpp`, скрипты заданий и `data/input_*.txt` для N = 200…2000.
Отправка (PowerShell, папка репозитория):

```powershell
scp C:\Users\kirill\Desktop\pp2026\lab3_bundle.tar.gz sk:
```

---

## Шаг 5. Запуск на кластере

```bash
mkdir -p pp2026 && tar xzf lab3_bundle.tar.gz -C pp2026 && cd pp2026
```

Разведка: какой планировщик, какие разделы и модули MPI.

```bash
which sbatch qsub; sinfo; module avail 2>&1 | grep -i mpi
```

На кластере (проверено 10.11.2026): SLURM, раздел `batch`, модули `intel/mpi3`, `intel/mpi4`, `intel/mpi5`
(Intel MPI 5.1.3), `openmpi/...`; GCC 4.8.5.

Тяжёлые вычисления на узле входа запрещены: программа запускается только через планировщик
([node/21](https://hpc.ssau.ru/node/21)). А **собирать нужно на узле входа**: на вычислительных узлах нет системных
заголовков (`fatal error: stdio.h: No such file or directory`).

```bash
module load intel/mpi5 && mkdir -p build && mpicxx -O3 -o build/matmul_mpi src/matmul_mpi.cpp
```

### SLURM (есть `sbatch`)

Пробная задача (2 узла × 8 ядер, N = 200 и 400, P = 1, 2, 8, 16, до 10 минут):

```bash
sbatch tools/slurm_lab3_test.sbatch
```

```bash
squeue -u $USER
```

Когда задача пропадёт из очереди: `cat matmul_test-*.out`. Что проверить:

- `Модуль MPI: ...` и `Сборка: mpicxx ...` без ошибок;
- строка `N=400 P=16 (... узлов: 2)`: процессы действительно на двух узлах;
- в CSV в конце у всех строк одинаковый MD5 (последний столбец): MPI-результат совпадает с последовательным.

Полная серия (4 узла × 8 ядер, P = 1, 2, 4, 8, 16, 32, N = 200…2000, по 3 запуска, лимит 2 часа):

```bash
sbatch tools/slurm_lab3.sbatch
```

Ход выполнения: `tail -f matmul_mpi-<номер>.out` (выход — Ctrl+C), снять задачу: `scancel <номер>`.

Если что-то не так:

| Симптом | Что сделать |
| --- | --- |
| `invalid partition` | взять имя раздела из `sinfo`: `sbatch -p <раздел> tools/slurm_lab3.sbatch` |
| требуется проект/аккаунт | `sbatch -A <проект> ...` |
| `Не найден компилятор MPI` | найти модуль в `module avail`, затем `MPI_MODULE=<имя> sbatch tools/slurm_lab3.sbatch` |
| P = 16 зависает или падает (многоузловой запуск не работает) | только один узел, P = 1, 2, 4, 8 (так делал одногруппник): `sbatch --nodes=1 tools/slurm_lab3.sbatch` |
| задача долго стоит в очереди (`PD`) | меньше узлов: `sbatch --nodes=2 ...` (P до 16) |

### Torque/PBS (есть только `qsub`)

```bash
glsaccount
```

Если код проекта есть, впишите его в строку `#PBS -A` в `tools/pbs_lab3_test.pbs` и `tools/pbs_lab3.pbs`
(убрав второй `#`). Затем `qsub tools/pbs_lab3_test.pbs`, состояние — `qstat -a`, вывод — `matmul_test.o<номер>`,
полная серия — `qsub tools/pbs_lab3.pbs`.

Прочее из правил: домашний каталог 10 ГБ (архив и результаты занимают ~0,2 ГБ); в последний рабочий день месяца
с 8:00 до 21:00 — плановые работы, задачи могут быть сняты.

---

## Шаг 6. Результаты обратно, проверка, отчёт

PowerShell, папка репозитория:

```powershell
cd C:\Users\kirill\Desktop\pp2026
```

```powershell
scp sk:pp2026/data/results_lab3.csv sk:pp2026/data/env_lab3.txt data/
```

```powershell
scp -r sk:pp2026/data/lab3_out data/
```

WSL — проверка через NumPy и генерация отчёта `report/lab3/README.md` с графиками:

```bash
cd /mnt/c/Users/kirill/Desktop/pp2026 && python3 tools/verify_lab3.py && python3 tools/plot_lab3.py --platform "суперкомпьютер «Сергей Королёв» (Самарский университет)"
```

Затем коммит и push (`git add data/results_lab3.csv data/env_lab3.txt report/lab3 && git commit && git push`).
