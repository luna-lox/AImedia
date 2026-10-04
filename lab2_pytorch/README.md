# ПР2. Введение в PyTorch

| Часть | Файл | Результат |
|---|---|---|
| Разделы 1–6 (tensor, autograd, DataLoader, три уровня API) | [`ПР2_Введение_в_PyTorch.ipynb`](ПР2_Введение_в_PyTorch.ipynb) | все ячейки выполнены |
| Задание 1 — MNIST, < 15 000 параметров, ≥ 95% на тесте за 5 эпох | там же, разделы 7–8 | **98.99%** на тесте, 5 058 параметров |
| Задание 2 — FashionMNIST, ≥ 88.5% на тесте | [`02_hw_fmnist_classification.ipynb`](02_hw_fmnist_classification.ipynb) | **92.74%** на тесте (97.03% на трейне), посылка — `submission_dict_fmnist_task_1.json` |
| Задание 3 — головоломка с тремя свёртками | [`task3/03_kainet_reconstruction.ipynb`](task3/03_kainet_reconstruction.ipynb) | восстановлен фильтр; MSE выходов = 0 на 1000 парах; [OK — полное решение](https://contest.yandex.ru/contest/75233/run-report/166717004/) |

## MNIST (задание 1)

`nn.Sequential` из трёх блоков `Conv3x3 → BatchNorm → ReLU → MaxPool` (1→8→16→16 каналов)
и одного `Linear(144, 10)`: **5 058 параметров**. Adam, `lr=3e-3`, batch 32, 5 эпох.

## FashionMNIST (задание 2)

4 слоя с весами: `Conv(1→32) → Conv(32→64) → Linear(3136→128) → Linear(128→10)`
с BatchNorm, MaxPool и Dropout 0.3. Adam + OneCycleLR (`max_lr=3e-3`), 10 эпох.

## Запуск локально

Блокноты рассчитаны на Colab, но работают и локально (на CPU):

```bash
pip install -r requirements.txt
```

На macOS с Python с python.org скачивание датасетов может падать с
`CERTIFICATE_VERIFY_FAILED` — помогает `export SSL_CERT_FILE=$(python3 -m certifi)`.

## KaiNet (задание 3)

Решение и объяснение неоднозначности порядка — в `task3/03_kainet_reconstruction.ipynb`.
Для контеста подготовлены `task3/reconstructed_algos.csv` и альтернативный
`task3/reconstructed_algos_alternative.csv`. Оба точно воспроизводят все выходные матрицы,
но проверка самих фильтров может требовать один конкретный порядок.
Повторный запуск: `python3 task3/reconstruct.py` (numpy, scipy, Pillow).
Исходный [архив задачи A](https://disk.yandex.ru/d/ECLHqVyX-F5hRA) нужно распаковать в `task3/data` (файлы PNG, TXT и `algos.csv` непосредственно в этой папке). Данные не включены в репозиторий. Полный отчёт — `task3/validation.json`.

Основной CSV принят: [посылка №166717004, OK — полное решение](https://contest.yandex.ru/contest/75233/run-report/166717004/), 28 сентября 2026 года.

## Дополнительная задача B — Operation: EchoTrace fixed

Поиск шести похожих изображений для каждого из 9605 файлов: ResNet-50 + косинусное сходство.
Код и описание — в `task_b/`, файл для контеста — `task_b/submission.csv`.
Оценка закрытого контеста пока не получена; прохождение порога 0.5 не подтверждено.

Архив для задачи B: https://disk.yandex.ru/d/Hu-um0ASI6eAUg — изображения распаковать непосредственно в `task_b/data`. Датасеты, веса модели и кэш признаков в репозиторий не включены; порядок загрузки весов указан в `task_b/README.md`.
