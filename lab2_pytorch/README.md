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

Полный код, объяснение и результаты проверок находятся в [`task3/03_kainet_reconstruction.ipynb`](task3/03_kainet_reconstruction.ipynb). Внешние скрипты не нужны.

[Архив задания](https://disk.yandex.ru/d/ECLHqVyX-F5hRA) нужно распаковать в `task3/data`: PNG, TXT и `algos.csv` непосредственно в этой папке. Данные не включены в репозиторий. Затем выполните все ячейки блокнота по порядку.

Ответ: [`task3/reconstructed_algos.csv`](task3/reconstructed_algos.csv). [Посылка №166717004](https://contest.yandex.ru/contest/75233/run-report/166717004/) принята 28 сентября 2026: **OK — полное решение**.
