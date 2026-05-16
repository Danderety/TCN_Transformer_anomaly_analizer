# Interpretable Adaptive TCN-Transformer Autoencoder for Log Anomaly Detection

Каркас проекта для темы:

> Интерпретируемая адаптивная TCN-Transformer Autoencoder модель для обнаружения и локализации аномалий в системных логах.

Проект закрывает полный pipeline:

```text
raw logs -> templates -> EventID sequences -> TCN-AE baseline
         -> TCN-Transformer-AE -> anomaly score
         -> fixed/adaptive threshold -> XAI heatmap
         -> detection/localization metrics -> saved outputs
```

## 1. Что внутри

```text
configs/                 YAML-конфиги экспериментов
src/data/                adapters, preprocessing, vocab, dataset
src/models/              TCN-AE и TCN-Transformer-AE
src/training/            loss, trainer, checkpoints
src/evaluation/          scoring, thresholds, metrics
src/xai/                 heatmap и gradient saliency
src/visualization/       графики
scripts/                 этапы запуска
notebooks/               notebook-first запуск экспериментов
outputs/                 модели, метрики, предсказания, графики
main.py                  полный pipeline
```

## 2. Установка

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
```

Проверка CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Если CUDA недоступна, код запустится на CPU. Для реальной работы лучше установить PyTorch под твою версию CUDA.

## 3. Быстрый запуск на demo-данных

```bash
python main.py --config configs/experiment_demo.yaml --generate_demo
```

Это создаст toy-логи, обучит baseline и основную модель, посчитает метрики и сохранит графики.

## 4. Запуск на LO2

Ссылки на данные также продублированы в `data/README.md`:

| Dataset | Куда распаковать | Source |
|---|---|---|
| LO2 | `data/raw/lo2/` | https://doi.org/10.5281/zenodo.14265858 |
| Loghub-2.0 | `data/raw/loghub2/` | https://zenodo.org/records/8275861 |
| RCAEval | `data/raw/rcaeval/` | https://github.com/phamquiluan/RCAEval |

1. Распакуй LO2 в:

```text
data/raw/lo2/
```

2. Подготовь данные:

```bash
python scripts/01_prepare_data.py --config configs/experiment_lo2.yaml
```

3. Используй generated config, потому что туда записывается `vocab_size`:

```bash
python main.py --config data/processed/lo2/used_config.yaml
```

Аналогично для Loghub-2.0 и RCAEval:

```bash
python scripts/01_prepare_data.py --config configs/experiment_loghub.yaml
python main.py --config data/processed/loghub2/used_config.yaml
```

```bash
python scripts/01_prepare_data.py --config configs/experiment_rcaeval.yaml
python main.py --config data/processed/rcaeval/used_config.yaml
```

## 5. Что надо адаптировать под реальные датасеты

Главное место:

```text
src/data/adapters.py
```

Сейчас adapter рекурсивно читает `.log`, `.txt`, `.out` и определяет label по пути:

```text
error / fault / anomaly / abnormal / failure / fail -> 1
иначе -> 0
```

Для строгой работы нужно уточнить:

```text
session_id  — что считать одной последовательностью логов
label       — где в датасете хранится normal/anomaly
service     — компонент/сервис
 timestamp  — правильный порядок событий
```

Это нормальная dataset-specific часть. Модель, scoring, XAI и метрики остаются общими.

## 6. Отдельный запуск этапов

```bash
python scripts/01_prepare_data.py --config configs/experiment_lo2.yaml
python scripts/02_train_baseline_tcn_ae.py --config data/processed/lo2/used_config.yaml
python scripts/03_train_tcn_transformer_ae.py --config data/processed/lo2/used_config.yaml
python scripts/08_train_tcn_transformer_ensemble.py --config data/processed/lo2/used_config.yaml
python scripts/04_evaluate_detection.py --config data/processed/lo2/used_config.yaml
python scripts/05_evaluate_localization.py --config data/processed/lo2/used_config.yaml
python scripts/06_evaluate_adaptive_threshold.py --config data/processed/lo2/used_config.yaml
python scripts/07_generate_report_assets.py --config data/processed/lo2/used_config.yaml
```

## 6.1 Запуск через Jupyter

Открывай:

```text
notebooks/04_final_results.ipynb
```

Тетрадка не дублирует код проекта: она вызывает stage-функции из
`src/utils/notebook_workflow.py`, которые ссылаются на `scripts/` и `src/`.
Графики также не пишутся в notebook: они создаются через
`scripts/07_generate_report_assets.py`, а реализация лежит в
`src/visualization/plots.py`.

Для TCN-Transformer ensemble используются:

```text
scripts/08_train_tcn_transformer_ensemble.py
src/evaluation/ensemble.py
src/evaluation/thresholds.py
```

Пороговая политика:

```text
member scores -> robust-z по train scores -> mean ensemble score
             -> SPOT/POT raw threshold
             -> validation safety-check: final=min(raw, validation recall-lock)
```

Это правило закреплено в конфиге через `ensemble.*`, `threshold.spot_for_models`
и `threshold.validation_safety_check`.

## 7. Куда смотреть результаты

```text
outputs/models/
  tcn_ae_best.pt
  tcn_transformer_ae_best.pt
  tcn_transformer_ae_seed*_best.pt

outputs/metrics/
  detection_metrics.csv
  localization_metrics.csv
  adaptive_threshold_metrics.csv
  final_comparison.csv

outputs/predictions/
  tcn_ae_test_predictions.json
  tcn_transformer_ae_test_predictions.json
  tcn_transformer_ae_ensemble_test_predictions.json
  tcn_transformer_ae_localization_predictions.json
  adaptive_predictions.json

outputs/figures/
  anomaly_score_distribution.png
  adaptive_threshold_dynamics.png
  model_comparison_f1.png
  heatmap_example_*.png

outputs/reports/
  summary.md
```

## 8. Как интерпретировать результаты

### Detection

`outputs/metrics/detection_metrics.csv`:

- `precision` — сколько найденных аномалий действительно аномальны;
- `recall` — сколько реальных аномалий найдено;
- `f1` — баланс precision и recall;
- `pr_auc` — качество ранжирования по anomaly score;
- `false_positive_rate` — доля ложных срабатываний.

### Localization

`outputs/metrics/localization_metrics.csv`:

- `top_1_hit` — попала ли самая подозрительная позиция в истинную аномалию;
- `top_3_hit` — попала ли одна из top-3 позиций;
- `mean_rank` — средний ранг истинной аномальной позиции;
- `iou_at_k` — пересечение top-k подозрительных позиций с истинной маской.

### XAI heatmap

Heatmap строится по ошибке восстановления каждой позиции:

```text
ошибка восстановления позиции = вклад позиции в anomaly score
```

Это корректно формулировать так:

> модель локализует наиболее подозрительный участок логовой последовательности, давший наибольший вклад в anomaly score.

Не стоит писать, что модель гарантированно нашла истинную причину сбоя.

## 9. Если не хватает VRAM

В конфиге уменьши:

```yaml
training:
  batch_size: 16

model:
  d_model: 64
  transformer_layers: 1
```

## 10. Типовые ошибки

### `model.vocab_size is None`

Ты запустил train script на исходном конфиге. Нужно сначала выполнить подготовку данных и использовать `used_config.yaml`.

### `No normal train sequences`

Adapter неправильно определил label. Проверь `data/processed/<dataset>/parsed_logs.csv`.

### Слишком мало sequences

Проверь `session_id` в adapter или уменьши:

```yaml
data:
  min_seq_len: 2
```

## 11. Эксперименты для научной работы

| Эксперимент | Что показывает |
|---|---|
| TCN-AE vs TCN-Transformer-AE | полезность Transformer-блока |
| Fixed vs Adaptive threshold | влияние адаптивного порога |
| Synthetic anomaly localization | способность heatmap подсвечивать внесённый участок |
| Score distribution | разделимость normal/anomaly по anomaly score |
