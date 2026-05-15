# PHM Tool Wear — Comparacion de modelos ML para predecir `VB_um`

Estudio **preliminar / investigativo** para encontrar el metodo de
Machine Learning con mejor desempeno predictivo de **desgaste de flanco
de herramienta `VB_um` (µm)** a partir de senales de vibracion, con
**solo 10 experimentos reales** y una sola herramienta.

Incluye dos pipelines complementarios, comparacion sistematica de
modelos clasicos, augmentation de datos como rama paralela, validacion
LOEO-CV, e interpretabilidad con SHAP al final del flujo.

> Los resultados **no** son aptos para produccion. Son hipotesis. Ver
> `reports/methodology_notes.md` para todas las advertencias.

---

## TL;DR

```bash
# Instalar
pip install -r requirements.txt

# Opcion A: pipeline lineal (7 pasos, recomendado primera vez)
python scripts/run_full_pipeline.py

# Opcion B: pipeline experimental por capas (12 ramas + dashboard)
python scripts/run_layered_pipeline.py
```

**Resultado actual** (LOEO-CV, la metrica honesta del proyecto):

| # | Modelo | Rama | MAE (µm) | R² |
|---|---|---|---|---|
| 1 | ElasticNet | A_ST_feature_noise | 26.92 | 0.601 |
| 2 | ElasticNet | N_ST (baseline) | 26.96 | 0.602 |
| 3 | Lasso | N_CT_random (tuned) | 27.86 | 0.498 |

Los **modelos lineales regularizados** dominan. Tuning empeora. Augmentation
no aporta. Modelos no-lineales sobreajustan. Mas detalle en
[Resultados actuales](#resultados-actuales).

---

## Tabla de contenidos

1. [Objetivo](#objetivo)
2. [Datos y unidad supervisada](#datos-y-unidad-supervisada)
3. [Dos pipelines en paralelo](#dos-pipelines-en-paralelo)
4. [Pipeline lineal `run_full_pipeline.py`](#pipeline-lineal-run_full_pipelinepy)
5. [Pipeline por capas `run_layered_pipeline.py`](#pipeline-por-capas-run_layered_pipelinepy)
6. [Inputs esperados](#inputs-esperados)
7. [Modelos comparados](#modelos-comparados)
8. [Tuning (Random vs Grid)](#tuning-random-vs-grid)
9. [Augmentation](#augmentation)
10. [Validacion: Hold-out vs LOEO-CV](#validacion-hold-out-vs-loeo-cv)
11. [SHAP — interpretabilidad](#shap--interpretabilidad)
12. [Data leakage — checks automaticos](#data-leakage--checks-automaticos)
13. [Estructura del proyecto](#estructura-del-proyecto)
14. [Outputs completos](#outputs-completos)
15. [Resultados actuales](#resultados-actuales)
16. [Metricas](#metricas)
17. [Inspeccion rapida](#inspeccion-rapida)
18. [CSVs bloqueados (Excel)](#csvs-bloqueados-excel)
19. [Limitaciones](#limitaciones)
20. [Extensiones futuras](#extensiones-futuras-no-implementadas)
21. [Legacy / archive](#legacy--archive)

---

## Objetivo

Encontrar el metodo de regresion con mejor **desempeno predictivo de
`VB_um`** con los datos disponibles, y luego **explicar** los mejores
modelos con SHAP. Cuando hablamos de "accuracy" nos referimos a
desempeno de regresion: MAE, RMSE, R², MAPE.

**Target unico:** `VB_um` (desgaste de flanco, µm) medido manualmente
con microscopio una sola vez por experimento.

**NO se usan otros targets** como VS, UVB, UVB/VB, corrosion termica.

---

## Datos y unidad supervisada

- **10 experimentos reales** con `experiment_id` ∈ {66, 67, 68, 69,
  70, 73, 74, 75, 76, 77}. Hay huecos en 71 y 72.
- Todos pertenecen a `tool_id = T01` (una sola herramienta).
- Cada experimento: **6 contactos** (`p1..p6`) × **2 direcciones**
  (A axial, R radial) = 12 segmentos TXT por experimento.
- `VB_um` se mide **una vez al final del experimento** → **una fila =
  un experimento completo**.

### Regla critica: nunca dividir por contacto

```
INCORRECTO                          CORRECTO
─────────────────────────           ─────────────────────────
1 fila = 1 contacto                 1 fila = 1 experimento
VB_um repetido 6 veces              VB_um aparece 1 sola vez
Split por contacto → LEAKAGE        Split por experiment_id
```

Si tratasemos cada contacto como muestra independiente con el mismo
`VB_um`, el split aleatorio podria poner contactos del mismo
experimento en train y test, y el modelo aprenderia "identidad" en
vez de desgaste. **Toda la arquitectura del proyecto fuerza la regla
correcta:** `dataset_builder.py` concatena los 12 segmentos en una
sola fila plana antes del merge con el target.

---

## Dos pipelines en paralelo

El proyecto ofrece **dos orquestadores complementarios** que coexisten
sin sobrescribirse:

| | **`run_full_pipeline.py`** (lineal) | **`run_layered_pipeline.py`** (por capas) |
|---|---|---|
| **Filosofia** | flujo secuencial paso a paso | comparacion sistematica en arbol |
| **Pasos** | 7 etapas | 12 ramas × 2 validaciones |
| **Modelos evaluados** | 8 baselines + 7 tuneados | 8 × 12 × 2 = 192 evaluaciones |
| **Tiempo** | ~2 min | ~2 min |
| **Outputs** | `outputs/{metrics,figures,...}/` | `outputs/.../layered_pipeline/` |
| **Para** | primera corrida / pipeline tipico | comparacion sistematica defendible |

Ambos generan `data/processed/experiment_features.csv` (compartido),
ambos hacen el mismo split hold-out (`outputs/splits/train_test_split.csv`)
y ambos ejecutan SHAP al final.

---

## Pipeline lineal `run_full_pipeline.py`

7 pasos secuenciales:

```
data/raw/segments/*.txt
        │
        ▼  paso 0 — audit_data.py
data_inventory.csv, missing_segments.csv, figures/data_quality/, figures/signals/
        │
        ▼  paso 1 — build_dataset.py
data/processed/{experiment_features, modeling_dataset}.csv
data/interim/contact_features.csv
outputs/metrics/feature_columns.csv
        │
        ▼  paso 2 — train_baselines.py
outputs/splits/{train_test_split, loeo_folds}.csv
outputs/metrics/model_comparison_{holdout, loeo}.csv
outputs/predictions/{holdout, loeo}_predictions.csv
outputs/figures/holdout/*.png, outputs/figures/loeo/*.png
outputs/models/*.joblib (8 baselines)
        │
        ▼  paso 3 — run_tuning.py
outputs/metrics/tuning_results{, _loeo}.csv
outputs/metrics/tuning_cv_results_<m>.csv (×6)
outputs/metrics/tuning_cv_results_xgboost_{random, grid}.csv
outputs/figures/tuning/*.png
outputs/models/best_<m>_tuned.joblib (×7)
        │
        ▼  paso 4 — run_augmentation_experiment.py
data/interim/augmentation/train_augmented_<s>.csv (×3 estrategias)
outputs/metrics/augmentation_comparison.csv
outputs/predictions/augmentation_predictions.csv
outputs/figures/augmentation/*.png
        │
        ▼  paso 5 — evaluate_models.py
outputs/metrics/final_model_ranking.csv
outputs/metrics/leakage_checks.csv
outputs/figures/features/*.png
        │
        ▼  paso 6 — run_shap_analysis.py
outputs/metrics/shap/shap_feature_ranking_<m>.csv
outputs/metrics/shap/shap_values_<m>.csv
outputs/figures/shap/*.png
```

Cada paso es **idempotente** (puedes re-ejecutarlo solo). El orquestador
purga CSVs con nombres legacy al inicio para mantener limpio
`outputs/metrics/`.

---

## Pipeline por capas `run_layered_pipeline.py`

Reorganiza el mismo experimento como un **arbol metodologico** donde
cada celda genera sus propios CSVs:

```
D (dataset = experiment_features.csv)
│
├── N (real)
│   ├── N_ST            (sin tuning)         → HO + LOEO
│   ├── N_CT_random     (RandomizedSearchCV) → HO + LOEO
│   └── N_CT_grid       (GridSearchCV)       → HO + LOEO
│
└── A (augmented, una rama por estrategia)
    ├── A_ST_<s>        (sin tuning)         → HO + LOEO
    ├── A_CT_random_<s> (RandomizedSearchCV) → HO + LOEO
    └── A_CT_grid_<s>   (GridSearchCV)       → HO + LOEO

  con s ∈ {feature_noise, feature_scaling, grouped_scaling}

→ comparacion + ranking final (LOEO MAE prioritario)
→ SHAP (top-2 LOEO + mejor no-lineal) sobre datos REALES
```

**Total: 12 ramas × 2 validaciones × 8 modelos = 192 evaluaciones**
recogidas en `all_metrics.csv` con etiquetas explicitas
(`data_branch`, `tuning_method`, `validation_type`,
`augmentation_strategy`, `branch_id`).

| Sigla | Significado |
|---|---|
| **D** | Dataset listo |
| **N** | Data real, sin augmentation |
| **A** | Train aumentado, test siempre real |
| **ST** | Sin tuning — defaults |
| **CT** | Con tuning — separado en `random` y `grid` |
| **HO** | Hold-out 8/2 |
| **LOEO** | Leave-One-Experiment-Out (metrica honesta) |

**Compromiso documentado:** la busqueda de hiperparametros se ejecuta
una sola vez sobre el train del hold-out con `GroupKFold` interno. En
LOEO se *reutilizan* esos mejores parametros y solo se refitea por
fold. Esto evita 10× re-tuning por modelo (nested-CV completo) que
con 10 experimentos amplificaria el sobreajuste mas que aportar.
Detalle en `reports/methodology_notes.md`.

### Outputs especificos del flujo por capas

```
outputs/metrics/layered_pipeline/
  cleanup_report.csv              auditoria de impurezas
  leakage_checks.csv              10 checks formales
  branch_execution_summary.csv    12 ramas con status + tiempo
  all_metrics.csv                 192 filas (model × branch × validation)
  tuning_results_all.csv          best_params + best_cv_score por modelo/rama
  augmentation_results_all.csv    sub-vista solo data_branch=A
  final_layered_ranking.csv       ranking con rank + interpretation_note
  shap_selected_models.csv        top-2 LOEO + mejor no-lineal

outputs/predictions/layered_pipeline/
  predictions_all_branches.csv    VB_real/pred/residual/abs_error/pct_error
                                  por (model, branch, validation, fold, exp)

outputs/figures/layered_pipeline/
  layered_flow_diagram.png        arbol metodologico con ganador anotado
  best_model_per_branch_mae.png   mejor modelo de cada rama
  mae_by_branch.png  rmse_by_branch.png  r2_by_branch.png  mape_by_branch.png
  sequential_comparison_MAE.png   dashboard de 4 paneles
  sequential_comparison_RMSE.png  sequential_comparison_R2.png
  actual_vs_predicted_best_global.png
  residuals_best_global.png  residuals_by_experiment_best_global.png

outputs/metrics/shap/   shap_feature_ranking_<model>_<branch>.csv
                        shap_values_<model>_<branch>.csv
outputs/figures/shap/   shap_bar_<model>_<branch>.png
                        shap_summary_<model>_<branch>.png
```

---

## Inputs esperados

### 1. Senales TXT — `data/raw/segments/{A|R}{exp_id}_p{n}.txt`

Convencion: `{A|R}{experiment_id}_p{contact_id}.txt`
- `A` = axial, `R` = rotacional
- `experiment_id` entero (66..77)
- `contact_id` entero (1..6)

Ejemplos: `A66_p1.txt`, `R77_p3.txt`.

Se esperan **120 archivos** (10 × 6 × 2). La corrida actual tiene
**116/120** (faltan 4 en exp 77: `A77_p5`, `A77_p6`, `R77_p5`,
`R77_p6`). El pipeline los maneja con NaN + imputer (mediana).

Cada TXT: 2 columnas `timestamp`, `vibration_value`. Separador
auto-detectado (`,`, `;`, `\t`, espacios). Soporta BOM `utf-8-sig`.

### 2. Target — `data/raw/targets/vb_targets.csv`

```csv
tool_id,experiment_id,experiment_order,VB_um
T01,66,1,85
T01,67,2,103
T01,68,3,119
T01,69,4,136
T01,70,5,150
T01,73,6,168
T01,74,7,190
T01,75,8,215
T01,76,9,245
T01,77,10,280
```

Solo `experiment_id` y `VB_um` son obligatorios.

### 3. Metadata opcional — `data/raw/metadata/experiment_metadata.csv`

```csv
tool_id,experiment_id,experiment_order,VC,F,end_of_life
T01,66,1,120,0.05,0
...
T01,77,10,120,0.05,1
```

Si existe se mergea. Si no, se ignora. Ninguna columna se usa como
feature predictora (todas en `NON_FEATURE_COLS`).

---

## Modelos comparados

| # | Modelo | Builder | Rol | Tuneado? |
|---|---|---|---|---|
| 1 | DummyRegressor | `build_dummy()` | baseline obligatorio (media) | NO |
| 2 | Ridge | `build_ridge()` | L2 fuerte — clave con p >> n | SI |
| 3 | Lasso | `build_lasso()` | L1 — seleccion automatica | SI |
| 4 | ElasticNet | `build_elasticnet()` | L1 + L2 | SI |
| 5 | SVR (RBF) | `build_svr()` | no-lineal robusto | SI |
| 6 | RandomForest | `build_rf()` | bagging de arboles | SI |
| 7 | XGBoost | `build_xgb()` | gradient boosting (si disponible) | SI (Random + Grid) |
| 8 | MLP (sklearn) | `build_mlp()` | NN opcional, no prioridad | NO |

Todos en `Pipeline(imputer → [scaler para lineales/SVR] → modelo)`.
Definidos en `src/phm/modeling.py`.

---

## Tuning (Random vs Grid)

Dos estrategias paralelas, definidas en `src/phm/tuning.py` y
`src/phm/layered_pipeline.py`:

### RandomizedSearchCV (n_iter=20)

| Modelo | Espacio |
|---|---|
| Ridge | `alpha ∈ {0.01, 0.1, 1, 10, 100}` |
| Lasso | `alpha ∈ {0.001, 0.01, 0.1, 1, 10}` |
| ElasticNet | `alpha ∈ {0.01, 0.1, 1, 10}`, `l1_ratio ∈ {0.1..0.9}` |
| SVR | `C ∈ {0.1, 1, 10, 100}`, `epsilon ∈ {0.1, 1, 5, 10}`, `gamma ∈ {scale, auto, 0.01, 0.1}` |
| RandomForest | `n_estimators ∈ {50, 100, 200}`, `max_depth ∈ {2, 3, 5, None}`, `min_samples_leaf ∈ {1, 2, 3}` |
| XGBoost | `n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_lambda` (~1200 combos) |

### GridSearchCV (cartesiano completo o reducido)

Misma grilla que Random para lineales (≤100 combos). Para XGBoost se
usa un **grid reducido a 3×3×3×1×1×2 = 54 combos** para mantener el
compute manejable (cartesiano completo seria intratable).

### Validacion durante el tuning

- `GroupKFold(n_splits=5)` con `groups=experiment_id`.
- Scoring: `neg_mean_absolute_error`.
- **El test del hold-out jamas se pasa a la busqueda** (anti-leakage).

---

## Augmentation

3 estrategias simples, aplicadas **solo al train** (definidas en
`src/phm/augmentation.py`):

| Estrategia | Que hace | VB_um cambia? |
|---|---|---|
| `feature_noise` | Ruido gaussiano relativo (σ=1% del std de cada feature) | NO |
| `feature_scaling` | Multiplicador `~U(0.98, 1.02)` por feature individual | NO |
| `grouped_scaling` | Mismo factor por grupo `{A\|R}_p{n}_*` (preserva relaciones intra-contacto) | NO |

`n_augmented = 3` filas por experimento del train → 8 + 24 = 32 filas.

### Reglas estrictas (verificadas en `leakage_checks.csv`)

- Augmentation solo despues del split.
- Solo en train. Test queda intacto en todos los escenarios.
- `VB_um` **NO** se altera. La fila aumentada hereda el VB original.
- Columnas protegidas: `experiment_id`, `tool_id`, `experiment_order`,
  `end_of_life`, `is_augmented`, `VB_um`.
- Marcadas con `is_augmented=True`.
- SHAP **nunca** ve filas augmentadas (filtro defensivo).

---

## Validacion: Hold-out vs LOEO-CV

| Modo | Detalle | Cuando confiar |
|---|---|---|
| **Hold-out 8/2** | `GroupShuffleSplit` por `experiment_id`, `random_state=42`. Deterministico (test = {67, 76}). | Solo referencia visual: con n_test=2, R² es muy inestable. |
| **LOEO-CV** | `LeaveOneGroupOut`, 10 folds. Predicciones agregadas. | **Metrica honesta del proyecto.** |

El split hold-out se guarda en `outputs/splits/train_test_split.csv` y
se **reutiliza en todos los scripts** (baselines, tuning, augmentation,
SHAP, layered pipeline). Los folds LOEO en
`outputs/splits/loeo_folds.csv`.

**Por que LOEO es mejor que HO con n=10:** con n_test=2, MAE/RMSE
oscilan muchisimo segun que 2 experimentos caigan en test. R² puede
pasar de 0.9 a -2.0 cambiando el seed. LOEO con n_test=1 por fold
elimina ese sesgo: cada experimento se predice exactamente una vez y
las metricas se calculan sobre el vector agregado de 10 predicciones
(no como promedio por fold — con 1 punto por fold no tiene sentido
para R²).

---

## SHAP — interpretabilidad

SHAP entra **al final**, despues de la seleccion de mejores modelos
por metricas.

### Seleccion automatica

- **Top-2 por MAE_loeo** (validation_type='loeo', pipeline_variant='baseline').
- **+ Mejor no-lineal disponible** si no aparece ya en el top-2.

### Explainer por tipo de modelo

- `shap.LinearExplainer` → Ridge, Lasso, ElasticNet.
- `shap.TreeExplainer`   → RandomForest, XGBoost.
- `shap.KernelExplainer` (fallback) → SVR, MLP.

### Datos sobre los que se explica

- **Background** = train real del hold-out (8 experimentos).
- **Datos a explicar** = las 10 filas reales del dataset.
- **Filtro defensivo:** si `is_augmented` esta en columnas, se filtra
  `df[df['is_augmented'] == False]`.

### Que NO hace SHAP

- NO entrena modelos.
- NO cambia hiperparametros.
- NO reemplaza las metricas.
- NO selecciona el mejor modelo.
- NO se aplica sobre filas augmentadas.

Si SHAP falla para un modelo, se omite con warning (el pipeline
continua).

---

## Data leakage — checks automaticos

`outputs/metrics/leakage_checks.csv` (linear) y
`outputs/metrics/layered_pipeline/leakage_checks.csv` (layered) contienen:

| Check | Que valida |
|---|---|
| `one_row_per_experiment` | 1 fila ≡ 1 experimento |
| `target_unique_per_experiment` | un solo `VB_um` por experimento |
| `no_experiment_in_both_splits` | sin solape train/test |
| `id_columns_excluded` | `experiment_id`, `tool_id`, `experiment_order`, etc. jamas como features |
| `test_not_augmented` | CSVs de augmentation no contienen experimentos de test |
| `augmented_rows_keep_vb` | `VB_um` inalterado en filas augmentadas |
| `no_contact_level_labels` | arquitectura no soporta filas por contacto |
| `experiment_order_excluded_by_default` | `experiment_order` esta en `NON_FEATURE_COLS` |
| `shap_real_data_only` | SHAP filtra `is_augmented` antes de explicar |
| `tuning_only_train` | `*SearchCV` usan `GroupKFold` sobre `X_train` |

Cada check devuelve `PASS`, `WARN` o `FAIL` con detalle textual.

---

## Estructura del proyecto

```
phm_tool_wear/
│
├── README.md                       ← este archivo
├── requirements.txt                ← numpy, pandas, scipy, sklearn,
│                                    matplotlib, xgboost, shap
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── segments/               ← *.txt (entrada)
│   │   ├── targets/vb_targets.csv  ← entrada
│   │   └── metadata/experiment_metadata.csv  ← opcional
│   ├── processed/
│   │   ├── experiment_features.csv ← 10 × 208 (wide)
│   │   └── modeling_dataset.csv    ← 10 × 205 (id + 203 features + target)
│   └── interim/
│       ├── contact_features.csv    ← long: 60 filas (10 exp × 6 contactos)
│       └── augmentation/
│           ├── train_augmented_feature_noise.csv
│           ├── train_augmented_feature_scaling.csv
│           └── train_augmented_grouped_scaling.csv
│
├── src/phm/
│   ├── __init__.py
│   ├── config.py                   ← rutas, constantes, ensure_output_dirs()
│   ├── filename_parser.py          ← {A|R}{id}_p{n}.txt → metadata
│   ├── data_loader.py              ← carga TXT robusta (auto-sep, NaN, BOM)
│   ├── preprocessing.py            ← NaN, dedup, sort, center, sampling_rate
│   ├── feature_extraction.py       ← 13 time + 3 freq por segmento
│   ├── dataset_builder.py          ← experiment_features + modeling + manifests
│   ├── splitting.py                ← hold-out 8/2 + LOEO + write_loeo_folds
│   ├── modeling.py                 ← 8 builders sklearn Pipeline
│   ├── tuning.py                   ← RandomizedSearchCV + GridSearchCV (XGB)
│   ├── augmentation.py             ← feature_noise / scaling / grouped_scaling
│   ├── evaluation.py               ← compute_metrics, make_predictions_df,
│   │                                 safe_to_csv, read_latest_csv
│   ├── visualization.py            ← plots por etapa (helpers _in())
│   ├── data_quality.py             ← inventario, missing, plots calidad
│   ├── leakage_audit.py            ← checks formales con CSV
│   ├── shap_analysis.py            ← LinearExplainer/TreeExplainer/fallback
│   ├── layered_pipeline.py         ← engine arbol (12 ramas)
│   └── layered_visuals.py          ← flow diagram + dashboard secuencial
│
├── scripts/
│   ├── audit_data.py               ← paso 0 (lineal)
│   ├── build_dataset.py            ← paso 1
│   ├── train_baselines.py          ← paso 2
│   ├── run_tuning.py               ← paso 3
│   ├── run_augmentation_experiment.py  ← paso 4
│   ├── evaluate_models.py          ← paso 5
│   ├── run_shap_analysis.py        ← paso 6
│   ├── run_full_pipeline.py        ← orquestador lineal (corre 0→6)
│   └── run_layered_pipeline.py     ← orquestador por capas (12 ramas + SHAP)
│
├── outputs/
│   ├── splits/                     ← train_test_split.csv, loeo_folds.csv
│   ├── models/                     ← *.joblib (8 baselines + 7 tuneados)
│   ├── metrics/                    ← CSVs por etapa
│   │   ├── shap/                   ← rankings y valores SHAP
│   │   └── layered_pipeline/       ← CSVs del flujo por capas
│   ├── predictions/                ← *_predictions.csv
│   │   └── layered_pipeline/       ← predictions_all_branches.csv
│   ├── figures/
│   │   ├── data_quality/           ← 4 figs
│   │   ├── signals/                ← 3 figs
│   │   ├── features/               ← 3 figs
│   │   ├── holdout/                ← 5 figs
│   │   ├── loeo/                   ← 6 figs
│   │   ├── tuning/                 ← 4 figs
│   │   ├── augmentation/           ← 6 figs (incluyendo delta_mae)
│   │   ├── shap/                   ← 2 figs por modelo explicado
│   │   └── layered_pipeline/       ← 12 figs (flow diagram, dashboards, ...)
│   ├── archive/                    ← outputs anteriores preservados
│   └── logs/                       ← reservado
│
├── reports/methodology_notes.md
└── legacy2/                        ← proyecto anterior completo
```

---

## Outputs completos

Conteos verificados de la corrida actual:

| Tipo | Cantidad | Ubicacion |
|---|---|---|
| CSVs (lineal) | 19 | `outputs/metrics/` + 6 SHAP + 2 splits + 3 predicciones |
| CSVs (layered) | 8 | `outputs/metrics/layered_pipeline/` + 1 prediction |
| CSVs (data) | 2 + 4 | `data/processed/` + `data/interim/` |
| `.joblib` models | 15 | `outputs/models/` |
| PNGs (lineal) | 24 | `outputs/figures/{data_quality,signals,features,holdout,loeo,tuning,augmentation,shap}/` |
| PNGs (layered) | 12 + 6 SHAP | `outputs/figures/layered_pipeline/` + `outputs/figures/shap/` |

### CSVs del flujo lineal (las claves)

| Archivo | Contenido |
|---|---|
| `data_inventory.csv` | 1 fila por TXT presente, con flags de validez |
| `missing_segments.csv` | TXT esperados que faltan |
| `feature_columns.csv` | 208 columnas con flag `used_in_model` |
| `train_test_split.csv` | `experiment_id`, `split`, `VB_um`, `tool_id` |
| `loeo_folds.csv` | 10 folds: `fold_id`, `test_experiment_id`, `train_experiment_ids` |
| `model_comparison_holdout.csv` | 8 baselines en hold-out |
| `model_comparison_loeo.csv` | 8 baselines en LOEO |
| `tuning_results.csv` | 7 tuneados HO + `best_params` JSON |
| `tuning_results_loeo.csv` | 7 tuneados LOEO |
| `tuning_cv_results_<m>.csv` | CV detallado por modelo (×6) |
| `tuning_cv_results_xgboost_{random,grid}.csv` | CV detallado XGB |
| `augmentation_comparison.csv` | 32 evaluaciones (8 × 4 estrategias) |
| `final_model_ranking.csv` | Ranking consolidado |
| `leakage_checks.csv` | 6 checks |
| `holdout_predictions.csv` | 16 filas con `residual`, `absolute_error`, `percentage_error` |
| `loeo_predictions.csv` | 80 filas con `fold_id` |
| `augmentation_predictions.csv` | 64 filas |
| `shap/shap_feature_ranking_<m>.csv` | feature × `mean_abs_shap`, `mean_shap`, `rank` |
| `shap/shap_values_<m>.csv` | Largo: 1 fila por (`experiment_id`, `feature`) |

### CSVs del flujo por capas

| Archivo | Contenido |
|---|---|
| `cleanup_report.csv` | 4 checks de impureza |
| `leakage_checks.csv` | 10 checks (mas estrictos que el lineal) |
| `branch_execution_summary.csv` | 12 ramas con status + duracion |
| `all_metrics.csv` | **192 filas** con etiquetas explicitas por rama |
| `tuning_results_all.csv` | best_params por (modelo, rama) |
| `augmentation_results_all.csv` | sub-vista solo A |
| `final_layered_ranking.csv` | ranking consolidado con `interpretation_note` |
| `shap_selected_models.csv` | 3 modelos con razon de seleccion |
| `predictions_all_branches.csv` | ~1700 filas, predicciones por (rama, fold, exp) |

### Modelos guardados (15 `.joblib`)

```
dummyregressor.joblib  ridge.joblib       lasso.joblib       elasticnet.joblib
svr.joblib             randomforest.joblib  xgboost.joblib   mlp.joblib

best_ridge_tuned.joblib       best_lasso_tuned.joblib    best_elasticnet_tuned.joblib
best_svr_tuned.joblib         best_randomforest_tuned.joblib
best_xgboost_random_tuned.joblib  best_xgboost_grid_tuned.joblib
```

### Figuras destacadas (que mostrar al supervisor)

| Figura | Que cuenta |
|---|---|
| `figures/data_quality/vb_vs_experiment_order.png` | Como evoluciona `VB_um` con el orden cronologico |
| `figures/shap/10_shap_bar_elasticnet_a_st_feature_noise.png` | Top features del ganador |
| **`figures/layered_pipeline/00_layered_flow_diagram_no_holdout.png`** | **El arbol del experimento entero (LOEO-only)** |
| **`figures/layered_pipeline/09_sequential_comparison_dashboard_MAE.png`** | **Dashboard de 4 paneles — la "historia"** |
| **`figures/layered_pipeline/09_model_evolution_MAE_LOEO.png`** | **Evolucion del MAE al agregar etapas (tuning, augmentation)** |
| `figures/layered_pipeline/09_best_model_per_branch_MAE.png` | Mejor modelo por cada una de las 12 ramas |
| `figures/layered_pipeline/09_delta_MAE_vs_baseline_N_ST.png` | ΔMAE de cada rama vs el baseline N_ST |
| `figures/layered_pipeline/09_random_vs_grid_MAE.png` | Random vs Grid lado a lado + delta |
| `figures/layered_pipeline/09_heatmap_model_vs_branch_MAE.png` | Heatmap modelo × rama (MAE) |

**La figura de evolucion del modelo**
(`09_model_evolution_MAE_LOEO.png`) muestra como cambia el desempeno al
agregar tuning y augmentation sobre la linea base. Esta visualizacion
permite verificar si cada nuevo agregado aporta una mejora real o si
solo anade complejidad sin reducir el error. La linea negra gruesa es
el "best per stage"; las lineas finas son los top-3 modelos siguiendo
las 6 etapas. Si la linea es plana, la complejidad no ayuda.

---

## Resultados actuales

> Para reproducir: `python scripts/run_layered_pipeline.py`.

### Top 10 LOEO (la metrica honesta)

| rank | modelo | rama | MAE (µm) | R² |
|---|---|---|---|---|
| 1 | **ElasticNet** | A_ST_feature_noise | **26.92** | 0.601 |
| 2 | ElasticNet | N_ST (baseline) | 26.96 | 0.602 |
| 3 | ElasticNet | A_ST_grouped_scaling | 27.04 | 0.599 |
| 4 | ElasticNet | A_ST_feature_scaling | 27.07 | 0.619 |
| 5 | Lasso | N_CT_random | 27.86 | 0.498 |
| 6 | Lasso | N_CT_grid | 27.86 | 0.498 |
| 7 | ElasticNet | N_CT_random | 30.55 | 0.562 |
| 8 | ElasticNet | N_CT_grid | 30.55 | 0.562 |
| 9 | ElasticNet | A_CT_random_feature_noise | 30.57 | 0.562 |
| 10 | ElasticNet | A_CT_grid_feature_noise | 30.57 | 0.562 |

### Lecturas honestas

- **Modelos lineales regularizados dominan**: ElasticNet y Lasso ocupan
  todos los top-10. Ningun modelo no-lineal aparece arriba.
- **Tuning empeora** en LOEO por +0.90 µm respecto al baseline. Con 8
  puntos de entrenamiento, el tuning sobreajusta al CV interior.
- **Augmentation no aporta**: la mejora es +0.04 µm — irrelevante con
  n=10. La primera y segunda fila estan a 0.04 µm entre si: **son
  empates estadisticos**.
- **Random vs Grid:** convergen al mismo optimo para Lasso (α=0.01),
  por eso ranks 5-6 y 9-10 son identicos.
- **SHAP — top feature consistente:** `A_p6_dominant_freq_hz` para
  ElasticNet, `A_p1_skewness` para XGBoost.

### Conclusion del estudio preliminar

> Con 10 experimentos y una sola herramienta, **el mejor estimador
> honesto del error es MAE ≈ 27 µm con ElasticNet sin tuning**.
> Cualquier diferencia inferior a ~5 µm entre configuraciones **no es
> estadisticamente significativa**. Modelos no-lineales, tuning y
> augmentation no aportan en este regimen de datos. Para mejorar
> sustancialmente hace falta MAS DATOS, no mas modelos.

---

## Metricas

| Metrica | Sentido | Interpretacion |
|---|---|---|
| MAE | error absoluto medio (µm) | menor es mejor — **metrica principal** |
| RMSE | raiz error cuadratico medio (µm) | menor es mejor |
| R² | coef. determinacion | con n=2 hold-out, oscila mucho |
| MAPE | error porcentual medio (%) | requiere VB_um > 0 (todos lo son) |

Los CSVs de predicciones contienen `residual`, `absolute_error` y
`percentage_error` por experimento para diagnostico fino.

---

## Inspeccion rapida

```bash
# Top 10 del ranking global (lineal)
python -c "import pandas as pd; print(pd.read_csv('outputs/metrics/final_model_ranking.csv').sort_values('MAE').head(10).to_string(index=False))"

# Top 10 del ranking por capas
python -c "import pandas as pd; print(pd.read_csv('outputs/metrics/layered_pipeline/final_layered_ranking.csv').head(10).to_string(index=False))"

# Checks de leakage
python -c "import pandas as pd; print(pd.read_csv('outputs/metrics/layered_pipeline/leakage_checks.csv').to_string(index=False))"

# Top features del ganador (ElasticNet baseline)
python -c "import pandas as pd; print(pd.read_csv('outputs/metrics/shap/shap_feature_ranking_elasticnet.csv').head(10).to_string(index=False))"

# Resumen ejecucion de ramas
python -c "import pandas as pd; print(pd.read_csv('outputs/metrics/layered_pipeline/branch_execution_summary.csv').to_string(index=False))"

# Predicciones del mejor modelo LOEO
python -c "
import pandas as pd
df = pd.read_csv('outputs/predictions/loeo_predictions.csv')
sub = df[df['model']=='ElasticNet'].sort_values('experiment_id')
print(sub[['experiment_id','VB_real','VB_pred','residual','absolute_error']].to_string(index=False))
"
```

---

## CSVs bloqueados (Excel)

Si un CSV de `outputs/metrics/` esta abierto en Excel u otro visor que
lo bloquee, Windows impedira la sobreescritura. El pipeline maneja
esto con `safe_to_csv()` en `src/phm/evaluation.py`:

1. Intenta `to_csv(path)` con **3 reintentos** y 1s de espera entre cada uno.
2. Si sigue bloqueado, escribe a `<path>.new.csv` y emite un warning
   recomendando cerrar el visor.
3. El paso siguiente usa `read_latest_csv()` que elige automaticamente
   la version mas reciente entre el archivo canonico y `.new.csv`.

Para resolver manualmente: cierra el visor y renombra `*.new.csv` al
nombre canonico, o re-ejecuta el paso correspondiente.

---

## Limitaciones

- Solo **10 experimentos** y **una sola herramienta** (`T01`).
- ~200 features vs 8 puntos de entrenamiento → ratio p/n alto.
- Hold-out con n_test=2 tiene varianza enorme.
- LOEO es mejor estimador pero sigue siendo escaso.
- SHAP con 10 filas: las direcciones de las features son sugerentes,
  no concluyentes.
- Tuning puede sobreajustar al CV interior con tan poca data.
- Augmentation no aporta informacion nueva (filas artificiales).
- 4 archivos TXT faltantes en exp 77 — manejados con NaN+imputer,
  pero introducen ruido.
- Las conclusiones deben reportarse con cautela.

---

## Extensiones futuras (no implementadas)

| Extension | Por que no esta |
|---|---|
| RUL (Remaining Useful Life) | requiere modelo temporal + datos secuenciales |
| Modelos fisicos / PINN | fuera del alcance preliminar |
| PCA / Health Index | feature engineering avanzado |
| Integracion acustica | falta data acustica |
| Degradation models | requiere mas experimentos por herramienta |
| Nested CV completo para tuning | demasiado coste con n=10 (sobreajuste amplificado) |
| Permutation test formal | poca potencia con n=10 |
| Time-based split (extrapolacion) | existio en legacy2, fallo predeciblemente |

Se mantienen fuera intencionalmente. El alcance actual es comparar
metodos de ML clasicos con data escasa y dejar el flujo trazable.

---

## Legacy / archive

- `legacy2/` — proyecto anterior completo (no se toca, sirve de
  referencia).
- `outputs/archive/` — outputs antiguos preservados:
  - `legacy_augmentation_interim/` — CSV viejos del proyecto anterior.
  - `legacy_figures/` — PNG sueltos previos al refactor.
  - `layered_pipeline_runs/run_<timestamp>/` — outputs de corridas
    layered anteriores (archivados automaticamente cada vez que se
    re-ejecuta `run_layered_pipeline.py`).

Nada de esto se versiona (esta en `.gitignore`). Se conserva en disco
por si hace falta recuperar codigo o resultados puntuales.

---

## Quick reference

```bash
# Cero a resultados (primera vez)
pip install -r requirements.txt
python scripts/run_full_pipeline.py          # pipeline lineal 7 pasos
python scripts/run_layered_pipeline.py       # pipeline por capas 12 ramas

# Solo regenerar el dataset
python scripts/build_dataset.py

# Solo SHAP sobre modelos ya entrenados
python scripts/run_shap_analysis.py

# Inspeccionar el ganador
python -c "import pandas as pd; print(pd.read_csv('outputs/metrics/layered_pipeline/final_layered_ranking.csv').head(5).to_string(index=False))"
```

Para preguntas metodologicas, ver `reports/methodology_notes.md`.
