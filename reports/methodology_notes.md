# Notas de metodologia — PHM Tool Wear (con SHAP)

## Contexto

Estudio **preliminar / investigativo**. El objetivo no es producir un
sistema de prediccion en produccion, sino **identificar el metodo de
ML con mejor desempeno predictivo de `VB_um`** con los datos
disponibles, y luego **explicar** los mejores modelos con SHAP.

Las conclusiones aqui son hipotesis, no verdades. Antes de usar
cualquier resultado en una decision real hace falta:

- mas experimentos (≥30),
- al menos 3 herramientas distintas (no solo `T01`),
- validacion cruzada anidada para tuning,
- analisis formal de incertidumbre.

---

## Decisiones de diseno

### 1. Una fila = un experimento completo

`VB_um` se mide una sola vez por experimento. Si tratamos cada
contacto como muestra independiente y repetimos `VB_um`, hay data
leakage garantizado: el split puede poner contactos del mismo
experimento en train y test, y el modelo aprende identidad en vez
de desgaste.

`dataset_builder.py` concatena las features de los 12 segmentos
(6 contactos × 2 direcciones) en una sola fila plana por experimento.

### 2. Split a nivel `experiment_id`

`GroupShuffleSplit` (hold-out) y `LeaveOneGroupOut` (LOEO) usan
`groups=experiment_id`. Hoy con 1 fila = 1 experimento equivaldria a un
ShuffleSplit normal, pero mantener `groups` explicito protege el dia
que reorganicemos a nivel contacto.

### 3. Tuning solo dentro del train

`RandomizedSearchCV` y `GridSearchCV` usan `GroupKFold` sobre el train
del hold-out. **El test del hold-out jamas se pasa a la busqueda.**
Es nested CV imperfecto: el ranking final por MAE de hold-out puede ser
optimista. La metrica honesta es **LOEO** con los hiperparametros ya
fijados.

### 4. LOEO es mas honesto que el hold-out

Con n_test = 2 en hold-out, MAE/RMSE oscilan muchisimo. R² puede pasar
de 0.9 a -2.0 con otro seed. LOEO con n_test = 1 por fold elimina ese
sesgo: cada experimento se predice una vez, y las metricas se calculan
sobre el vector agregado de 10 predicciones (no como promedio por fold).

`final_model_ranking.csv` se ordena por MAE en sus columnas, pero el
ranking de referencia recomendado es **MAE_loeo**.

### 5. Augmentation NO reemplaza datos reales

Las filas augmentadas son variaciones artificiales del mismo
experimento real con el mismo `VB_um`. Pueden regularizar (ayudar) o
empeorar (si el modelo memoriza "VB constante + features perturbadas"
como senal contradictoria). En la corrida actual no mejora
significativamente al mejor modelo regularizado.

### 6. Tuning puede sobreajustar con pocos datos

Con 8 puntos de entrenamiento y espacios de busqueda de docenas de
combinaciones, hay riesgo real de que el mejor hiperparametro lo sea
por azar del fold particular. Por eso:

- los espacios son moderados (no exhaustivos),
- `n_iter` ≤ 20 en Random,
- el grid de XGBoost es maximo 9 combos,
- **siempre comparamos contra el modelo sin tuning**.

### 7. SHAP no entrena, no decide

SHAP entra **al final**, despues de que las metricas hayan decidido
los mejores modelos. SHAP:

- **NO** entrena modelos.
- **NO** cambia hiperparametros.
- **NO** reemplaza metricas.
- **NO** selecciona el mejor modelo por si solo.
- **NO** se aplica sobre filas augmentadas.

El script `run_shap_analysis.py`:

1. Carga `final_model_ranking.csv`.
2. Selecciona top-2 por MAE_loeo + mejor no-lineal disponible.
3. Carga los `.joblib` correspondientes.
4. **Filtra `is_augmented == False`** si la columna existe.
5. Background = train real del hold-out.
6. Explica las 10 filas reales del dataset.
7. Guarda CSVs + figuras. Si SHAP falla para un modelo: warning + skip.

### 8. Metricas

| Metrica | Por que la usamos | Cuidado |
|---|---|---|
| MAE | error medio absoluto, robusto | metrica principal |
| RMSE | penaliza outliers | usar junto a MAE |
| R² | varianza explicada | con n=2 hold-out, oscila mucho |
| MAPE | error porcentual | seguro porque VB_um siempre > 0 |

---

## Trazabilidad — CSVs por etapa

| Etapa | CSV(s) | Para que sirve |
|---|---|---|
| Auditoria | `data_inventory.csv`, `missing_segments.csv` | saber que archivos hay y cuales faltan |
| Build | `experiment_features.csv`, `modeling_dataset.csv`, `feature_columns.csv`, `contact_features.csv` | feature engineering reproducible |
| Split | `train_test_split.csv`, `loeo_folds.csv` | quien va a train/test y los folds LOEO |
| Baselines | `model_comparison_holdout.csv`, `model_comparison_loeo.csv`, `holdout_predictions.csv`, `loeo_predictions.csv` | metricas + predicciones por experimento |
| Tuning | `tuning_results.csv`, `tuning_results_loeo.csv`, `tuning_cv_results_<m>.csv` | que tan distinto sale el tuneado |
| Augmentation | `augmentation_comparison.csv`, `augmentation_predictions.csv` | con/sin augmentation lado a lado |
| Evaluacion | `final_model_ranking.csv`, `leakage_checks.csv` | tabla consolidada + auditoria de leakage |
| SHAP | `shap/shap_feature_ranking_<m>.csv`, `shap/shap_values_<m>.csv` | importancia por feature y valor por (experimento, feature) |

---

## Limitaciones conocidas

1. **Dataset extremadamente pequeno.** 10 experimentos no son
   estadisticamente representativos. Cualquier resultado es hipotesis.

2. **Una sola herramienta (`T01`).** Solo aprendemos "como envejece
   T01"; no podemos validar generalizacion a otras herramientas.

3. **Sin replicas.** Cada experimento es un valor unico de `VB_um` sin
   medidas de incertidumbre.

4. **Extrapolacion no garantizada.** Los modelos pueden interpolar en
   el rango observado (85-280 µm), pero no garantizan predicciones
   correctas fuera.

5. **Features potencialmente correlacionadas.** Las features agregadas
   multi-contacto son combinaciones de las features por contacto. Esto
   infla el conteo de features sin anadir informacion independiente.

6. **SHAP con n=10.** Las direcciones (positivas/negativas) son
   sugerentes, no concluyentes. Con tan pocos puntos un cambio de seed
   o de modelo puede reordenar el ranking SHAP.

---

## Como reportar resultados

Si en una presentacion sale un MAE de, digamos, 25 µm en hold-out:

- mencionar que es un solo split con 2 puntos de test,
- reportar tambien `MAE_loeo` (probablemente mas alto),
- decir explicitamente "con 10 experimentos y una sola herramienta",
- NO decir "el modelo tiene 25 µm de error" sin contexto,
- mencionar la varianza esperada al cambiar el split.

Para SHAP:

- describirlo como "estimacion preliminar de importancia de features",
- mencionar las features top pero advertir que con n=10 la lista puede
  cambiar al duplicar el dataset.

---

## Flujo experimental por capas (segundo orquestador)

Ademas del pipeline lineal (`run_full_pipeline.py`), el proyecto ofrece
un orquestador alterno (`run_layered_pipeline.py`) que ejecuta el
experimento como un **arbol metodologico**:

```
D → {N, A} → {ST, CT} → {(random, grid) si CT} → {HO, LOEO} → ranking → SHAP
```

### Por que un flujo por capas

1. **Trazabilidad por construccion.** Cada celda del arbol genera filas
   con etiquetas explicitas (`data_branch`, `tuning_method`,
   `validation_type`, `augmentation_strategy`, `branch_id`). Cualquier
   metrica del CSV final se puede atribuir sin ambiguedad a una
   configuracion concreta.

2. **Separacion entre Random y Grid.** El tuning queda como dos ramas
   independientes (`CT_random`, `CT_grid`) en lugar de "una sola caja
   opaca". El supervisor puede ver si Grid encontro algo que Random no.

3. **Augmentation como rama paralela.** Las 3 estrategias
   (`feature_noise`, `feature_scaling`, `grouped_scaling`) corren cada
   una con sus propias ramas {ST, CT_random, CT_grid} × {HO, LOEO},
   produciendo 9 sub-ramas A. Esto permite responder de manera limpia:
   *¿alguna estrategia + tuning supera al baseline normal?*

4. **HO y LOEO se reportan lado a lado.** No se mezclan en el mismo
   ranking: el HO se anota como `optimistic (n_test=2)` y el LOEO como
   `honest`. El ranking ordena LOEO antes que HO.

5. **SHAP estrictamente al final.** SHAP no entra en la decision del
   ganador; solo explica los modelos que las metricas ya eligieron.

### Compromiso de compute en el flujo por capas

Con 12 ramas × 2 validaciones × 8 modelos = 192 evaluaciones, hacer
*nested CV completo* (re-tunear hyperparams en cada uno de los 10 folds
de LOEO) multiplicaria el compute por 10 y, con 8 experimentos de
entrenamiento, **amplificaria el sobreajuste mas que aportar**.

Decision: la busqueda (`RandomizedSearchCV` o `GridSearchCV`) se hace
**una sola vez** sobre el train del hold-out usando `GroupKFold`
interno. Para LOEO con CT, **se reutilizan esos mejores parametros** y
solo se refitea el modelo en cada fold. Esto:

- evita el coste del nested-CV completo,
- da un estimador honesto de "¿con estos params, el modelo generaliza
  fold a fold?",
- queda documentado aqui para no presentarlo como nested-CV.

### Lectura de los outputs del flujo por capas

| Pregunta | CSV / figura a mirar |
|---|---|
| ¿Cual es el mejor modelo? | `final_layered_ranking.csv` ordenado por LOEO/MAE |
| ¿Tuning ayudo? | comparar `N_ST` vs `N_CT_*` en `all_metrics.csv` |
| ¿Augmentation ayudo? | comparar `N_ST` vs `A_ST_*` (LOEO) |
| ¿Random vs Grid? | filtrar `tuning_method` en `tuning_results_all.csv` |
| ¿Que feature es importante? | `shap/shap_feature_ranking_<model>_<branch>.csv` |
| ¿Hay leakage? | `leakage_checks.csv` (10 checks PASS/WARN/FAIL) |
| ¿Que rama gano? | `best_model_per_branch_mae.png` |
| Contar la historia | `sequential_comparison_MAE.png` (4 paneles) |

### Limitacion conocida del flujo por capas

Con 12 ramas y 192 filas de metricas, **el dashboard puede inducir a
"cherry-picking" si se lee mal**. Las metricas LOEO de las ramas
A_CT_* tienden a ser muy parecidas a sus equivalentes N_CT_* (la
augmentation no aporta nueva informacion). Si una rama particular gana
por 0.1 µm contra otra, **eso no es estadisticamente significativo con
n=10**. Solo diferencias > 5 µm deberian considerarse interesantes.

### Figura de evolucion del modelo (09_model_evolution_*_LOEO.png)

La figura de evolucion del modelo muestra como cambia el desempeno al
agregar tuning y augmentation sobre la linea base. Esta visualizacion
permite verificar si cada nuevo agregado aporta una mejora real o si
solo anade complejidad sin reducir el error.

Las 6 etapas que recorre la grafica son progresivamente mas complejas,
pero **no son una cadena causal**: solo representan configuraciones del
pipeline que un practicante ensayaria una tras otra.

1. **N_ST** — baseline simple, data normal, sin tuning, sin augmentation.
2. **N_CT_random** — agrega RandomizedSearchCV sobre el baseline.
3. **N_CT_grid** — reemplaza Random por GridSearchCV.
4. **A_ST (best aug)** — vuelve a sin tuning, pero con la mejor de las 3
   estrategias de augmentation (feature_noise / feature_scaling /
   grouped_scaling).
5. **A_CT_random (best aug)** — combina augmentation con RandomizedSearchCV.
6. **A_CT_grid (best aug)** — combina augmentation con GridSearchCV.

La grafica tiene dos capas:

- **Linea principal** (azul oscuro, gruesa, diamantes) = mejor (modelo,
  rama) por etapa. Es la lectura rapida "¿la complejidad ayuda?".
- **Lineas finas** = top-3 modelos siguiendo las 6 etapas. Permite ver si
  el ganador cambia entre etapas o si un mismo modelo (p.ej. ElasticNet)
  domina toda la progresion.

Anotaciones (solo MAE): Δ entre etapas consecutivas, con codigo de color
(verde si baja, rojo si sube, gris si |Δ|<0.5 µm). La linea horizontal
punteada marca el baseline N_ST.

**Interpretacion honesta**: si la linea es plana o sube, eso es lo que
la figura debe mostrar — *sin* forzar una narrativa de mejora. El CSV
asociado (`09_model_evolution_summary.csv`) trae una columna
`interpretation` con etiquetas como `negligible change (<1 µm)`,
`marginal worsening`, `improved`, `worsened`, `best overall`.

Cuidado tipico: el R² puede subir incluso cuando el MAE empeora (porque
RMSE/var es diferente). Mira siempre MAE/RMSE/R²/MAPE juntos antes de
afirmar "mejora" o "empeora".

---

## Que **no** esta en este proyecto (intencionalmente)

- RUL (Remaining Useful Life)
- PINN / modelos fisicos
- PCA / Health Index
- Integracion acustica
- SHAP obligatorio en el flujo de entrenamiento (esta solo al final)
- Permutation test, time-split, LOEO con augmentation, etc.

Todo eso existe en `legacy2/` por si hace falta consultarlo.
