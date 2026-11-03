# ♟️ Chess Match Predictor

Sistema de predicción de resultados de partidas de ajedrez mediante Machine Learning.

---

## Descripción

Este proyecto predice el resultado de una partida de ajedrez entre dos usuarios de Chess.com antes de que ésta comience. El modelo se entrena con el dataset público de Kaggle [chesscom-user-games-60000-games](https://www.kaggle.com/datasets/adityajha1504/chesscom-user-games-60000-games) y, en producción, consulta la API de Chess.com para obtener el rating actual de cada jugador. A partir del rating, la diferencia de rating y el ritmo de juego construye el vector de características (enfoque "Opción A", por partida) y aplica el mejor de cuatro modelos candidatos entrenados con datos reales.

> El contenido de las fuentes externas fue reescrito para cumplir con las restricciones de licencia.

---

## Problema que resuelve

Dado el nombre de usuario de dos jugadores en Chess.com, el color asignado al jugador A, el ritmo de juego y si la partida es puntuable, el sistema estima la probabilidad de:

- Victoria del jugador A
- Victoria del jugador B
- Empate

En producción el modelo recibe las dos cuentas, consulta la API de Chess.com para obtener el rating actual de cada jugador en la modalidad elegida y predice el resultado.

---

## Objetivos

1. Entrenar el modelo con el dataset público de Kaggle de partidas de Chess.com.
2. Construir el conjunto de características "Opción A" (por partida), sin depender del historial.
3. Entrenar y comparar cuatro candidatos: un XGBoost multiclase único, un XGBoost de dos etapas, un RandomForest y una línea base de regresión logística que solo usa el rating.
4. Seleccionar automáticamente el mejor modelo (mayor accuracy; empates se rompen por menor log-loss) y desplegarlo en producción.
5. Ofrecer una interfaz web intuitiva con Streamlit.
6. Explicar las predicciones mediante SHAP.

### Características (Opción A, 6 variables)

| Variable | Descripción |
|----------|-------------|
| `white_rating` | Rating ELO de las blancas |
| `black_rating` | Rating ELO de las negras |
| `rating_diff` | Diferencia de rating (blancas − negras) |
| `abs_rating_diff` | Valor absoluto de la diferencia |
| `time_class_encoded` | Ritmo de juego (bullet=0, blitz=1, rapid=2, daily=3) |
| `rated` | Si la partida es puntuable (binaria 0/1) |

Como ~86 % de los usuarios aparece una sola vez en el dataset, las variables basadas en historial reciente son inviables; por eso la Opción A usa solo el contexto de la propia partida.

Las variables `base_time` e `increment` (tiempo base e incremento del reloj) se eliminaron del modelo para mantener coherencia entre entrenamiento y producción: la interfaz solo permite elegir el ritmo/clase de tiempo, no el reloj exacto, por lo que el modelo no debe depender de un control de tiempo concreto al predecir. Las columnas se conservan en `dataset.csv` para uso futuro.

### Arquitecturas candidatas y selección automática

El pipeline entrena cuatro candidatos, los evalúa sobre la **misma** partición de test y despliega automáticamente el mejor (mayor accuracy; los empates se rompen por menor log-loss):

1. **XGBoost multiclase** (`multi:softprob`, 3 clases): un único clasificador XGBoost afinado. Suele ganar por su accuracy y, sobre todo, por su log-loss mucho más bajo (probabilidades mejor calibradas, que es lo que muestra la app).
2. **XGBoost de dos etapas** (modelo de comparación): la **Etapa 1** estima P(empate) y, si supera el umbral `t1` (barrido sobre {0.3, 0.4, 0.5, 0.6, 0.7} maximizando el F1 macro en validación), predice Empate; la **Etapa 2** decide el color (si P(blancas) supera `t2`, por defecto 0.5, ganan blancas). Se conserva como comparación por su valor conceptual.
3. **RandomForest** multiclase.
4. **Línea base** (`LogisticRegression`) que solo usa columnas de rating.

El modelo desplegado expone una interfaz uniforme (`predict`, `predict_proba` con columnas `[blancas, negras, empate]` y `feature_names`), de modo que el motor de predicción y SHAP no dependen de qué arquitectura ganó. Se guarda completo en `models/best_model.pkl` junto con sus metadatos en `models/model_metadata.json` (algoritmo desplegado, métricas, matriz de confusión 3x3, métricas de empate y `all_model_results` con los cuatro candidatos; si gana la de dos etapas se añaden además sus umbrales y el barrido de `t1`).

### Matriz de confusión y métricas de empate

Para el modelo desplegado se generan además:

- La **matriz de confusión 3x3** (filas = real, columnas = predicho; orden `[gana blancas, gana negras, tablas]`), guardada como lista anidada en los metadatos y renderizada en `data/processed/confusion_matrix.png`.
- Una **matriz de confusión binaria Empate vs No-Empate** (2x2) con sus conteos VP/FN/FP/VN y las métricas de precisión, recall y F1 del empate, bajo la clave `draw_metrics` en los metadatos y renderizada en `data/processed/confusion_matrix_draw.png`.

La clase Empate (~3.5 % del dataset) tiene muy poca señal predictiva en rating y contexto, por lo que su recall se mantiene bajo; es un hallazgo esperado y documentado, no un error.

### Trabajo futuro

- **Razón de victoria o tablas** (tiempo, jaque mate, material insuficiente, etc.): el dataset conserva la columna `termination_reason` para abordar esto más adelante.
- **Opción B basada en historial:** combinar la Opción A con un enfoque que use las últimas partidas de cada jugador (`src/data_collection.py` se mantiene como base para este camino).

---

## Tecnologías

| Tecnología | Uso |
|------------|-----|
| Python | Lenguaje principal |
| Pandas / NumPy | Manipulación de datos |
| Scikit-learn | Entrenamiento y evaluación |
| Matplotlib / Seaborn | Visualizaciones |
| Requests | API Chess.com |
| SHAP | Explicabilidad |
| Streamlit | Interfaz web |
| Joblib | Persistencia de modelos |

---

## Fuente de datos

**Entrenamiento:** dataset público de Kaggle, descargado con `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("adityajha1504/chesscom-user-games-60000-games")
print("Path to dataset files:", path)
```

`src/data_loader.py` descarga el dataset, filtra a ajedrez clásico (`rules == 'chess'`), deriva el objetivo de 3 clases (0 = ganan blancas, 1 = ganan negras, 2 = empate), parsea el control de tiempo y genera `data/processed/dataset.csv`.

**Producción:** API pública de Chess.com (sin API Key) para obtener el rating actual de cada jugador en la modalidad elegida:

```
https://api.chess.com/pub/player/{username}/stats
```

No se almacenan contraseñas ni datos privados. Solo se accede a datos públicos.

---

## Instalación

```bash
# 1. Clonar o descargar el proyecto
cd chess_prediction

# 2. (Recomendado) Crear un entorno virtual
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux / macOS

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Uso

### 1. Preparación de datos

```bash
python -m src.data_loader
```

Descarga el dataset de Kaggle (con `kagglehub`), lo filtra y procesa, y genera `data/processed/dataset.csv` (y el crudo filtrado en `data/raw/games_raw.csv`).

> `src/data_collection.py` se mantiene para un futuro enfoque basado en historial (Opción B); no se usa en el flujo actual de entrenamiento.

### 2. Entrenamiento del modelo

```bash
python -m src.train
```

Entrena los cuatro candidatos (XGBoost multiclase, XGBoost de dos etapas, RandomForest y línea base de regresión logística), los evalúa sobre la misma partición de test, selecciona automáticamente el mejor y lo guarda en `models/best_model.pkl` junto con los metadatos en `models/model_metadata.json`. Genera además `data/processed/model_comparison.png`, `data/processed/confusion_matrix.png` y `data/processed/confusion_matrix_draw.png`.

### 3. Interfaz web

```bash
streamlit run app.py
```

Abre el navegador en `http://localhost:8501`.

### 4. Tests

```bash
pytest tests/
```

---

## Estructura del proyecto

```
chess_prediction/
├── app.py                  # Interfaz Streamlit
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                # Datos crudos de Chess.com
│   └── processed/          # Dataset limpio + gráficos EDA
├── models/
│   ├── best_model.pkl      # Modelo entrenado (se genera con train.py)
│   └── model_metadata.json # Métricas y metadatos
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── chess_api.py        # Cliente API Chess.com (perfil + rating actual)
│   ├── data_loader.py      # Descarga y procesa el dataset de Kaggle
│   ├── data_collection.py  # Recolección por historial (Opción B futura)
│   ├── feature_engineering.py  # Vector de características (Opción A, 6 variables)
│   ├── preprocessing.py    # Limpieza y pipelines sklearn
│   ├── exploratory_analysis.py  # EDA y visualizaciones
│   ├── train.py            # Entrenamiento y comparación
│   ├── evaluate.py         # Métricas y evaluación
│   ├── explainability.py   # SHAP
│   ├── prediction.py       # Motor de predicción
│   └── monitoring.py       # PSI y detección de drift
└── tests/
    ├── __init__.py
    ├── test_api.py
    ├── test_preprocessing.py
    └── test_prediction.py
```

---

## Limitaciones

- El modelo se basa en el rating actual y el contexto de la partida; no tiene en cuenta el estado de ánimo, la preparación específica ni factores psicológicos.
- La clase Empate está habitualmente infrarepresentada en los datasets de ajedrez online (~3-4 % del dataset), por lo que su recall es bajo.
- Se requiere conexión a internet para consultar la API de Chess.com en producción.
- Las predicciones son estimaciones probabilísticas, no certezas.
- La razón de victoria/empate aún no se predice; es trabajo futuro.
- El modelo puede degradarse con el tiempo; se recomienda reentrenar periódicamente con datos frescos.

---

## Licencia

Proyecto de uso académico / educativo.
