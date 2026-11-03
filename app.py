"""
app.py
======
Streamlit web interface for the Chess Match Predictor.

Run with:
    streamlit run app.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="♟️ Chess Match Predictor",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def model_trained() -> bool:
    return MODEL_PATH.exists()


def load_metadata() -> dict:
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def result_badge(label: str, pct: float, color: str) -> None:
    """Render a single probability result block."""
    st.metric(label=label, value=f"{pct:.1f}%")


# -----------------------------------------------------------------------
# Navigation
# -----------------------------------------------------------------------
page = st.sidebar.radio(
    "Navegación",
    ["🏠 Inicio", "🔮 Predicción", "📊 Análisis", "ℹ️ Información del Modelo"],
)

# ===================================================================
# PAGE 1 — INICIO
# ===================================================================
if page == "🏠 Inicio":
    st.title("♟️ Chess Match Predictor")
    st.subheader("Predicción de resultados de partidas de ajedrez con Machine Learning")

    st.markdown(
        """
        Este sistema utiliza historial real de partidas de Chess.com para predecir el
        resultado más probable entre dos jugadores antes de que la partida comience.

        ---
        ### ¿Cómo funciona?

        1. **Consulta la API** de Chess.com para obtener el rating actual de cada jugador
           en la modalidad elegida.
        2. **Construye el vector de características** de la partida: ratings, diferencia de
           rating, ritmo de juego y si la partida es puntuable.
        3. **Aplica un modelo de ML de dos etapas** entrenado con partidas reales para estimar
           las probabilidades (primero empate, luego color ganador).
        4. **Muestra los resultados** con un desglose visual y explicaciones SHAP de las variables
           más influyentes.

        ---
        ### Instrucciones de uso

        | Paso | Acción |
        |------|--------|
        | 1 | Ve a **🔮 Predicción** en el menú lateral. |
        | 2 | Introduce los nombres de usuario de Chess.com de ambos jugadores. |
        | 3 | Selecciona el color asignado al Jugador A. |
        | 4 | Elige el ritmo de juego (Bullet, Blitz, Rapid o Daily). |
        | 5 | Indica si la partida es puntuable. |
        | 6 | Pulsa **Predecir resultado** y espera unos segundos. |

        ---
        ### Limitaciones del sistema

        - Las predicciones se basan **únicamente** en el rating actual y el contexto de la
          partida; el ajedrez es un juego complejo con muchos factores no capturables en datos.
        - El modelo requiere que ambos jugadores tengan **rating disponible** en la modalidad
          seleccionada.
        - Los resultados son estimaciones probabilísticas, **no garantías**.
        - Se necesita conexión a internet para consultar la API de Chess.com.
        """
    )

    if not model_trained():
        st.warning(
            "⚠️ **Modelo no entrenado.** Para usar las predicciones, primero ejecuta:\n\n"
            "```\npython -m src.data_loader\npython -m src.train\n```"
        )
    else:
        meta = load_metadata()
        algo = meta.get("algorithm", "desconocido")
        date = meta.get("training_date", "desconocida")[:10]
        st.success(f"✅ Modelo listo ({algo}, entrenado el {date})")

# ===================================================================
# PAGE 2 — PREDICCIÓN
# ===================================================================
elif page == "🔮 Predicción":
    st.title("🔮 Predicción de Partido")

    if not model_trained():
        st.error(
            "No se encontró ningún modelo entrenado. "
            "Ejecuta primero:\n\n"
            "```\npython -m src.data_loader\npython -m src.train\n```"
        )
        st.stop()

    # ----------------------------------------------------------------
    # Input form
    # ----------------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        username_a = st.text_input(
            "Usuario A (Chess.com)",
            placeholder="ej: magnuscarlsen",
            help="Nombre de usuario exacto de Chess.com",
        ).strip()
        color_a = st.selectbox(
            "Color del Jugador A",
            ["white", "black"],
            format_func=lambda x: "Blancas ♔" if x == "white" else "Negras ♚",
        )

    with col2:
        username_b = st.text_input(
            "Usuario B (Chess.com)",
            placeholder="ej: hikaru",
        ).strip()
        time_class = st.selectbox(
            "Ritmo de juego",
            ["blitz", "bullet", "rapid", "daily"],
            format_func=lambda x: x.capitalize(),
        )

    rated = st.checkbox(
        "Partida puntuable (rated)",
        value=True,
        help="Si hay puntos de ELO en juego",
    )

    predict_btn = st.button("🔮 Predecir resultado", type="primary", use_container_width=True)

    if predict_btn:
        if not username_a or not username_b:
            st.error("Introduce los nombres de usuario de ambos jugadores.")
            st.stop()

        if username_a.lower() == username_b.lower():
            st.error("Los dos jugadores deben ser diferentes.")
            st.stop()

        from src.prediction import PredictionEngine

        engine = PredictionEngine()

        with st.spinner(f"Consultando Chess.com y calculando predicción..."):
            result = engine.predict(
                username_a=username_a,
                username_b=username_b,
                color_a=color_a,
                time_class=time_class,
                rated=bool(rated),
                model_path=str(MODEL_PATH),
            )

        # ----------------------------------------------------------------
        # Warnings
        # ----------------------------------------------------------------
        for w in result.get("warnings", []):
            st.warning(w)

        if not result.get("success"):
            st.error(f"❌ Error: {result.get('error')}")
            st.stop()

        probs = result["probabilities"]
        player_a = result["player_a"]
        player_b = result["player_b"]
        predicted = result["predicted_class"]

        # ----------------------------------------------------------------
        # Prediction result
        # ----------------------------------------------------------------
        st.markdown("---")
        st.subheader("📊 Probabilidades de resultado")

        label_map = {
            "win_a": f"Victoria {username_a}",
            "win_b": f"Victoria {username_b}",
            "draw": "Empate",
        }
        predicted_label = label_map.get(predicted, predicted)
        st.success(f"**Resultado más probable: {predicted_label}**")

        # Horizontal bar chart
        fig, ax = plt.subplots(figsize=(8, 2.5))
        categories = [label_map["win_a"], label_map["draw"], label_map["win_b"]]
        values = [probs["win_a"], probs["draw"], probs["win_b"]]
        colors_bar = ["#2196F3", "#9E9E9E", "#F44336"]
        bars = ax.barh(categories, values, color=colors_bar, edgecolor="white", height=0.5)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                fontsize=11,
            )
        ax.set_xlim(0, 110)
        ax.set_xlabel("Probabilidad (%)")
        ax.set_title("Distribución de probabilidades")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Metrics row
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric(label_map["win_a"], f"{probs['win_a']:.1f}%")
        mcol2.metric("Empate", f"{probs['draw']:.1f}%")
        mcol3.metric(label_map["win_b"], f"{probs['win_b']:.1f}%")

        # ----------------------------------------------------------------
        # Player cards
        # ----------------------------------------------------------------
        st.markdown("---")
        st.subheader("👥 Estadísticas de los jugadores")

        def player_card(player: dict, label: str):
            rating = player.get("rating", "N/D")
            color_icon = "♔ Blancas" if player.get("color") == "white" else "♚ Negras"

            st.markdown(f"### {label}: **{player['username']}**")
            st.write(f"Rating actual: **{rating}** | Color: {color_icon}")

        pcol1, pcol2 = st.columns(2)
        with pcol1:
            player_card(player_a, "Jugador A")
        with pcol2:
            player_card(player_b, "Jugador B")

        # ----------------------------------------------------------------
        # Match context
        # ----------------------------------------------------------------
        st.markdown("---")
        st.subheader("📋 Contexto de la partida")
        features = result.get("features_used", {})
        ctx_data = {
            "Variable": [
                "Rating blancas", "Rating negras", "Diferencia de rating",
                "Ritmo de juego", "Puntuable",
            ],
            "Valor": [
                int(features.get("white_rating", 0)),
                int(features.get("black_rating", 0)),
                f"{features.get('rating_diff', 0):+.0f}",
                time_class.capitalize(),
                "Sí" if int(features.get("rated", 0)) == 1 else "No",
            ],
        }
        st.dataframe(pd.DataFrame(ctx_data), use_container_width=True)

        # ----------------------------------------------------------------
        # SHAP explanation
        # ----------------------------------------------------------------
        shap_exp = result.get("shap_explanation")
        if shap_exp and shap_exp.get("top_features"):
            st.markdown("---")
            st.subheader("🔍 ¿Por qué esta predicción?")
            st.caption(
                "Las siguientes variables influyeron más en la predicción "
                "(basado en valores SHAP):"
            )
            shap_data = []
            for item in shap_exp["top_features"]:
                direction_icon = "📈" if item["direction"] == "positive" else "📉"
                shap_data.append(
                    {
                        "Variable": item["feature"],
                        "Influencia": f"{direction_icon} {item['direction']}",
                        "Valor SHAP": f"{item['shap_value']:.4f}",
                    }
                )
            st.dataframe(pd.DataFrame(shap_data), use_container_width=True)

        st.markdown("---")
        st.caption(
            "⚠️ Esta predicción es una estimación estadística basada en historial reciente. "
            "El ajedrez contiene una gran componente de variabilidad individual que ningún "
            "modelo puede capturar completamente."
        )

# ===================================================================
# PAGE 3 — ANÁLISIS
# ===================================================================
elif page == "📊 Análisis":
    st.title("📊 Análisis Exploratorio de Datos")

    dataset_path = DATA_PROCESSED_DIR / "dataset.csv"

    if not dataset_path.exists():
        st.info(
            "No se encontró ningún dataset procesado. "
            "Para generarlo, ejecuta:\n\n"
            "```\npython -m src.data_loader\n```"
        )
        st.stop()

    df = pd.read_csv(dataset_path)
    st.success(f"Dataset cargado: **{len(df)} partidas**")

    from src.exploratory_analysis import ExploratoryAnalyzer

    analyzer = ExploratoryAnalyzer()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Resultados", "Ratings", "Ventaja de color", "Correlaciones"]
    )

    with tab1:
        st.subheader("Distribución de resultados")
        fig = analyzer.plot_result_distribution(df)
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        st.subheader("Distribución de ratings")
        fig = analyzer.plot_rating_distribution(df)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Diferencia de rating vs resultado")
        fig = analyzer.plot_rating_diff_vs_result(df)
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        st.subheader("Ventaja de las blancas")
        fig = analyzer.plot_white_advantage(df)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Comparación por ritmo de juego")
        fig = analyzer.plot_time_class_comparison(df)
        st.pyplot(fig)
        plt.close(fig)

    with tab4:
        st.subheader("Matriz de correlaciones")
        fig = analyzer.plot_correlation_matrix(df)
        st.pyplot(fig)
        plt.close(fig)

    # Show saved EDA images if present
    eda_images = list(DATA_PROCESSED_DIR.glob("*.png"))
    if eda_images:
        st.markdown("---")
        st.subheader("Imágenes guardadas del análisis")
        for img_path in sorted(eda_images):
            if img_path.stem not in ("model_comparison",):
                st.image(str(img_path), caption=img_path.stem, use_column_width=True)

# ===================================================================
# PAGE 4 — INFORMACIÓN DEL MODELO
# ===================================================================
elif page == "ℹ️ Información del Modelo":
    st.title("ℹ️ Información del Modelo")

    if not model_trained():
        st.warning(
            "No hay modelo entrenado. Ejecuta:\n\n"
            "```\npython -m src.data_loader\npython -m src.train\n```"
        )
        st.stop()

    meta = load_metadata()

    st.subheader("📋 Detalles del modelo en producción")
    col1, col2 = st.columns(2)
    col1.metric("Algoritmo", meta.get("algorithm", "N/D"))
    col2.metric("Fecha de entrenamiento", meta.get("training_date", "N/D")[:10])

    thresholds = meta.get("thresholds", {})
    chosen_t1 = meta.get("chosen_t1")
    if thresholds or chosen_t1 is not None:
        st.subheader("🎚️ Umbrales del modelo de dos etapas")
        tcol1, tcol2, tcol3 = st.columns(3)
        tcol1.metric("Umbral empate (t1)", f"{thresholds.get('t1', 'N/D')}")
        tcol2.metric("Umbral color (t2)", f"{thresholds.get('t2', 'N/D')}")
        tcol3.metric("t1 elegido (sweep)", f"{chosen_t1 if chosen_t1 is not None else 'N/D'}")

    metrics = meta.get("metrics", {})
    if metrics:
        st.subheader("📈 Métricas en test")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        mc2.metric("F1 Macro", f"{metrics.get('f1_macro', 0):.4f}")
        mc3.metric("Log-loss", f"{metrics.get('log_loss', 0):.4f}")
        mc4.metric("Recall Empate", f"{metrics.get('recall_draw', 0):.4f}")

    # Draw-detection metrics ("tabla de verdad" binaria) + matrices
    draw_metrics = meta.get("draw_metrics", {})
    if draw_metrics:
        with st.expander("🎯 Análisis del modelo: detección de empates"):
            st.caption(
                "Matriz de confusión binaria Empate vs No-Empate. La clase "
                "Empate (~3.5 % del dataset) es muy difícil de predecir con "
                "rating y contexto, por lo que su recall suele ser bajo."
            )
            dc1, dc2, dc3, dc4 = st.columns(4)
            dc1.metric("VP (empates aciertos)", draw_metrics.get("tp", 0))
            dc2.metric("FN (empates perdidos)", draw_metrics.get("fn", 0))
            dc3.metric("FP (falsos empates)", draw_metrics.get("fp", 0))
            dc4.metric("VN (no-empates aciertos)", draw_metrics.get("tn", 0))

            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("Precisión Empate", f"{draw_metrics.get('precision', 0):.4f}")
            pc2.metric("Recall Empate", f"{draw_metrics.get('recall', 0):.4f}")
            pc3.metric("F1 Empate", f"{draw_metrics.get('f1', 0):.4f}")

            cm_img = DATA_PROCESSED_DIR / "confusion_matrix.png"
            if cm_img.exists():
                st.image(
                    str(cm_img),
                    caption="Matriz de confusión (3 clases)",
                    use_column_width=True,
                )
            cm_draw_img = DATA_PROCESSED_DIR / "confusion_matrix_draw.png"
            if cm_draw_img.exists():
                st.image(
                    str(cm_draw_img),
                    caption="Matriz de confusión (Empate vs No-Empate)",
                    use_column_width=True,
                )

    feature_names = meta.get("feature_names", [])
    if feature_names:
        st.subheader("🔧 Variables utilizadas")
        st.write(", ".join(feature_names))

    all_results = meta.get("all_model_results", {})
    if all_results:
        st.subheader("📊 Comparación de todos los modelos")
        rows = []
        for model_name, m in all_results.items():
            rows.append(
                {
                    "Modelo": model_name,
                    "Accuracy": round(m.get("accuracy", 0), 4),
                    "F1 Macro": round(m.get("f1_macro", 0), 4),
                    "Log-loss": round(m.get("log_loss", 9999), 4),
                    "Recall Empate": round(m.get("recall_draw", 0), 4),
                }
            )
        cdf = pd.DataFrame(rows).sort_values("F1 Macro", ascending=False)
        st.dataframe(cdf, use_container_width=True)

    comp_img = DATA_PROCESSED_DIR / "model_comparison.png"
    if comp_img.exists():
        st.image(str(comp_img), caption="Comparación de modelos", use_column_width=True)

    st.markdown("---")
    st.subheader("⚠️ Limitaciones del sistema")
    st.markdown(
        """
        - El modelo predice basándose en **historial reciente**, no en preparación específica
          para la partida ni en el estado mental del jugador.
        - El ajedrez tiene una **alta variabilidad**: incluso con gran diferencia de rating,
          el jugador inferior puede ganar.
        - La clase **Empate** tiende a estar infrarepresentada y puede tener menor recall.
        - Las probabilidades deben interpretarse como estimaciones estadísticas,
          no como certezas.
        - El modelo puede degradarse con el tiempo; se recomienda reentrenar periódicamente.
        """
    )
