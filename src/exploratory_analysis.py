"""
exploratory_analysis.py
=======================
Exploratory Data Analysis for the chess prediction dataset.

Generates publication-ready figures with titles, axis labels, and
text conclusions for each analysis.
"""

import os
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless/server use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Use a professional, consistent style
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

PALETTE = "Set2"
TARGET_LABELS = {0: "Victoria A", 1: "Victoria B", 2: "Empate"}


class ExploratoryAnalyzer:
    """
    Generates EDA plots from a processed chess prediction DataFrame.

    All plot methods accept an optional save_path; if provided the figure
    is saved to disk and the figure object is still returned.
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_fig(fig: plt.Figure, path: Optional[str]) -> plt.Figure:
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.savefig(path, bbox_inches="tight", dpi=120)
        return fig

    @staticmethod
    def _target_series(df: pd.DataFrame) -> pd.Series:
        return df["target"].map(TARGET_LABELS)

    # ------------------------------------------------------------------
    # Individual plots
    # ------------------------------------------------------------------

    def plot_result_distribution(
        self, df: pd.DataFrame, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Bar chart of match outcome distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Distribución de Resultados", fontsize=14, fontweight="bold")

        counts = df["target"].value_counts().sort_index()
        counts.index = [TARGET_LABELS.get(i, i) for i in counts.index]

        sns.barplot(x=counts.index, y=counts.values, palette=PALETTE, ax=axes[0])
        axes[0].set_title("Frecuencia absoluta")
        axes[0].set_xlabel("Resultado")
        axes[0].set_ylabel("Partidas")
        for bar in axes[0].patches:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{int(bar.get_height())}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        pct = counts / counts.sum() * 100
        axes[1].pie(
            pct.values,
            labels=pct.index,
            autopct="%1.1f%%",
            colors=sns.color_palette(PALETTE, len(pct)),
            startangle=140,
        )
        axes[1].set_title("Porcentaje por clase")

        plt.tight_layout()
        return self._save_fig(fig, save_path)

    def plot_rating_distribution(
        self, df: pd.DataFrame, save_path: Optional[str] = None
    ) -> plt.Figure:
        """KDE + histogram of player ratings."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Distribución de Ratings", fontsize=14, fontweight="bold")

        for ax, col, label in [
            (axes[0], "rating_a", "Rating Jugador A"),
            (axes[1], "rating_b", "Rating Jugador B"),
        ]:
            if col in df.columns:
                sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue", bins=20)
                ax.set_title(label)
                ax.set_xlabel("Rating ELO")
                ax.set_ylabel("Frecuencia")
                mean_val = df[col].mean()
                ax.axvline(mean_val, color="red", linestyle="--", label=f"Media: {mean_val:.0f}")
                ax.legend()

        plt.tight_layout()
        return self._save_fig(fig, save_path)

    def plot_rating_diff_vs_result(
        self, df: pd.DataFrame, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Box plot of rating difference by match result."""
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle("Diferencia de Rating vs Resultado", fontsize=14, fontweight="bold")

        if "rating_diff" in df.columns:
            plot_df = df.copy()
            plot_df["resultado"] = plot_df["target"].map(TARGET_LABELS)
            sns.boxplot(data=plot_df, x="resultado", y="rating_diff", palette=PALETTE, ax=ax)
            ax.axhline(0, color="red", linestyle="--", alpha=0.7, label="Sin diferencia")
            ax.set_xlabel("Resultado")
            ax.set_ylabel("Diferencia de Rating (A − B)")
            ax.set_title("¿Influye la diferencia de rating en el resultado?")
            ax.legend()

        plt.tight_layout()
        return self._save_fig(fig, save_path)

    def plot_white_advantage(
        self, df: pd.DataFrame, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Win rate by color."""
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle("Ventaja de las Blancas", fontsize=14, fontweight="bold")

        if "color_a_is_white" in df.columns:
            white_games = df[df["color_a_is_white"] == 1]
            black_games = df[df["color_a_is_white"] == 0]

            categories = ["Blancas (A)", "Negras (A)"]
            win_rates = [
                (white_games["target"] == 0).mean() * 100,
                (black_games["target"] == 0).mean() * 100,
            ]

            bars = ax.bar(categories, win_rates, color=["#FAFAFA", "#444444"],
                         edgecolor="black", width=0.5)
            for bar, val in zip(bars, win_rates):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                )
            ax.set_ylabel("Tasa de victorias de A (%)")
            ax.set_ylim(0, 100)
            ax.set_title("Victoria de A según color asignado")

        plt.tight_layout()
        return self._save_fig(fig, save_path)

    def plot_time_class_comparison(
        self, df: pd.DataFrame, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Stacked bar chart of results per time class."""
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle("Comparación por Ritmo de Juego", fontsize=14, fontweight="bold")

        time_map = {0: "Bullet", 1: "Blitz", 2: "Rapid"}

        if "time_class_encoded" in df.columns:
            plot_df = df.copy()
            plot_df["time_class_label"] = plot_df["time_class_encoded"].map(time_map)
            plot_df["resultado"] = plot_df["target"].map(TARGET_LABELS)

            ct = pd.crosstab(
                plot_df["time_class_label"],
                plot_df["resultado"],
                normalize="index",
            ) * 100
            ct.plot(kind="bar", stacked=True, colormap=PALETTE, ax=ax)
            ax.set_xlabel("Ritmo de juego")
            ax.set_ylabel("Porcentaje (%)")
            ax.set_title("Distribución de resultados por ritmo de juego")
            ax.legend(title="Resultado", bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

        plt.tight_layout()
        return self._save_fig(fig, save_path)

    def plot_correlation_matrix(
        self, df: pd.DataFrame, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Heatmap of numeric feature correlations."""
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(14, 12))
        fig.suptitle("Matriz de Correlaciones", fontsize=14, fontweight="bold")
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=ax,
            linewidths=0.5,
            annot_kws={"size": 7},
        )
        ax.set_title("Correlación entre variables predictivas y objetivo")
        plt.tight_layout()
        return self._save_fig(fig, save_path)

    def plot_feature_distributions(
        self, df: pd.DataFrame, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Small-multiples histograms for all numeric features."""
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != "target"
        ]
        n_cols = 4
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
        fig.suptitle("Distribución de Variables Predictivas", fontsize=14, fontweight="bold")

        axes_flat = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

        for i, col in enumerate(numeric_cols):
            sns.histplot(df[col].dropna(), kde=True, ax=axes_flat[i], bins=20,
                        color="steelblue")
            axes_flat[i].set_title(col, fontsize=9)
            axes_flat[i].set_xlabel("")
            axes_flat[i].set_ylabel("")

        for j in range(len(numeric_cols), len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        return self._save_fig(fig, save_path)

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def generate_full_report(
        self, df: pd.DataFrame, output_dir: str
    ) -> Dict[str, object]:
        """
        Generate and save all EDA plots, returning paths and text conclusions.

        Parameters
        ----------
        df : pd.DataFrame
        output_dir : str

        Returns
        -------
        dict with keys 'plots' (dict of name->path) and 'conclusions' (list of str)
        """
        os.makedirs(output_dir, exist_ok=True)

        plots = {}
        figures = [
            ("result_distribution", self.plot_result_distribution),
            ("rating_distribution", self.plot_rating_distribution),
            ("rating_diff_vs_result", self.plot_rating_diff_vs_result),
            ("white_advantage", self.plot_white_advantage),
            ("time_class_comparison", self.plot_time_class_comparison),
            ("correlation_matrix", self.plot_correlation_matrix),
            ("feature_distributions", self.plot_feature_distributions),
        ]

        for name, method in figures:
            path = os.path.join(output_dir, f"{name}.png")
            try:
                fig = method(df, save_path=path)
                plt.close(fig)
                plots[name] = path
            except Exception as e:
                plots[name] = f"Error: {e}"

        conclusions = self._generate_conclusions(df)

        return {"plots": plots, "conclusions": conclusions}

    def _generate_conclusions(self, df: pd.DataFrame) -> list:
        """Compute text conclusions from the dataset statistics."""
        conclusions = []

        if "target" in df.columns:
            counts = df["target"].value_counts(normalize=True) * 100
            win_a = counts.get(0, 0)
            win_b = counts.get(1, 0)
            draw = counts.get(2, 0)
            conclusions.append(
                f"Distribución de resultados: Victoria A {win_a:.1f}%, "
                f"Victoria B {win_b:.1f}%, Empate {draw:.1f}%. "
                f"{'El dataset está desbalanceado.' if max(win_a, win_b, draw) > 50 else 'Distribución relativamente balanceada.'}"
            )

        if "rating_diff" in df.columns and "target" in df.columns:
            mean_diff_wina = df[df["target"] == 0]["rating_diff"].mean()
            mean_diff_winb = df[df["target"] == 1]["rating_diff"].mean()
            conclusions.append(
                f"Cuando A gana, la diferencia de rating media es {mean_diff_wina:.0f}. "
                f"Cuando B gana, es {mean_diff_winb:.0f}. "
                "Esto confirma que el rating es un predictor relevante."
            )

        if "color_a_is_white" in df.columns and "target" in df.columns:
            white_wr = (
                df[df["color_a_is_white"] == 1]["target"] == 0
            ).mean() * 100
            black_wr = (
                df[df["color_a_is_white"] == 0]["target"] == 0
            ).mean() * 100
            conclusions.append(
                f"Tasa de victoria con blancas: {white_wr:.1f}% "
                f"vs con negras: {black_wr:.1f}%. "
                f"{'La ventaja de blancas es visible en este dataset.' if white_wr > black_wr + 2 else 'La ventaja de blancas no es pronunciada en este dataset.'}"
            )

        return conclusions
