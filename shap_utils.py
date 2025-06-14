# -*- coding: utf-8 -*-
"""shap_utils.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hkfchzNMNnMHjqHe25vlU-8AKMqbwBzb
"""

# ================================================
# shap_utils.py
# SHAP解析用ユーティリティ関数集
# ================================================

import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os
import logging
from typing import Any, Optional, Union
import numpy as np


# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ================================================
# SHAP値計算関数
# ================================================

def compute_shap_values(model: Any, X_sample: pd.DataFrame) -> shap.Explanation:
    """
    SHAP値を計算する関数。モデルの種類に応じて自動でExplainerを初期化。

    Args:
        model (Any): 学習済みモデル（XGBoost, RandomForestなど）
        X_sample (pd.DataFrame): SHAP計算に使う入力データ（通常はX_test）

    Returns:
        shap.Explanation: SHAP値オブジェクト
    """
    try:
        logger.info("🔍 SHAP Explainer を初期化しています...")
        if hasattr(model, "predict_proba"):
            explainer = shap.Explainer(model.predict_proba, X_sample)
        else:
            explainer = shap.Explainer(model, X_sample)

        shap_values = explainer(X_sample)
        logger.info("✅ SHAP値計算完了")
        return shap_values

    except Exception as e:
        logger.error(f"❌ SHAP値計算エラー: {e}")
        return None

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import PartialDependenceDisplay
from scipy.interpolate import griddata

# ================================================
# SHAP可視化関数（Summary／Waterfall／Heatmap／Decision Boundary／PDP／ICE）
# ================================================

def plot_shap_summary(shap_values: shap.Explanation, features: pd.DataFrame) -> None:
    try:
        logger.info("📈 SHAP Summary Plotを描画中...")
        shap.summary_plot(shap_values.values, features)
        logger.info("✅ Summary Plot描画完了")
    except Exception as e:
        logger.error(f"❌ Summary Plot描画エラー: {e}")


def plot_shap_waterfall(shap_values: shap.Explanation, row_index: int = 0) -> None:
    try:
        logger.info(f"📈 SHAP Waterfall Plotを描画中（サンプル index={row_index}）...")
        shap.plots.waterfall(shap_values[row_index])
        logger.info("✅ Waterfall Plot描画完了")
    except Exception as e:
        logger.error(f"❌ Waterfall Plot描画エラー: {e}")


def plot_shap_interaction_heatmap_nodiag(
    interaction_matrix: np.ndarray,
    feature_names: list[str],
    title: str = "SHAP Interaction Heatmap (No Diagonal)",
    figsize: tuple = (12, 10),
    center: float = 0.0,
    vmin: float = None,
    vmax: float = None
) -> None:
    try:
        logger.info("📊 SHAP Interaction Heatmap（対角除外）を描画中...")
        import seaborn as sns
        matrix_nodiag = interaction_matrix.copy()
        np.fill_diagonal(matrix_nodiag, 0)
        plt.figure(figsize=figsize)
        sns.heatmap(
            matrix_nodiag,
            xticklabels=feature_names,
            yticklabels=feature_names,
            cmap="Reds",
            center=center,
            vmin=vmin,
            vmax=vmax,
            square=True,
            linewidths=0.5,
            cbar_kws={"label": "Mean SHAP Interaction"}
        )
        plt.title(title, fontsize=16)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        logger.info("✅ 対角除外ヒートマップ描画完了")
    except Exception as e:
        logger.error(f"❌ 対角除外ヒートマップ描画エラー: {e}")


def plot_shap_decision_boundary(model, explainer, X_scaled, shap_values, feature_x, feature_y,
                                class_index=1, cmap_boundary="RdBu", cmap_shap="viridis",
                                figsize=(10, 8), grid_resolution=100, title=None):
    try:
        logger.info(f"📈 SHAP + 決定境界を可視化中: {feature_x} × {feature_y}")
        fx_idx = X_scaled.columns.get_loc(feature_y)
        x_vals = np.linspace(X_scaled[feature_x].min(), X_scaled[feature_x].max(), grid_resolution)
        y_vals = np.linspace(X_scaled[feature_y].min(), X_scaled[feature_y].max(), grid_resolution)
        xx, yy = np.meshgrid(x_vals, y_vals)
        grid = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=[feature_x, feature_y])
        for col in X_scaled.columns:
            if col not in [feature_x, feature_y]:
                grid[col] = X_scaled[col].mean()
        grid = grid[X_scaled.columns]
        Z = model.predict_proba(grid)[:, class_index].reshape(xx.shape)
        shap_val = shap_values.values[:, fx_idx, class_index]
        df_plot = pd.DataFrame({feature_x: X_scaled[feature_x], feature_y: X_scaled[feature_y], "SHAP": shap_val})
        plt.figure(figsize=figsize)
        plt.contourf(xx, yy, Z, levels=20, cmap=cmap_boundary, alpha=0.6)
        sc = plt.scatter(df_plot[feature_x], df_plot[feature_y], c=df_plot["SHAP"], cmap=cmap_shap, edgecolor="k")
        plt.colorbar(sc, label="SHAP value")
        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        plt.title(title or f"SHAP + Decision Boundary: {feature_x} × {feature_y}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        logger.info("✅ SHAP + 決定境界の重ね描画完了")
    except Exception as e:
        logger.error(f"❌ 決定境界可視化エラー: {e}")


def plot_shap_meshgrid(shap_values, X_scaled, feature_x, feature_y, class_index=1,
                       resolution=100, cmap="coolwarm", figsize=(10, 8), title=None):
    try:
        fx_idx = X_scaled.columns.get_loc(feature_y)
        x = X_scaled[feature_x].values
        y = X_scaled[feature_y].values
        z = shap_values.values[:, fx_idx, class_index]
        xi = np.linspace(x.min(), x.max(), resolution)
        yi = np.linspace(y.min(), y.max(), resolution)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x, y), z, (xi, yi), method='cubic')
        plt.figure(figsize=figsize)
        contour = plt.contourf(xi, yi, zi, levels=20, cmap=cmap, alpha=0.8)
        sc = plt.scatter(x, y, c=z, cmap=cmap, edgecolor='k', s=40)
        plt.colorbar(contour, label='SHAP value')
        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        plt.title(title or f"SHAP 2D Mesh: {feature_x} × {feature_y}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        logger.info("✅ SHAP メッシュプロット描画完了")
    except Exception as e:
        logger.error(f"❌ SHAPメッシュ描画エラー: {e}")


def plot_pdp_1d(model, X_scaled, feature_name, class_index=1):
    try:
        disp = PartialDependenceDisplay.from_estimator(
            model, X_scaled, features=[feature_name], kind="average",
            grid_resolution=50, feature_names=X_scaled.columns, target=class_index
        )
        disp.figure_.set_size_inches(8, 6)
    except Exception as e:
        print(f"[Error] plot_pdp_1d: {e}")


def plot_pdp_ice(model, X_scaled, feature_name, class_index=1):
    try:
        disp = PartialDependenceDisplay.from_estimator(
            model, X_scaled, features=[feature_name], kind="both",
            grid_resolution=50, feature_names=X_scaled.columns, target=class_index
        )
        disp.figure_.set_size_inches(8, 6)
    except Exception as e:
        print(f"[Error] plot_pdp_ice: {e}")


def plot_pdp_2d(model, X_scaled, feature_pair, class_index=1):
    try:
        disp = PartialDependenceDisplay.from_estimator(
            model, X_scaled, features=[feature_pair], kind="average",
            grid_resolution=50, feature_names=X_scaled.columns, target=class_index
        )
        disp.figure_.set_size_inches(8, 6)
    except Exception as e:
        print(f"[Error] plot_pdp_2d: {e}")

# ================================================
# SHAP 3D可視化関数（拡張版）
# ================================================
def plot_shap_3d(shap_values, X_sample, feature_x, feature_y,
                 shap_feature=None, class_index=1,
                 width=1200, height=900, renderer="colab"):
    print("📊 plot_shap_3d() 実行中...")
    try:
        import pandas as pd
        import plotly.express as px
        import shap
        import numpy as np

        if isinstance(shap_values, shap.Explanation):
            values = shap_values.values
        else:
            values = shap_values

        if values.ndim == 3:
            values = values[:, :, class_index]

        if shap_feature:
            if shap_feature not in X_sample.columns:
                raise ValueError(f"指定された特徴名 '{shap_feature}' は X_sample に存在しません。")
            feature_index = X_sample.columns.get_loc(shap_feature)
            shap_z = values[:, feature_index].flatten()
            title_z = shap_feature
        else:
            shap_z = values.mean(axis=1).flatten()
            title_z = "SHAP mean"

        df_plot = pd.DataFrame({
            "X": X_sample[feature_x].values.flatten(),
            "Y": X_sample[feature_y].values.flatten(),
            "SHAP": shap_z
        })

        fig = px.scatter_3d(
            df_plot, x="X", y="Y", z="SHAP",
            color="SHAP", opacity=0.7,
            title=f"3D SHAP Plot: {feature_x} × {feature_y} × {title_z}",
            color_continuous_scale="Greys_r"
        )

        fig.update_layout(
            width=width,
            height=height,
            paper_bgcolor="rgba(0,0,0,0)",  # 外枠：透明
            scene=dict(
                xaxis=dict(
                    title=feature_x,
                    showbackground=True,
                    backgroundcolor="burlywood",
                    gridcolor="white",
                    title_font=dict(size=22),
                    tickfont=dict(size=18)
                ),
                yaxis=dict(
                    title=feature_y,
                    showbackground=True,
                    backgroundcolor="burlywood",
                    gridcolor="white",
                    title_font=dict(size=22),
                    tickfont=dict(size=18)
                ),
                zaxis=dict(
                    title="SHAP Value",
                    showbackground=True,
                    backgroundcolor="burlywood",
                    gridcolor="white",
                    title_font=dict(size=22),
                    tickfont=dict(size=18)
                )
            )
        )

        fig.show(renderer=renderer)

    except Exception as e:
        print(f"❌ 3D SHAPプロットエラー: {e}")

# ================================================
# SHAP値保存・読み込み関数
# ================================================

def save_shap_values(shap_values: shap.Explanation, path: str) -> None:
    """
    SHAP値をファイルに保存する関数（pkl形式）

    Args:
        shap_values (shap.Explanation): 保存するSHAP値
        path (str): 保存先パス
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(shap_values, path)
        logger.info(f"✅ SHAP値保存完了: {path}")

    except Exception as e:
        logger.error(f"❌ SHAP保存エラー: {e}")


def load_shap_values(path: str) -> Optional[shap.Explanation]:
    """
    ファイルからSHAP値を読み込む関数

    Args:
        path (str): 読み込み元パス

    Returns:
        shap.Explanation: 読み込んだSHAP値オブジェクト
    """
    try:
        shap_values = joblib.load(path)
        logger.info(f"✅ SHAP値読込完了: {path}")
        return shap_values

    except Exception as e:
        logger.error(f"❌ SHAP読込エラー: {e}")
        return None

# ================================================
# SHAP Interaction Heatmap（明るさ＝強さ）
# ================================================
def plot_interaction_heatmap(
    interaction_matrix, feature_names, title="SHAP Interaction Heatmap",
    cmap="OrRd", figsize=(12, 10), linewidths=0.5, annot=False
):
    """
    SHAP相互作用行列のヒートマップを可視化する関数。

    Parameters:
    ----------
    interaction_matrix : np.ndarray
        2次元の SHAP interaction 値の行列（平均絶対値を取ったものを想定）

    feature_names : list or pd.Index
        行列の各軸に対応する特徴名

    title : str
        プロットのタイトル

    cmap : str
        色マップ（デフォルト: "OrRd"）

    figsize : tuple
        表示サイズ（横, 縦）

    linewidths : float
        セル境界の線の太さ

    annot : bool
        各セルに数値を表示するか（True推奨は小規模行列時のみ）
    """
    print("📊 plot_interaction_heatmap() 実行中...")
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=figsize)
        sns.heatmap(
            interaction_matrix,
            xticklabels=feature_names,
            yticklabels=feature_names,
            cmap=cmap,
            center=0,
            linewidths=linewidths,
            annot=annot
        )
        plt.title(f"{title}（明るさ＝強さ）", fontsize=14)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ ヒートマップ描画エラー: {e}")

