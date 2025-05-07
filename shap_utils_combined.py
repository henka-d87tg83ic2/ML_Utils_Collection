# -*- coding: utf-8 -*-
"""shap_utils.py 統合版
SHAP解析の Step1〜3 をサポートする可視化関数集
更新日: 2025-05-07
主な機能: SHAP決定境界図, メッシュ構造, PDP+ICE, 相互作用行列 など
"""

import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
from sklearn.inspection import PartialDependenceDisplay
import plotly.express as px
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def compute_shap_values(model, X_sample):
    try:
        logger.info("🔍 SHAP Explainer 初期化中...")
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

def plot_shap_decision_boundary(model, explainer, X_scaled, shap_values, feature_x, feature_y,
                                class_index=1, cmap_boundary="RdBu", cmap_shap="viridis",
                                figsize=(10, 8), grid_resolution=100, title=None):
    try:
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
    except Exception as e:
        logger.error(f"❌ SHAPメッシュ描画エラー: {e}")

def plot_shap_combined(model, explainer, X_scaled, shap_values, feature_x, feature_y,
                       class_index=1, resolution=100, cmap_boundary="RdBu", cmap_shap="viridis",
                       figsize=(10, 8), title_prefix="Step1&2: "):
    logger.info(f"🎯 Step1&2 統合可視化: {feature_x} × {feature_y}")
    plot_shap_decision_boundary(model, explainer, X_scaled, shap_values, feature_x, feature_y,
                                class_index, cmap_boundary, cmap_shap, figsize, resolution,
                                title=f"{title_prefix}決定境界 × SHAP重ね図")
    plot_shap_meshgrid(shap_values, X_scaled, feature_x, feature_y, class_index,
                       resolution, cmap_shap, figsize, title=f"{title_prefix}SHAP メッシュ構造図")

def compute_interaction_matrix_class1(explainer, X_scaled, class_index=1):
    interaction_values = explainer.shap_interaction_values(X_scaled)
    matrix = interaction_values[:, :, :, class_index].mean(axis=0)
    np.fill_diagonal(matrix, 0)
    return matrix
