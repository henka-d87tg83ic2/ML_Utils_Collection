# -*- coding: utf-8 -*-
"""anomaly_detection_utils.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KcwZl5wWLNdBOqH09QSd9oSnVmfvXcwP

1. インポート・環境設定（ロギング含む）
2. データロード・前処理（標準化など）
3. 異常検知アルゴリズム適用（Isolation Forest, One-Class SVM対応予定）
4. モデル評価・異常スコア分布可視化
5. モデル保存・読込（Google Drive対応）
6. ファイルアップロード支援
"""

# ================================
# インポートと環境設定
# ================================

import os
import joblib
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from google.colab import drive
from google.colab import files

from typing import Any, Dict, List, Optional, Tuple, Union

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
# ================================
# データロード・前処理
# ================================

def load_csv_data(file_path: str, features_to_drop: Optional[List[str]] = None) -> pd.DataFrame:
    """CSVファイルからデータをロードする"""
    try:
        if not os.path.exists(file_path) and '/content/drive' not in file_path:
            file_path = os.path.join('/content/drive/MyDrive/', file_path)

        df = pd.read_csv(file_path)
        logger.info(f"📊 データロード完了: {df.shape}")

        if features_to_drop:
            df = df.drop(columns=features_to_drop)

        return df
    except Exception as e:
        logger.error(f"❌ CSVロード失敗: {e}")
        return None

def standardize_data(X: pd.DataFrame) -> pd.DataFrame:
    """特徴量を標準化する"""
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logger.info("✅ 特徴量標準化完了")
        return pd.DataFrame(X_scaled, columns=X.columns)
    except Exception as e:
        logger.error(f"❌ 標準化失敗: {e}")
        return None
# ================================
# 異常検知アルゴリズム適用
# ================================

def train_isolation_forest(X: pd.DataFrame, contamination: float = 0.05) -> Any:
    """Isolation Forestで異常検知モデルを学習"""
    try:
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(X)
        logger.info("✅ Isolation Forest学習完了")
        return model
    except Exception as e:
        logger.error(f"❌ Isolation Forest学習失敗: {e}")
        return None

def train_one_class_svm(X: pd.DataFrame, nu: float = 0.05, kernel: str = 'rbf') -> Any:
    """One-Class SVMで異常検知モデルを学習"""
    try:
        model = OneClassSVM(nu=nu, kernel=kernel)
        model.fit(X)
        logger.info("✅ One-Class SVM学習完了")
        return model
    except Exception as e:
        logger.error(f"❌ One-Class SVM学習失敗: {e}")
        return None
# ================================
# モデル評価・異常スコア分布可視化
# ================================

def evaluate_anomaly_detection(model: Any, X: pd.DataFrame) -> pd.Series:
    """異常スコアを計算して返す"""
    try:
        scores = model.decision_function(X)
        logger.info("✅ 異常スコア計算完了")
        return pd.Series(scores, index=X.index)
    except Exception as e:
        logger.error(f"❌ 異常スコア計算失敗: {e}")
        return pd.Series()

def plot_anomaly_scores(scores: pd.Series, threshold: Optional[float] = None, title: str = "Anomaly Scores") -> None:
    """異常スコアのヒストグラムをプロット"""
    try:
        plt.figure(figsize=(10, 5))
        sns.histplot(scores, bins=50, kde=True)
        if threshold is not None:
            plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold = {threshold}")
        plt.title(title)
        plt.xlabel("Anomaly Score")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid()
        plt.show()
    except Exception as e:
        logger.error(f"❌ 異常スコアプロット失敗: {e}")
# ================================
# モデル保存・読込
# ================================

def save_model_to_drive(model: Any, relative_path: str) -> None:
    """異常検知モデルをGoogle Driveに保存"""
    try:
        full_path = os.path.join('/content/drive/MyDrive/', relative_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        joblib.dump(model, full_path)
        logger.info(f"✅ モデル保存完了: {full_path}")
    except Exception as e:
        logger.error(f"❌ モデル保存失敗: {e}")

def load_model_from_drive(relative_path: str) -> Any:
    """Google Driveから異常検知モデルをロード"""
    try:
        full_path = os.path.join('/content/drive/MyDrive/', relative_path)
        return joblib.load(full_path)
    except Exception as e:
        logger.error(f"❌ モデルロード失敗: {e}")
        return None
# ================================
# ファイルアップロード支援
# ================================

def upload_file_from_local() -> Dict[str, Any]:
    """ローカルPCからファイルをアップロード"""
    logger.info("📂 ローカルファイルをアップロードしてください")
    uploaded = files.upload()
    logger.info(f"✅ アップロード完了: {list(uploaded.keys())}")
    return uploaded