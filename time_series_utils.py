# -*- coding: utf-8 -*-
"""time_series_utils.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gJAt7L3Xlx03b76fpwUm8v2h5lMrTPqz

1. インポート・環境設定（ロギング含む）
2. データロード・前処理（欠損補完・標準化）
3. ARIMAによる時系列予測（statsmodels使用）
4. Prophetによる時系列予測（オプション）
5. 予測結果の可視化（実測 vs 予測プロット）
6. モデル保存・読込（Google Drive対応）
7. ファイルアップロード支援
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
from statsmodels.tsa.arima.model import ARIMA
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

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

def load_csv_data(file_path: str, date_column: str, target_column: str) -> pd.DataFrame:
    """CSVファイルから時系列データをロードする"""
    try:
        if not os.path.exists(file_path) and '/content/drive' not in file_path:
            file_path = os.path.join('/content/drive/MyDrive/', file_path)

        df = pd.read_csv(file_path, parse_dates=[date_column])
        df = df[[date_column, target_column]].dropna()
        df = df.sort_values(by=date_column)
        logger.info(f"✅ データロード完了: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"❌ データロード失敗: {e}")
        return None

def standardize_series(series: pd.Series) -> pd.Series:
    """時系列データを標準化する"""
    try:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
        logger.info("✅ 標準化完了")
        return pd.Series(scaled, index=series.index)
    except Exception as e:
        logger.error(f"❌ 標準化失敗: {e}")
        return series
# ================================
# ARIMAによる時系列予測
# ================================

def train_arima_model(series: pd.Series, order: Tuple[int, int, int] = (1, 1, 1)) -> Any:
    """ARIMAモデルを学習"""
    try:
        model = ARIMA(series, order=order)
        fitted_model = model.fit()
        logger.info("✅ ARIMAモデル学習完了")
        return fitted_model
    except Exception as e:
        logger.error(f"❌ ARIMA学習失敗: {e}")
        return None

def forecast_arima(model: Any, steps: int) -> pd.Series:
    """ARIMAモデルで未来予測を行う"""
    try:
        forecast = model.forecast(steps=steps)
        logger.info(f"✅ ARIMA予測完了: {steps}ステップ")
        return forecast
    except Exception as e:
        logger.error(f"❌ ARIMA予測失敗: {e}")
        return None
# ================================
# Prophetによる時系列予測
# ================================

def train_prophet_model(df: pd.DataFrame) -> Optional[Any]:
    """Prophetモデルを学習"""
    if not PROPHET_AVAILABLE:
        logger.error("❌ Prophetがインストールされていません")
        return None
    try:
        model = Prophet()
        model.fit(df.rename(columns={"ds": "ds", "y": "y"}))
        logger.info("✅ Prophetモデル学習完了")
        return model
    except Exception as e:
        logger.error(f"❌ Prophet学習失敗: {e}")
        return None

def forecast_prophet(model: Any, periods: int, freq: str = 'D') -> Optional[pd.DataFrame]:
    """Prophetモデルで未来予測を行う"""
    try:
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)
        logger.info(f"✅ Prophet予測完了: {periods}期間")
        return forecast
    except Exception as e:
        logger.error(f"❌ Prophet予測失敗: {e}")
        return None
# ================================
# 予測結果の可視化
# ================================

def plot_forecast(actual: pd.Series, forecast: pd.Series, title: str = "Forecast vs Actual") -> None:
    """実測値と予測値を比較プロット"""
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(actual.index, actual.values, label="Actual", marker='o')
        plt.plot(forecast.index, forecast.values, label="Forecast", marker='x')
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid()
        plt.show()
    except Exception as e:
        logger.error(f"❌ 予測プロット失敗: {e}")
# ================================
# モデル保存・読込
# ================================

def save_model_to_drive(model: Any, relative_path: str) -> None:
    """時系列モデルをGoogle Driveに保存"""
    try:
        full_path = os.path.join('/content/drive/MyDrive/', relative_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        joblib.dump(model, full_path)
        logger.info(f"✅ モデル保存完了: {full_path}")
    except Exception as e:
        logger.error(f"❌ モデル保存失敗: {e}")

def load_model_from_drive(relative_path: str) -> Any:
    """Google Driveから時系列モデルをロード"""
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