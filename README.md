# ML_Utils_Collection

## 🔹 リポジトリ概要
本リポジトリは、機械学習における主要タスク（分類・回帰・クラスタリング・次元削減・強化学習・時系列予測・異常検知）に対応したユーティリティ関数群をまとめたものです。

各ファイルは役割ごとに整理されており、学習・予測・可視化・保存・ロードを簡潔に実施できます。

---

## 🔹 使用方法（基本フロー）

1. 必要なモジュールをインポート
   ```python
from classification_utils_seiri import train_model, evaluate_model, save_model

# データをロード・前処理
from data_utils import load_csv_data, preprocess_and_split
X, y = load_csv_data('path/to/data.csv', target_column='target')

# 学習・評価・保存
X_train, X_test, y_train, y_test = preprocess_and_split(X, y)
model = train_model(X_train, y_train)
evaluate_model(model, X_test, y_test)
save_model(model, 'model.pkl')
---
## 🔹 ファイル一覧と主な役割

| ファイル名 | 主な役割 |
|:---|:---|
| data_utils.py | データロード・前処理・分割 |
| shap_utils.py | SHAP解析・可視化・保存読込 |
| classification_utils_seiri.py | 分類タスクの学習・評価・保存 |
| regression_utils_seiri.py | 回帰タスクの学習・評価・保存 |
| clustering_utils_seiri.py | クラスタリングの学習・評価・保存 |
| dimensionality_reduction_utils_seiri.py | 次元削減（PCA/t-SNE）学習・適用・保存 |
| reinforcement_learning_utils_seiri.py | 強化学習（Q-Learning簡易版）学習・保存 |
| time_series_utils_seiri.py | 時系列予測（ARIMA）学習・評価・保存 |
| anomaly_detection_utils_seiri.py | 異常検知（IsolationForest/OneClassSVM）学習・評価・保存 |

## 各ファイル概要
- classification_utils.py
- regression_utils.py
- clustering_utils.py
- dimensionality_reduction_utils.py
- reinforcement_learning_utils.py
- time_series_utils.py
- anomaly_detection_utils.py
