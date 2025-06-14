# -*- coding: utf-8 -*-
"""reinforcement_learning_utils.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1SOpNI_erNMh6kJR6KzX2BkQLfpzRzKb_

1. インポート・環境設定（ロギング含む）
2. 環境セットアップ（例：OpenAI Gym環境）
3. Q-Learning（簡単な表形式）
4. モデル保存・読込（Google Drive対応）
5. 学習ログ・報酬推移の可視化
6. ファイルアップロード支援
"""

# ================================
# インポートと環境設定
# ================================

import os
import joblib
import logging
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import gym
from google.colab import drive
from google.colab import files

from typing import Any, Dict, List, Optional, Tuple, Union

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
# ================================
# 環境セットアップ
# ================================

def create_gym_environment(env_name: str = "FrozenLake-v1") -> gym.Env:
    """OpenAI Gymの環境を作成する"""
    try:
        env = gym.make(env_name)
        logger.info(f"✅ 環境作成成功: {env_name}")
        return env
    except Exception as e:
        logger.error(f"❌ 環境作成失敗: {e}")
        return None
# ================================
# Q-Learning実装
# ================================

def train_q_learning(env: gym.Env,
                     num_episodes: int = 5000,
                     alpha: float = 0.1,
                     gamma: float = 0.99,
                     epsilon: float = 1.0,
                     epsilon_decay: float = 0.995,
                     epsilon_min: float = 0.01) -> Tuple[np.ndarray, List[float]]:
    """
    Q-Learningによる強化学習
    """
    try:
        n_actions = env.action_space.n
        n_states = env.observation_space.n
        Q = np.zeros((n_states, n_actions))
        rewards = []

        for episode in range(num_episodes):
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(Q[state])

                next_state, reward, done, info = env.step(action)

                best_next_action = np.argmax(Q[next_state])
                td_target = reward + gamma * Q[next_state][best_next_action]
                td_error = td_target - Q[state][action]
                Q[state][action] += alpha * td_error

                state = next_state
                total_reward += reward

            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            rewards.append(total_reward)

            if (episode + 1) % 500 == 0:
                logger.info(f"エピソード {episode + 1}/{num_episodes} - 平均報酬: {np.mean(rewards[-500:]):.2f}")

        logger.info("✅ Q-Learningトレーニング完了")
        return Q, rewards
    except Exception as e:
        logger.error(f"❌ Q-Learningトレーニング失敗: {e}")
        return None, []
# ================================
# 学習ログ・報酬推移の可視化
# ================================

def plot_rewards(rewards: List[float], window: int = 100) -> None:
    """報酬推移を可視化する"""
    try:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.figure(figsize=(10, 5))
        plt.plot(moving_avg)
        plt.title("Moving Average of Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.grid()
        plt.show()
    except Exception as e:
        logger.error(f"❌ 報酬プロット失敗: {e}")
# ================================
# モデル保存・読込
# ================================

def save_q_table_to_drive(Q: np.ndarray, relative_path: str) -> None:
    """QテーブルをGoogle Driveに保存する"""
    try:
        full_path = os.path.join('/content/drive/MyDrive/', relative_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        joblib.dump(Q, full_path)
        logger.info(f"✅ Qテーブル保存完了: {full_path}")
    except Exception as e:
        logger.error(f"❌ Qテーブル保存失敗: {e}")

def load_q_table_from_drive(relative_path: str) -> np.ndarray:
    """Google DriveからQテーブルをロードする"""
    try:
        full_path = os.path.join('/content/drive/MyDrive/', relative_path)
        return joblib.load(full_path)
    except Exception as e:
        logger.error(f"❌ Qテーブルロード失敗: {e}")
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