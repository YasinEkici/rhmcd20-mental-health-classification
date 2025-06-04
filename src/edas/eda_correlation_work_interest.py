# eda_correlation_work_interest.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from legacy.preprocessing2 import preprocess_data
from feature_engineering import engineer_features

def plot_correlation_matrix(df: pd.DataFrame, title: str):
    corr = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                cbar_kws={'shrink': .75}, linewidths=0.5)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    raw_path = Path(__file__).parents[1] / 'data' / 'raw' / 'mental_health_finaldata_1.csv'
    df_raw = pd.read_csv(raw_path)

    # 1) Preprocess
    df_prep = preprocess_data(df_raw.copy())

    # 2) Feature engineering (opsiyonel adım)
    df_fe = engineer_features(df_prep.copy())

    # 3) Korelasyon (preprocess) – Hedef = “Work_Interest”
    print("🔍 Preprocess sonrası korelasyon matrisi (Work_Interest dahil):")
    plot_correlation_matrix(df_prep, title="Correlation Matrix (Preprocess Only, Work_Interest)")

    # 4) Korelasyon (feature engineering) – Hedef = “Work_Interest”
    print("🔍 Feature engineering sonrası korelasyon matrisi (Work_Interest dahil):")
    plot_correlation_matrix(df_fe, title="Correlation Matrix (After Feature Engineering, Work_Interest)")

    # 5) Work_Interest ile en yüksek korelasyona sahip sütunları listele
    print("\n→ Work_Interest ile korelasyonlar (Preprocess Only):")
    corr_prep = df_prep.corr()['Work_Interest'].sort_values(ascending=False)
    print(corr_prep)

    print("\n→ Work_Interest ile korelasyonlar (After Feature Engineering):")
    corr_fe = df_fe.corr()['Work_Interest'].sort_values(ascending=False)
    print(corr_fe)

if __name__ == "__main__":
    main()
