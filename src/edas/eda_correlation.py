# eda_correlation.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from preprocessing import preprocess_data
from feature_engineering import engineer_features

def plot_correlation_matrix(df: pd.DataFrame, title: str = "Correlation Matrix"):
    """
    df: Tam sayısal DataFrame (ölümler: preprocess_data + engineer_features)
    title: Plot başlığı
    """
    corr = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", 
                cbar_kws={'shrink': .75}, linewidths=0.5)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    # 1) Ham veri yükle + preprocess
    raw_path = Path(__file__).parents[1] / 'data' / 'raw' / 'mental_health_finaldata_1.csv'
    df_raw = pd.read_csv(raw_path)
    df_prep = preprocess_data(df_raw.copy())

    # 2) Feature engineering (opsiyonel: bu aşamaya dahil etmek istiyorsanız)
    df_fe = engineer_features(df_prep.copy())

    # 3) Korelasyon matrisi çiz (sadece preprocess_data sonrası ise df_prep, 
    #    yok eğer feature engineering de görmek isterseniz df_fe kullanın)
    print("🔍 Preprocess sonrası korelasyon matrisi (sadece ön işleme adımları):")
    plot_correlation_matrix(df_prep, title="Correlation Matrix (Preprocess Only)")

    print("🔍 Feature engineering sonrası korelasyon matrisi (yeni feature’lar dahil):")
    plot_correlation_matrix(df_fe, title="Correlation Matrix (After Feature Engineering)")

    # 4) Özellikle "Coping_Struggles" sütununun korelasyonlarını listeleyelim:
    print("\n→ Coping_Struggles ile korelasyonlar (Preprocess Only):")
    corr_prep = df_prep.corr()['Coping_Struggles'].sort_values(ascending=False)
    print(corr_prep)

    print("\n→ Coping_Struggles ile korelasyonlar (After Feature Engineering):")
    corr_fe = df_fe.corr()['Coping_Struggles'].sort_values(ascending=False)
    print(corr_fe)

if __name__ == "__main__":
    main()
