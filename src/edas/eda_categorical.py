# eda_categorical_single_screen.py

import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from preprocessing import preprocess_data

def plot_categorical_distributions_single_screen(df: pd.DataFrame, cat_cols: list):
    """
    df: preprocess_data sonrası sayısal hale gelmiş DataFrame (Coping_Struggles dahil)
    cat_cols: Dummy kodlanmış (one-hot) nominal sütun listesi (string listesi)
    """
    n = len(cat_cols)
    if n == 0:
        print("Hiç kategorik sütun bulunamadı.")
        return

    # Alt grafik düzenini belirleyelim: kareye yakın bir düzen olmasına çalışalım
    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()  # Düz bir liste haline getirelim

    for idx, col in enumerate(cat_cols):
        # 0/1 sütunu için Coping_Struggles ayrımı
        freq_table = df.groupby('Coping_Struggles')[col].mean().reset_index()
        print(f"\n→ {col} için Coping_Struggles grup dağılımı (ortalama):")
        print(freq_table)
        print()

        ax = axes[idx]
        sns.barplot(x='Coping_Struggles', y=col, data=df, palette='Set1', ax=ax)
        ax.set_title(f"{col} Mean by Coping_Struggles")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Coping_Struggles (0 vs 1)")
        ax.set_ylabel(f"Mean of {col}")

    # Kullanılmayan subplot'ları gizleyelim
    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    # 1) Ham CSV’yi yükle + preprocess
    raw_path = Path(__file__).parents[1] / 'data' / 'raw' / 'mental_health_finaldata_1.csv'
    df_raw = pd.read_csv(raw_path)
    df = preprocess_data(df_raw.copy())

    # 2) Dummy kodlanmış sütunları belirleyelim
    cat_cols = [c for c in df.columns
                if c.startswith("Gender_")
                or c.startswith("Occupation_")
                or c.startswith("Work_Interest_")
                or c.startswith("Social_Weakness_")]

    print("🔍 Kategorik (dummy) sütunlar:", cat_cols)
    print("🔍 Bu sütunların Coping_Struggles’a göre dağılımını gösteriyoruz...\n")
    plot_categorical_distributions_single_screen(df, cat_cols)


if __name__ == "__main__":
    main()
