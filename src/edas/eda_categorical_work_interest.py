# eda_categorical_work_interest.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from legacy.preprocessing2 import preprocess_data

def plot_categorical_distributions(df: pd.DataFrame, cat_cols: list, target_col: str):
    """
    df: preprocess_data sonrası sayısal hale gelmiş DataFrame (hedef dahil)
    cat_cols: Dummy kodlanmış (one-hot) kategorik sütunlar listesi
    target_col: Örneğin "Work_Interest"
    """
    for col in cat_cols:
        freq_table = df.groupby(target_col)[col].mean().reset_index()
        print(f"\n→ {col} için {target_col} grup ortalamaları:")
        print(freq_table)
        print()
        plt.figure(figsize=(5, 3))
        sns.barplot(x=target_col, y=col, data=df, palette='Set1')
        plt.title(f"{col} Mean by {target_col}")
        plt.ylim(0, 1)
        plt.xlabel(target_col)
        plt.ylabel(f"Mean of {col}")
        plt.tight_layout()
        plt.show()

def main():
    raw_path = Path(__file__).parents[1] / 'data' / 'raw' / 'mental_health_finaldata_1.csv'
    df_raw = pd.read_csv(raw_path)
    df = preprocess_data(df_raw.copy())

    # “Work_Interest” binary hale geldi, bu yüzden artık dummy kodlama yapılmıyor.
    # Kategorik özellikler: Gender, Occupation, Social_Weakness
    # Bunları preprocess_data() içinde one-hot ettik (drop_first=True kullanıldı).
    cat_cols = [c for c in df.columns
                if c.startswith("Gender_")
                or c.startswith("Occupation_")
                or c.startswith("Social_Weakness_")]

    target_col = "Work_Interest"
    print(f"🔍 Kategorik (dummy) sütunları hedef=“{target_col}” için görselleştiriyoruz...\n")
    plot_categorical_distributions(df, cat_cols, target_col)

if __name__ == "__main__":
    main()
