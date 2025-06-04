# eda_numeric_work_interest.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from legacy.preprocessing2 import preprocess_data

def plot_numeric_distributions_side_by_side(df: pd.DataFrame, numeric_cols: list, target_col: str):
    """
    df: preprocess_data sonrası tam sayısal DataFrame
    numeric_cols: Görselleştirilecek sayısal sütun isimleri listesi
    target_col: "Work_Interest" veya başka chosen target sütunu
    """
    for col in numeric_cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Boxplot
        sns.boxplot(
            x=target_col,
            y=col,
            hue=target_col,
            data=df,
            palette='Set2',
            legend=False,
            ax=axes[0]
        )
        axes[0].set_title(f"{col} Boxplot by {target_col}")
        axes[0].set_xlabel(target_col)
        axes[0].set_ylabel(col)
        
        # Violinplot
        sns.violinplot(
            x=target_col,
            y=col,
            hue=target_col,
            data=df,
            inner='quartile',
            palette='Set3',
            legend=False,
            ax=axes[1]
        )
        axes[1].set_title(f"{col} Violin Plot by {target_col}")
        axes[1].set_xlabel(target_col)
        axes[1].set_ylabel(col)
        
        plt.tight_layout()
        plt.show()

def main():
    # 1) Ham CSV'yi yükle & preprocess_data ile tüm sayısal dönüşümleri yap
    raw_path = Path(__file__).parents[1] / 'data' / 'raw' / 'mental_health_finaldata_1.csv'
    df_raw = pd.read_csv(raw_path)

    df = preprocess_data(df_raw.copy())  
    # preprocess_data() içinde artık Work_Interest sütunu binary olarak eklendi.

    # 2) Sayısal sütunları listele
    numeric_cols = [
        "Age",
        "Days_Indoors",
        "Growing_Stress",
        "Quarantine_Frustrations",
        "Changes_Habits",
        "Mental_Health_History",
        "Weight_Change",
        "Mood_Swings"
    ]

    target_col = "Work_Interest"   # Yeni hedef

    print(f"🔍 Sayısal sütunları hedef=“{target_col}” için yan yana görselleştiriyoruz...\n")
    plot_numeric_distributions_side_by_side(df, numeric_cols, target_col)

if __name__ == "__main__":
    main()
