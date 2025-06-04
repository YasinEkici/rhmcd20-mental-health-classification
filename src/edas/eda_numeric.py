# eda_numeric_half_and_half.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec

from preprocessing import preprocess_data

def plot_numeric_half_and_half(df: pd.DataFrame, numeric_cols: list):
    """
    df: preprocess_data sonrası tam sayısal DataFrame (Coping_Struggles dahil)
    numeric_cols: Analiz etmek istediğimiz sayısal sütun isimleri (string listesi)

    - Sol yarıda: tüm sayısal sütunların boxplot’ları (Coping_Struggles = 0 vs 1)
    - Sağ yarıda: tüm sayısal sütunların violinplot’ları (Coping_Struggles = 0 vs 1)
    """
    n = len(numeric_cols)
    if n == 0:
        print("Hiç sayısal sütun bulunamadı.")
        return

    # 1 figür, GridSpec ile (n satır × 2 sütun) ama burada soldaki sütun boxplot'lar,
    # sağdaki sütun violinplot'lar şeklinde düzenleniyor.
    fig = plt.figure(figsize=(12, 5 * n))
    gs = GridSpec(n, 2, figure=fig, width_ratios=[1, 1], wspace=0.3, hspace=0.4)

    for i, col in enumerate(numeric_cols):
        # --------- Sol yarı (boxplot) ---------
        ax_box = fig.add_subplot(gs[i, 0])
        sns.boxplot(
            x='Coping_Struggles',
            y=col,
            hue='Coping_Struggles',
            data=df,
            palette='Set2',
            legend=False,
            ax=ax_box
        )
        ax_box.set_title(f"{col} Boxplot")
        ax_box.set_xlabel("Coping_Struggles")
        ax_box.set_ylabel(col)

        # --------- Sağ yarı (violinplot) ---------
        ax_vl = fig.add_subplot(gs[i, 1])
        sns.violinplot(
            x='Coping_Struggles',
            y=col,
            hue='Coping_Struggles',
            data=df,
            inner='quartile',
            palette='Set3',
            legend=False,
            ax=ax_vl
        )
        ax_vl.set_title(f"{col} Violinplot")
        ax_vl.set_xlabel("Coping_Struggles")
        ax_vl.set_ylabel(col)

    plt.tight_layout()
    plt.show()


def main():
    # 1) Ham CSV’yi yükle + preprocess_data
    raw_path = Path(__file__).parents[1] / 'data' / 'raw' / 'mental_health_finaldata_1.csv'
    df_raw = pd.read_csv(raw_path)
    df = preprocess_data(df_raw.copy())

    # 2) Hangi sütunların sayısal olduğunu listeleyelim
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

    print("🔍 Sayısal sütunları yarı ekran şeklinde (solda boxplot, sağda violinplot) görselleştiriyoruz...\n")
    plot_numeric_half_and_half(df, numeric_cols)


if __name__ == "__main__":
    main()
