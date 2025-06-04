# preprocessing.py

import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer

# ---------- Dosya Yol Tanımları (eğer doğrudan kullanmak isterseniz) ----------
# Ham veri: data/raw/mental_health_finaldata_1.csv
RAW_PATH = Path(__file__).parents[1] / 'data' / 'raw' / 'mental_health_finaldata_1.csv'


def load_data(path: Path) -> pd.DataFrame:
    """
    Verilen path'ten CSV yükleyip DataFrame olarak döndürür.
    """
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ham DataFrame üzerinde yapılacak önişleme adımları:
      1) Eksik değer raporu (ekrana basılıyor, silme/yada doldurma burada)
      2) Manuel ordinal eşlemeler (Age, Days_Indoors, Mood_Swings vb.)
      3) Eksik sayısal değerler için SimpleImputer (medyan stratejisi)
      4) One-hot encode / dummy değişkenler (Gender, Occupation, Work_Interest, Social_Weakness)
      5) Target (Coping_Struggles) sütununu numeric olarak kodlama
         - Burada yalnızca 0/1 veya 0/1/2... gibi kategorik etiketleri sayısala çeviriyoruz.
         - Ancak bu kodlama **hemen ardından** X ve y ayrımında kullanılacak.
      6) Sonuç olarak “tamamen sayısal” bir DataFrame dönüyoruz.
    """
    # 1) Eksik değerleri ekrana yazdır
    missing = df.isnull().sum()
    print("🔍 Missing values per column:\n", missing, "\n")

    # 2) Manuel ordinal mapping
    age_map = {
        "16-20": 0,
        "20-25": 1,
        "25-30": 2,
        "30-Above": 3
    }
    days_map = {
        "Go out Every day":   0,
        "1-14 days":          1,
        "15-30 days":         2,
        "31-60 days":         3,
        "More than 2 months": 4
    }
    mood_map = {
        "Low":    0,
        "Medium": 1,
        "High":   2
    }
    yesno_map = {
        "No":    0,
        "Maybe": 1,
        "Yes":   2
    }
    
    # Sadece “varsa” hatalı veri olursa, .map sonrası NaN olacak; bunları imputer ile dolduracağız.
    df["Age"] = df["Age"].map(age_map)
    df["Days_Indoors"] = df["Days_Indoors"].map(days_map)
    df["Mood_Swings"] = df["Mood_Swings"].map(mood_map)

    for col in [
        "Growing_Stress",
        "Quarantine_Frustrations",
        "Changes_Habits",
        "Mental_Health_History",
        "Weight_Change"
    ]:
        df[col] = df[col].map(yesno_map)
    df["Work_Interest"] = df["Work_Interest"].map(yesno_map).apply(lambda x: 1 if x==2 else 0)
    # 3) Eksik sayısal (numeric) sütunlar için medyan ile doldurma
    num_cols = [
        "Age",
        "Days_Indoors",
        "Growing_Stress",
        "Quarantine_Frustrations",
        "Changes_Habits",
        "Mental_Health_History",
        "Weight_Change",
        "Mood_Swings"
    ]
    imputer = SimpleImputer(strategy='median')
    df[num_cols] = imputer.fit_transform(df[num_cols])

    # 4) Nominal (kategorik) sütunlara one-hot encode
    #    (örn. Gender, Occupation, Work_Interest, Social_Weakness)
    #    drop_first=True demek, bir sütunu diğerleri referans olarak kullanmak demektir.
    nom_cols = ["Gender", "Occupation", "Social_Weakness"]
    df = pd.get_dummies(df, columns=nom_cols, drop_first=True)

    # 5) Target sütununu (Coping_Struggles) numeric hale çevirme
    #    Burada sadece “kategori → kod” (0/1 vs.) işlemi yaptık; leakage yok.
    #    Örneğin: "No" → 0, "Yes" → 1 gibi bir dönüşüm varsa zaten string ise:
    df["Coping_Struggles"] = df["Coping_Struggles"].astype('category').cat.codes

    # 6) Sonuç: Tamamen sayısal bir DataFrame döndür
    return df


# Eğer bu dosyayı tek başına çalıştırırsanız, preprocess_data'yı test etmek için:
if __name__ == "__main__":
    # Ham veriyi oku
    df_raw = load_data(RAW_PATH)

    # Önişleme yap
    df_clean = preprocess_data(df_raw)

    # Ekrana örnek satır göster
    print("\n📋 Önişlenmiş DataFrame örneği:\n", df_clean.head(), "\n")
    print("ℹ️ preprocess_data fonksiyonu başarıyla çalıştı.")
