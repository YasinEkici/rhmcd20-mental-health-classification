# preprocessing.py

import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer

# ---------- Dosya Yol Tanımları (eğer doğrudan kullanmak isterseniz) ----------
# RAW_PATH: Proje kök dizininden bir üst klasöre çıkıp, data/raw/mental_health_finaldata_1.csv dosyasını işaret eder.
# Böylece fonksiyonları dışarıdan doğrudan çağırırken bu sabit yol üzerinden ham veriyi okuyabiliriz.
RAW_PATH = Path(__file__).parents[1] / 'data' / 'raw' / 'mental_health_finaldata_1.csv'


def load_data(path: Path) -> pd.DataFrame:
    """
    Verilen path'ten CSV dosyasını pandas DataFrame olarak yükleyip döndürür.

    Parametreler:
    -----------
    path : Path
        Yüklenecek CSV dosyasının dosya sistemi yolu (örneğin: RAW_PATH).

    Dönüş:
    ------
    pd.DataFrame
        CSV içeriğini barındıran pandas DataFrame. Başlık satırları ve veri tipleri
        otomatik olarak algılanır. Eksik (NaN) değerler pandas tarafından NaN olarak temsil edilir.
    """
    # pd.read_csv ile CSV dosyasını okutuyoruz. Eğer dosya bulunamazsa hata fırlatır.
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ham DataFrame üzerinde ayrıntılı önişleme adımları gerçekleştirir ve tamamen sayısal bir DataFrame döner.

    Önişleme Adımları:
    1) Eksik değer raporu: Her bir sütundaki eksik (NaN) değer sayısını hesaplayıp ekrana yazdırır.
    2) Manuel ordinal eşlemeler: Metinsel kategorik verileri (Age, Days_Indoors, Mood_Swings vb.)
       önceden tanımlı sıralı (ordinal) sayısal değerlere dönüştürür. Eğer hatalı veya beklenmeyen
       bir kategori gelmişse map sonrası NaN dönecektir; bunlar sonraki adımda doldurulacaktır.
    3) Eksik sayısal değerler için SimpleImputer (medyan stratejisi) uygulanır.
       Yani sütun içinde kalan NaN değerler o sütunun medyanıyla doldurulur.
    4) Nominal/kategorik sütunlar (Gender, Occupation, Work_Interest, Social_Weakness) için
       one-hot encode (dummy değişkenler) uygulanır. Böylece her kategori için yeni bir sütun oluşturulur.
       drop_first=True ile bir kategori referans alınır ve ona karşılık gelen sütun düşürülür,
       böylece multikolineeritenin önüne geçilmiş olur.
    5) Target sütunu (Coping_Struggles) eğer string veya kategori türünde ise kategori kodlarına
       dönüştürülür (örneğin "No" → 0, "Maybe" → 1, "Yes" → 2). Eğer zaten sayısalsa dokunulmaz.
    6) Sonuç olarak tamamen sayısal (integer veya float) sütunlardan oluşan bir DataFrame döndürülür.
       Bu, daha sonra makine öğrenmesi modellerinde (sklearn vb.) kullanılmak üzere hazır halidir.

    Parametreler:
    -----------
    df : pd.DataFrame
        Ham pandas DataFrame. Yüklenen CSV'den direkt gelen ve henüz önişlenmemiş verileri içerir.

    Dönüş:
    ------
    pd.DataFrame
        Tamamen sayısal sütunlardan oluşan, eksik değerleri doldurulmuş ve kategorik
        sütunları sayısal kodlamaya çevrilmiş DataFrame. İçerisinde hedef değişken (Coping_Struggles)
        de sayısal bir kodlamaya dönüştürülmüş şekliyle yer alır.
    """
    # ------------------------------------------------------------------------------------------------
    # 1) Eksik değerler raporu
    # ------------------------------------------------------------------------------------------------
    # DataFrame.isnull().sum() ile her bir sütundaki NaN sayısını alıyoruz.
    missing = df.isnull().sum()
    # Eksik değer raporunu ekrana çok detaylı bir şekilde yazdırıyoruz.
    print("🔍 Sütun bazında eksik değer sayıları:\n", missing, "\n")

    # ------------------------------------------------------------------------------------------------
    # 2) Manuel ordinal mapping (sıralı kategorik verilerin sayısallaştırılması)
    # ------------------------------------------------------------------------------------------------
    # age_map: Age sütununda beklenen kategorik değerler ve bunların sıralı sayısal karşılıkları.
    age_map = {
        "16-20": 0,
        "20-25": 1,
        "25-30": 2,
        "30-Above": 3
    }
    # days_map: Days_Indoors sütununda beklenen kategorik değerler ve sıralı sayısal karşılıkları.
    days_map = {
        "Go out Every day":   0,
        "1-14 days":          1,
        "15-30 days":         2,
        "31-60 days":         3,
        "More than 2 months": 4
    }
    # mood_map: Mood_Swings sütununda beklenen kategorik değerler ve sıralı sayısal karşılıkları.
    mood_map = {
        "Low":    0,
        "Medium": 1,
        "High":   2
    }
    # yesno_map: Bir dizi sütun ("Yes"/"No"/"Maybe") yanıtlarını sayısal değerlere dönüştürmek için.
    yesno_map = {
        "No":    0,
        "Maybe": 1,
        "Yes":   2
    }

    # Age sütununu age_map sözlüğüne göre eşliyoruz. Eğerbeklenmeyen bir string gelirse NaN olur.
    df["Age"] = df["Age"].map(age_map)
    # Days_Indoors sütununu days_map sözlüğüne göre eşliyoruz.
    df["Days_Indoors"] = df["Days_Indoors"].map(days_map)
    # Mood_Swings sütununu mood_map sözlüğüne göre eşliyoruz.
    df["Mood_Swings"] = df["Mood_Swings"].map(mood_map)

    # Birden fazla sütun için aynı eşleme (yesno_map) uygulanıyor.
    # Bu sütunlar string "Yes"/"No"/"Maybe" şeklinde geldiği varsayılıyor.
    for col in [
        "Growing_Stress",
        "Quarantine_Frustrations",
        "Changes_Habits",
        "Mental_Health_History",
        "Weight_Change"
    ]:
        # Her sütunu eşleme ile sayısal değerlere çeviriyoruz.
        df[col] = df[col].map(yesno_map)

    # ------------------------------------------------------------------------------------------------
    # 3) Eksik sayısal değerler için SimpleImputer (medyan stratejisi)
    # ------------------------------------------------------------------------------------------------
    # num_cols listesi: sayısal olması gereken ve yukarıdaki map işlemi sonrası NaN kalmış sütunları içerir.
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
    # SimpleImputer: eksik değerleri belirttiğimiz stratejiye göre doldurur.
    # strategy='median' demek, her sütunun kendi medyan değerini hesaplar ve NaN'ları bu medyanla doldurur.
    imputer = SimpleImputer(strategy='median')
    # fit_transform: önce ilgili sütunların medyanını hesaplar, ardından her NaN'ı o medyanla doldurur.
    df[num_cols] = imputer.fit_transform(df[num_cols])

    # ------------------------------------------------------------------------------------------------
    # 4) Nominal (kategorik) sütunlara one-hot encode (dummy değişkenler)
    # ------------------------------------------------------------------------------------------------
    # get_dummies ile kategorik sütunları dummies sütunlarına çeviriyoruz.
    # drop_first=True, bir kategori referans alınacağı için bir sütun düşürülür.
    # Bu, çoklu doğrusal bağlantı (multicollinearity) sorununu azaltmaya yardımcı olur.
    nom_cols = ["Gender", "Occupation", "Work_Interest", "Social_Weakness"]
    # orijinal DataFrame üzerine one-hot encoding uygulanarak yeni sütunlar eklenir.
    df = pd.get_dummies(df, columns=nom_cols, drop_first=True)

    # ------------------------------------------------------------------------------------------------
    # 5) Target sütunu (Coping_Struggles) numeric hale çevirme
    # ------------------------------------------------------------------------------------------------
    # Eğer hedef sütunu hâlen object veya category türündeyse:
    if df["Coping_Struggles"].dtype.name in ['object', 'category']:
        # pandas'ın category kodlama yöntemini kullanıyoruz.
        # Örneğin: Eğer "Coping_Struggles" sütununda "No" ve "Yes" gibi değerler varsa,
        # bunlar otomatik olarak 0 ve 1 şeklinde kodlanır. (Alfabetik sıra bazlı veya label sırasına göre)
        df["Coping_Struggles"] = df["Coping_Struggles"].astype('category').cat.codes
    else:
        # Eğer sütun zaten sayısal tipindeyse (int64 veya float64 gibi), bu adımda hiçbirşey yapılmaz.
        pass

    # ------------------------------------------------------------------------------------------------
    # 6) Sonuç: tamamen sayısal bir DataFrame döndürme
    # ------------------------------------------------------------------------------------------------
    # Artık df içindeki tüm sütunlar sayısal tipte (int veya float) ve eksik değerlerden arındırılmış durumda.
    return df


# ----------------------------------------------
# Eğer bu dosyayı doğrudan bir komut satırı aracı
# olarak çalıştırırsanız (python preprocessing.py),
# aşağıdaki bölüm devreye girer ve preprocess_data'nın
# doğru çalışıp çalışmadığını hızlıca test edebilirsiniz.
# ----------------------------------------------
if __name__ == "__main__":
    # 1) Ham veriyi oku: RAW_PATH sabitinde tanımlı olan dosya yolundan CSV'yi yükle.
    df_raw = load_data(RAW_PATH)

    # 2) Önişleme fonksiyonunu çalıştır ve çıktısını df_clean değişkenine ata.
    df_clean = preprocess_data(df_raw)

    # 3) Önişlenmiş veriden bir örnek satırı ekrana yazdır.
    print("\n📋 Önişlenmiş DataFrame'den ilk örnek satırlar:\n", df_clean.head(), "\n")
    # 4) Başarı mesajı bas: preprocess_data fonksiyonunun sorunsuz çalıştığını gösterir.
    print("ℹ️ preprocess_data fonksiyonu başarıyla çalıştı.")
