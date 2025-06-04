# feature_engineering.py

import pandas as pd
import numpy as np

def add_high_risk_stress_frustration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mevcut alta düzey risk bayraklarını ekler:
      - High_Stress_Mood_Swings: Growing_Stress >= 2 ve Mood_Swings >= 2 ise yüksek risk
      - Days_In_Frustration   : Days_Indoors >= 2 ve Quarantine_Frustrations >= 2 ise yüksek risk

    Parametreler:
    -----------
    df : pd.DataFrame
        Önişlenmiş ve sayısal değerlere dönüştürülmüş bir DataFrame. Aşağıdaki sütunları barındırdığı varsayılır:
          - 'Growing_Stress'           : Bireyin yaşadığı stres seviyesini sayısal olarak (örneğin 0,1,2) temsil eder.
          - 'Mood_Swings'              : Bireyin duygu dalgalanmalarını sayısal kodlamayla (örneğin 0,1,2) ifade eder.
          - 'Days_Indoors'             : Bireyin kapalı alanda (evde) kalma süresini sayısal kodlamayla (örneğin 0,1,2,3,4) gösterir.
          - 'Quarantine_Frustrations'  : Bireyin karantina sırasında yaşadığı hayal kırıklığı/rahatsızlığı sayısal olarak (örneğin 0,1,2) belirtir.

    Dönüş:
    ------
    pd.DataFrame
        Orijinal DataFrame kopyası üzerinde iki yeni bayrak sütunu eklenmiş olarak döner:
          - 'High_Stress_Mood_Swings'  : Bireyin hem stres seviyesi hem de duygu dalgalanmalarının eşik değerinin üstünde olduğunu gösteren 0/1 bayrak sütunu.
          - 'Days_In_Frustration'      : Bireyin hem Days_Indoors hem de Quarantine_Frustrations eşiklerini aştığında 1, aksi halde 0 değeri alan bayrak sütunu.
    """
    # Kopya oluşturuyoruz, böylece orijinal df değişmez
    df = df.copy()

    # High_Stress_Mood_Swings: Growing_Stress >= 2 ve Mood_Swings >= 2 ise 1 (yüksek risk), değilse 0
    df['High_Stress_Mood_Swings'] = (
        (df['Growing_Stress'] >= 2) &   # Stres seviyesi en az 2 mi?
        (df['Mood_Swings']   >= 2)      # Duygu dalgalanması en az 2 mi?
    ).astype(int)                       # Boolean sonucu 0/1 değerine çevir

    # Days_In_Frustration: Days_Indoors >= 2 ve Quarantine_Frustrations >= 2 ise 1 (yüksek risk), değilse 0
    df['Days_In_Frustration'] = (
        (df['Days_Indoors'] >= 2) &                     # Kapalı alanda kalma süresi en az 2 mi?
        (df['Quarantine_Frustrations'] >= 2)            # Karantina rahatsızlığı en az 2 mi?
    ).astype(int)                                        # Boolean sonucu 0/1 değerine çevir

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gelişmiş Feature Engineering (Özellik Mühendisliği) adımları:
      1) Orijinal yüksek risk bayrakları (add_high_risk_stress_frustration)
      2) Stres ile duygu dalgalanmasının etkileşimi (Stress × Mood)
      3) Stres ile hayal kırıklığının etkileşimi (Stress × Frustration)
      4) Toplam stres + hayal kırıklığı (Total_Stress_Frustration)
      5) Fraksiyonel hayal kırıklığı (Frac_Frustration)
      6) Quadratic (kare) özellikler (Mood^2, Stress^2, Frustration^2)
      7) Medyan bazlı binary bayraklar (High_Total_Stress, High_Frustration_Rate)
      8) Quantile‐bazlı üçlü seviye bucket’lar (Low/Medium/High)
         → Eğer quantile eşitse (aynı değer çıkarsa), tüm satırlara 0 atar
      9) Yaş × Kilo_Değişimi etkileşimi (Age × Weight_Change)
     10) Log dönüşümü (Days_Indoors)
     11) Bileşkesel skor: Mental_Health_History + Growing_Stress + Quarantine_Frustrations
     12) Alışkanlık değişimi ile duygu dalgalanması etkileşimi (Changes_Habits × Mood_Swings)
     13) İş İlgisi × Sosyal Zayıflık toplamı (Work_Interest × Social_Weakness)
     14) Meslek dummy sütunlarından oluşan Meslek Sayısı (Occupation_Count)
     15) Kilo Değişimi Bayrağı (Weight_Yes_Flag)
     16) Mood Seviyesi (Mood_Swings tercile seviyeleri; quantile eşit ise 0)
     17) Stres Seviyesi (Growing_Stress tercile seviyeleri; quantile eşit ise 0)
     18) Uzun Süre Kapalı Kalma bayrağı (Long_Time_Indoors: Days_Indoors >= 3)

    Parametreler:
    -----------
    df : pd.DataFrame
        Önişlenmiş ve sayısal sütunları içeren bir DataFrame. Aşağıdaki sütunların zaten var olduğu varsayılır:
          - 'Age'                     : Yaş aralığı ordinal (0-3 arası integer)
          - 'Growing_Stress'          : Stres seviyesi (0-2 arası integer)
          - 'Quarantine_Frustrations' : Karantina hayal kırıklığı seviyesi (0-2 arası integer)
          - 'Mood_Swings'             : Duygu dalgalanma seviyesi (0-2 arası integer)
          - 'Days_Indoors'            : Kapalı alanda kalma süresi ordinal (0-4 arası integer)
          - 'Mental_Health_History'   : Ruh sağlığı geçmişi (0-2 arası integer)
          - 'Changes_Habits'          : Alışkanlık değişimi (0-2 arası integer)
          - 'Weight_Change'           : Kilo değişimi (0-2 arası integer)
          - Work_Interest_*           : One-hot encode edilmiş İş İlgisi dummy sütunları (örneğin 'Work_Interest_Maybe', 'Work_Interest_Yes')
          - Social_Weakness_*         : One-hot encode edilmiş Sosyal Zayıflık dummy sütunları (örneğin 'Social_Weakness_Maybe', 'Social_Weakness_Yes')
          - Occupation_*              : One-hot encode edilmiş Meslek dummy sütunları (örneğin 'Occupation_Student', 'Occupation_Others', vb.)

    Dönüş:
    ------
    pd.DataFrame
        Orijinal DataFrame kopyası üzerinde yukarıda tanımlanan tüm ek özellikler (feature’lar) eklenmiş olarak döner.
        Her bir adımın açıklaması aşağıda detaylıca yer alır.
    """
    # Orijinal DataFrame'i değiştirmemek için bir kopyasını oluşturuyoruz.
    df = df.copy()

    # ------------------------------------------------------------------------------------------------
    # 1) Orijinal yüksek risk bayrakları
    #    (Growing_Stress ve Mood_Swings, Days_Indoors ve Quarantine_Frustrations üzerinden oluşturulan bayraklar)
    # ------------------------------------------------------------------------------------------------
    df = add_high_risk_stress_frustration(df)

    # ------------------------------------------------------------------------------------------------
    # 2) Etkileşim terimleri (Interaction Features)
    #    Stres ve duygu dalgalanmasının çarpımı → 'Stress_Mood_Interaction'
    #    Stres ve hayal kırıklığının çarpımı → 'Stress_Frustration_Interaction'
    # ------------------------------------------------------------------------------------------------
    df['Stress_Mood_Interaction']        = df['Growing_Stress'] * df['Mood_Swings']
    df['Stress_Frustration_Interaction'] = df['Growing_Stress'] * df['Quarantine_Frustrations']

    # ------------------------------------------------------------------------------------------------
    # 3) Toplam stres + hayal kırıklığı
    #    Her bireyin yaşadığı stres (Growing_Stress) ile hayal kırıklığını (Quarantine_Frustrations) topluyor.
    #    Yeni sütun: 'Total_Stress_Frustration'
    # ------------------------------------------------------------------------------------------------
    df['Total_Stress_Frustration'] = df['Growing_Stress'] + df['Quarantine_Frustrations']

    # ------------------------------------------------------------------------------------------------
    # 4) Fraksiyonel hayal kırıklığı
    #    'Quarantine_Frustrations' değerini gün sayısına (Days_Indoors + 1) bölerek
    #    göreceli (oran) hayal kırıklığı seviyesi hesaplanır.
    #    +1 ekleyerek bölme sıfıra bölme hatasını önlüyoruz.
    #    Yeni sütun: 'Frac_Frustration'
    # ------------------------------------------------------------------------------------------------
    df['Frac_Frustration'] = df['Quarantine_Frustrations'] / (df['Days_Indoors'] + 1)

    # ------------------------------------------------------------------------------------------------
    # 5) Quadratic (kare) özellikler:
    #    Mood_Swings^2, Growing_Stress^2, Quarantine_Frustrations^2
    #    Modelin doğrusal olmayan (nonlinear) ilişkileri yakalamasını kolaylaştırır.
    # ------------------------------------------------------------------------------------------------
    df['Mood_Squared']        = df['Mood_Swings'] ** 2
    df['Stress_Squared']      = df['Growing_Stress'] ** 2
    df['Frustration_Squared'] = df['Quarantine_Frustrations'] ** 2

    # ------------------------------------------------------------------------------------------------
    # 6) Medyan bazlı binary bayraklar:
    #    Toplam stres ve fraksiyonel hayal kırıklığı değerlerinin medyanını hesaplayıp,
    #    bu medyan değerlerin üstünde olan satırlara 1 (yüksek), altında veya eşit olanlara 0 atar.
    #    Yeni sütunlar: 'High_Total_Stress', 'High_Frustration_Rate'
    # ------------------------------------------------------------------------------------------------
    # 6.1) Toplam stres + hayal kırıklığı medyanı
    med_total = df['Total_Stress_Frustration'].median()
    df['High_Total_Stress']     = (df['Total_Stress_Frustration'] > med_total).astype(int)

    # 6.2) Fraksiyonel hayal kırıklık medyanı
    med_frac = df['Frac_Frustration'].median()
    df['High_Frustration_Rate'] = (df['Frac_Frustration'] > med_frac).astype(int)

    # ------------------------------------------------------------------------------------------------
    # 7) Quantile‐bazlı üçlü seviye bucket’lar (Low/Medium/High)
    #    Total_Stress_Frustration ve Frac_Frustration sütunları için
    #    33. ve 66. yüzdelik değerleri (tercile) hesaplanır.
    #    Eğer bu iki değer birbirinden farklı ise (tercile belirgin ise):
    #      - Değerler (-inf, 33. yüzdelik] → 0 (Low)
    #                 (33., 66. yüzdelik]  → 1 (Medium)
    #                 (66., +inf)         → 2 (High)
    #    Aksi takdirde (tercile eşit çıkarsa) tüm satırlara 0 atanır.
    #    Yeni sütunlar: 'Total_Stress_Level', 'Frac_Frustration_Level'
    # ------------------------------------------------------------------------------------------------
    # 7.1) Total_Stress_Frustration için tercile değerleri
    terciles = df['Total_Stress_Frustration'].quantile([0.33, 0.66]).values
    if terciles[0] < terciles[1]:
        df['Total_Stress_Level'] = pd.cut(
            df['Total_Stress_Frustration'],
            bins=[-np.inf, terciles[0], terciles[1], np.inf],
            labels=[0, 1, 2]
        ).astype(int)
    else:
        # Eğer 33. ve 66. yüzdelik aynıysa, tüm satırlara 0 ata
        df['Total_Stress_Level'] = 0

    # 7.2) Frac_Frustration için tercile değerleri
    terciles_frac = df['Frac_Frustration'].quantile([0.33, 0.66]).values
    if terciles_frac[0] < terciles_frac[1]:
        df['Frac_Frustration_Level'] = pd.cut(
            df['Frac_Frustration'],
            bins=[-np.inf, terciles_frac[0], terciles_frac[1], np.inf],
            labels=[0, 1, 2]
        ).astype(int)
    else:
        # Eğer 33. ve 66. yüzdelik aynıysa, tüm satırlara 0 ata
        df['Frac_Frustration_Level'] = 0

    # ------------------------------------------------------------------------------------------------
    # 8) Age × Weight_Change etkileşimi
    #    Bireyin yaşı (Age) ile kilo değişimi (Weight_Change) sütunlarının çarpımı.
    #    Yeni sütun: 'Age_Weight_Interaction'
    # ------------------------------------------------------------------------------------------------
    df['Age_Weight_Interaction'] = df['Age'] * df['Weight_Change']

    # ------------------------------------------------------------------------------------------------
    # 9) Log dönüşümü: Days_Indoors
    #    Kapalı kalma süresi sayısının logaritmasını alarak dağılımı normalize etmeye çalışır.
    #    np.log1p(x) fonksiyonu, x=0 için log(1)=0 sonucunu verecek şekilde çalışır.
    #    Yeni sütun: 'Log_Days_Indoors'
    # ------------------------------------------------------------------------------------------------
    df['Log_Days_Indoors'] = np.log1p(df['Days_Indoors'])

    # ------------------------------------------------------------------------------------------------
    # 10) Bileşkesel skor: Mental_Stress_Frustration_Score
    #     Ruh sağlığı geçmişi (Mental_Health_History), stres (Growing_Stress) ve
    #     hayal kırıklığı (Quarantine_Frustrations) değerlerini toplayarak tek bir skor oluşturur.
    #     Yeni sütun: 'Mental_Stress_Frustration_Score'
    # ------------------------------------------------------------------------------------------------
    df['Mental_Stress_Frustration_Score'] = (
        df['Mental_Health_History'] +
        df['Growing_Stress'] +
        df['Quarantine_Frustrations']
    )

    # ------------------------------------------------------------------------------------------------
    # 11) Changes_Habits × Mood_Swings etkileşimi
    #     Alışkanlık değişimi (Changes_Habits) ile duygu dalgalanması (Mood_Swings) değerlerinin çarpımı.
    #     Yeni sütun: 'Habit_Mood_Interaction'
    # ------------------------------------------------------------------------------------------------
    df['Habit_Mood_Interaction'] = df['Changes_Habits'] * df['Mood_Swings']

    # ------------------------------------------------------------------------------------------------
    # 12) Work_Interest × Social_Weakness toplamı
    #     Work_Interest konusundaki dummy sütunlarını (örneğin 'Work_Interest_Maybe', 'Work_Interest_Yes')
    #     kullanarak orijinal Work_Interest değerini yeniden oluşturuyoruz:
    #       - Eğer 'Work_Interest_Maybe' = 1 ise value = 1
    #       - Eğer 'Work_Interest_Yes'   = 1 ise value = 2
    #       - Her ikisi 0 ise value = 0
    #     Sonra bu değeri, Social_Weakness dummy sütunlarının toplamıyla çarpıyoruz:
    #       WI_SW_Sum = Work_Interest_Value × (tüm Social_Weakness_* dummy sütunlarının toplamı)
    #     Eğer Work_Interest veya Social_Weakness sütunları hiç yoksa, WI_SW_Sum = 0 atanır.
    # ------------------------------------------------------------------------------------------------
    sw_dummy_cols = [c for c in df.columns if c.startswith("Social_Weakness_")]
    wi_dummy_cols = [c for c in df.columns if c.startswith("Work_Interest_")]
    if wi_dummy_cols and sw_dummy_cols:
        # Work_Interest değerini dummy sütunlarından yeniden oluştur
        df['Work_Interest_Value'] = (
            df.get('Work_Interest_Maybe', 0) * 1 +
            df.get('Work_Interest_Yes',   0) * 2
        )
        # WI_SW_Sum: Work_Interest_Value × sosyal zayıflık dummy sütunlarının toplamı
        df['WI_SW_Sum'] = df['Work_Interest_Value'] * df[sw_dummy_cols].sum(axis=1)
        # Geçici oluşturduğumuz 'Work_Interest_Value' sütununu temizliyoruz, artık gerek yok
        df.drop(columns=['Work_Interest_Value'], inplace=True)
    else:
        # Eğer dummy sütunlar yoksa, 0 atıyoruz
        df['WI_SW_Sum'] = 0

    # ------------------------------------------------------------------------------------------------
    # 13) Occupation_Count (Meslek sayısı)
    #     One-hot encode edilmiş Occupation_* sütunlarının her bir satırdaki toplamı,
    #     yani o satırdaki kullanıcıya ait meslek kategorilerinin sayısını verir.
    #     Yeni sütun: 'Occupation_Count'
    # ------------------------------------------------------------------------------------------------
    occ_dummy_cols = [c for c in df.columns if c.startswith("Occupation_")]
    df['Occupation_Count'] = df[occ_dummy_cols].sum(axis=1)

    # ------------------------------------------------------------------------------------------------
    # 14) Weight_Yes_Flag (Kilo değişimi 'Yes' bayrağı)
    #     Weight_Change sütunu 2 (Yes) olduğunda 1, diğer durumlarda 0 ata.
    #     Yeni sütun: 'Weight_Yes_Flag'
    # ------------------------------------------------------------------------------------------------
    df['Weight_Yes_Flag'] = (df['Weight_Change'] == 2).astype(int)

    # ------------------------------------------------------------------------------------------------
    # 15) Mood_Level (Mood_Swings tercile bucket’ları; quantile eşit ise 0)
    #     Mood_Swings sütunu için %33 ve %66'lık yüzdelik değerlerini hesaplayıp
    #     tercile aralıklara bölerek 0,1,2 şeklinde seviyeler oluşturuyoruz.
    #     Eğer yüzdelik değerler eşit çıkarsa tüm satırlara 0 atanır.
    #     Yeni sütun: 'Mood_Level'
    # ------------------------------------------------------------------------------------------------
    mood_terc = df['Mood_Swings'].quantile([0.33, 0.66]).values
    if mood_terc[0] < mood_terc[1]:
        df['Mood_Level'] = pd.cut(
            df['Mood_Swings'],
            bins=[-np.inf, mood_terc[0], mood_terc[1], np.inf],
            labels=[0, 1, 2]
        ).astype(int)
    else:
        # Tercile belirlenemediyse tüm satırlar 0 alır
        df['Mood_Level'] = 0

    # ------------------------------------------------------------------------------------------------
    # 16) Stress_Level (Growing_Stress tercile bucket’ları; quantile eşit ise 0)
    #     Growing_Stress sütunu için %33 ve %66'lık yüzdelik değerlerini hesaplayıp
    #     tercile aralıklara bölerek 0,1,2 şeklinde seviyeler oluşturuyoruz.
    #     Eğer yüzdelik değerler eşit çıkarsa tüm satırlara 0 atanır.
    #     Yeni sütun: 'Stress_Level'
    # ------------------------------------------------------------------------------------------------
    stress_terc = df['Growing_Stress'].quantile([0.33, 0.66]).values
    if stress_terc[0] < stress_terc[1]:
        df['Stress_Level'] = pd.cut(
            df['Growing_Stress'],
            bins=[-np.inf, stress_terc[0], stress_terc[1], np.inf],
            labels=[0, 1, 2]
        ).astype(int)
    else:
        # Tercile belirlenemediyse tüm satırlar 0 alır
        df['Stress_Level'] = 0

    # ------------------------------------------------------------------------------------------------
    # 17) Uzun Süre Kapalı Kalma bayrağı (Long_Time_Indoors)
    #     Days_Indoors >= 3 olduğunda 1, aksi halde 0.
    #     Yeni sütun: 'Long_Time_Indoors'
    # ------------------------------------------------------------------------------------------------
    df['Long_Time_Indoors'] = (df['Days_Indoors'] >= 3).astype(int)

    # ------------------------------------------------------------------------------------------------
    # Sonuç: Tüm yeni feature’lar DataFrame’e eklenmiş olarak geri döner.
    # ------------------------------------------------------------------------------------------------
    return df


# -----------------------------------------------------------
# Eğer bu dosyayı doğrudan çalıştırırsanız (__main__),
# işlenmiş veriyi okuyup engineer_features fonksiyonunu uygulayarak
# sonuçları kaydeden basit bir çıktı işlemi yapar.
# -----------------------------------------------------------
if __name__ == '__main__':
    from pathlib import Path

    # İşlenmiş (encoding yapılmış) CSV dosyasının yolu:
    raw   = Path(__file__).parents[1] / 'data' / 'processed' / 'mental_health_encoded.csv'
    # Yeni feature’ler eklenmiş haliyle kaydedilecek dosyanın yolu:
    out   = Path(__file__).parents[1] / 'data' / 'processed' / 'mental_health_fe_extended.csv'

    # CSV’yi pandas ile oku
    df0   = pd.read_csv(raw)
    # Yeni feature’leri ekle
    df1   = engineer_features(df0)
    # Dosyayı diske kaydet, index ekleme
    df1.to_csv(out, index=False)

    print(f"🔧 Feature‐engineered (extended) veri kaydedildi: {out}")
