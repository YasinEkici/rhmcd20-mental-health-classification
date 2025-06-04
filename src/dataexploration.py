# dataexploration.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------------------------------------
# Aşağıdaki script, veri keşfi (exploratory data analysis) amacıyla örnekler içermektedir.
# Her adımda yaptığımız işlemler ve nedenleri, Türkçe olarak ayrıntılı şekilde açıklanacaktır.
# ------------------------------------------------------------------------------------------------

# 1) Veri setini yükleyin
#    - Windows ortamında çalışıyorsanız, pd.read_csv içinde CSV dosyanızın tam yolunu (örneğin
#      r'..\\data\\raw\\mental_health_finaldata_1.csv') şeklinde yazmanız gerekebilir.
#    - Bu örnekte, script dosyasının bir üst klasöründe "data/raw/mental_health_finaldata_1.csv"
#      dosyasının bulunduğunu varsayıyoruz. Yani script ile aynı klasörde değil, bir üst dizinde yer alıyor.
#    - pd.read_csv işlevi, CSV içeriğini pandas DataFrame’e yükler.
df = pd.read_csv('../data/raw/mental_health_finaldata_1.csv')

# ───────────────────────────────────────────────────────────────────────────────
# 2) FREQUENCY (SIKLIK) ve PERCENTAGE (YÜZDE) BAR GRAFİKLERİ (Her bir sütun için)
# ───────────────────────────────────────────────────────────────────────────────
# Amaç: Her bir sütundaki kategori değerlerinin kaç kez tekrarlandığını (frekans) ve
#       bu frekansın toplam veri kümesine göre yüzdesel oranını görselleştirmek.
# Adımlar:
#   a) df[col].value_counts() ile her sütunun benzersiz değerlerinin frekanslarını alıyoruz.
#   b) df[col].value_counts(normalize=True) * 100 ile aynı değerlerin yüzde dağılımlarını hesaplıyoruz.
#   c) df[col].mode()[0] ile o sütunun en sık tekrar eden (tepe değeri / mode) değerini alıyoruz.
#   d) Matplotlib kullanarak bar grafiği çiziyoruz. Çubukların üstüne frekans ve yüzde etiketleri ekliyoruz.
#   e) Eksik değer varsa, pandas bunları NaN olarak alır; burada NaN değerleri ayrı bir kategori olarak
#      değil, görselde atlayacağız. İsterseniz eksik değerleri de özel bir kategoriye ekleyebilirsiniz.
#
# Not: Eğer sütun çok sayıda benzersiz değere sahipse (örneğin yazı tipli serbest metin sütunları),
#      o sütunda görselleştirme okunması zorlaşabilir. Bu durumda yalnızca sınırlı benzersiz değerleri
#      veya gruplama (örneğin üst sıralardaki N değeri) göstermek isteyebilirsiniz.
for col in df.columns:
    # frekanslar: her benzersiz değerin kaç kere tekrarlandığını verir
    freq = df[col].value_counts()
    # yüzde: normalize=True ile yüzde oranlarını hesapla, *100 ile yüzde cinsine çevir
    perc = df[col].value_counts(normalize=True) * 100
    # mod (en sık tekrar eden değer)
    mode_val = df[col].mode()[0]
    
    # Yeni bir şekil (figure) ve eksen (axes) yaratıyoruz; boyutları 6×4 inç
    fig, ax = plt.subplots(figsize=(6, 4))
    # Bar grafiğini çiziyoruz; x ekseninde kategoriler (frekans.index), y ekseninde frekans değerleri
    bars = ax.bar(
        freq.index.astype(str),  # Kategori değerlerini str'e çeviriyoruz, döngü boyunca farklı tipler olabilir
        freq.values,             # Her bir kategorinin frekansı
        color='skyblue',         # Çubuk rengini açık mavi olarak belirliyoruz
        edgecolor='black'        # Çubuk kenar çizgilerini siyah yapıyoruz
    )
    
    # Her bir çubuğun (bar) üstüne, hem frekans hem de yüzde değeri ekleyelim
    for bar, p in zip(bars, perc.values):
        # bar.get_height() ile çubuğun yüksekliğini (frekansı) alıyoruz
        h = bar.get_height()
        # ax.text ile metni çiziyoruz:
        #   - x koordinatı: bar.get_x() + bar.get_width() / 2 → çubuğun tam ortası
        #   - y koordinatı: h + 1 → çubuğun tepesinin 1 birim üstü
        #   - metin: f'{int(h)}\n{p:.1f}%' → alt alta frekans ve yüzde (ondalık bir basamak)
        #   - ha='center', va='bottom' → yatay/dikey hizalama
        #   - fontsize=7 → metin boyutunu küçültüyoruz ki sığsın
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1,
            f'{int(h)}\n{p:.1f}%',
            ha='center',
            va='bottom',
            fontsize=7
        )
    
    # Başlık, eksen etiketleri ve eksen düzenlemeleri
    ax.set_title(f'{col} Dağılımı (Mode: {mode_val})', fontsize=10)  # Başlıkta hangi sütun ve mod bilgisi
    ax.set_xlabel(col)                                               # x ekseni etiketi
    ax.set_ylabel('Frekans')                                         # y ekseni etiketi
    # x ekseni için tick (işaret) değerlerini ayarlıyoruz:
    #   - range(len(freq.index)) → 0,1,2,... şeklinde dizinler
    ax.set_xticks(range(len(freq.index)))
    # x tick etiketleri: her bir frekans indeksi, string olarak gösterilecek
    ax.set_xticklabels(
        freq.index.astype(str),
        rotation=45,  # 45 derece döndürme, etiketlerin çakışmasını önlemek için
        ha='right',   # hizalama sağa
        fontsize=7    # küçük font boyutu
    )
    # subplot çevresindeki boşlukları otomatik ayarla
    plt.tight_layout()
    
    # İsteğe bağlı: Her grafiği ayrı bir PNG dosyasına kaydet
    #   Dosya adı: 'freq_{col}.png', DPI=150
    plt.savefig(f'freq_{col}.png', dpi=150)
    plt.show()  # Grafiği ekranda göster


# ───────────────────────────────────────────────────────────────────────────────
# 3) OLAP‐TARZI PIVOT: Age x Gender → Mental_Health_History (Count) ve HEATMAP
# ───────────────────────────────────────────────────────────────────────────────
# Amaç: Veri kümesinde "Age" ve "Gender" kombinasyonlarına göre "Mental_Health_History"
#       sütununun kaç kez (her satır bir kişi/adım) tekrarlandığını pivot tablo şeklinde görmek.
#       Ardından bu tabloyu ısı haritası (heatmap) olarak görselleştirmek.
#
# Adımlar:
#   a) pd.pivot_table(index='Age', columns='Gender', values='Mental_Health_History', aggfunc='count')
#      ile hangi yaş aralığında ve hangi cinsiyette kaç katılımcı olduğunu sayıyoruz. Zaten her bir satır
#      bir katılımcı olduğundan, count işlemi Mental_Health_History sütunundaki dolu hücreleri sayar.
#   b) fillna(0).astype(int) → eksik kombinasyonlarda NaN yerine 0 ve tam sayıya çevir.
#   c) plt.imshow ile 2D değerleri ısı haritası şeklinde gösteriyoruz.
#   d) Renk skalası (cmap='viridis') seçildi; dilerseniz başka bir colormap kullanabilirsiniz (örneğin 'plasma', 'inferno').
#   e) eksen etiketlerini ve başlığı ayarlıyoruz.
pivot_age_gender = pd.pivot_table(
    df,
    index='Age',                   # Satırlar: yaş aralıkları (örneğin '16-20', '20-25', vb.)
    columns='Gender',              # Sütunlar: cinsiyet kategorileri (örneğin 'Male', 'Female')
    values='Mental_Health_History', # Kaç kez sayılacağını merak ettiğimiz sütun
    aggfunc='count'                # Count, her bir hücrede o kombinasyonun kaç kayıt içerdiğini hesaplar
).fillna(0).astype(int)            # Eksik hücrelere 0 ve tam sayı (int) dönüşümü

# Isı haritasını (heatmap) çizeceğiz
fig, ax = plt.subplots(figsize=(6, 5))
# ax.imshow, pivot tablodaki değerleri 2 boyutlu bir matrise dönüştürür ve renklerle gösterir
im = ax.imshow(pivot_age_gender.values, aspect='auto', cmap='viridis')
# Renk çubuğu (colorbar) ekleyerek hangi renk hangi sayıya karşılık geliyor gösteriyoruz
plt.colorbar(im, ax=ax, label='Count')

# Eksen etiketleri: x ekseni için cinsiyet kategorileri, y ekseni için yaş aralıkları
ax.set_xticks(np.arange(len(pivot_age_gender.columns)))
ax.set_yticks(np.arange(len(pivot_age_gender.index)))
ax.set_xticklabels(pivot_age_gender.columns, rotation=0)  # Cinsiyet etiketleri yatay
ax.set_yticklabels(pivot_age_gender.index, rotation=0)    # Yaş etiketleri yatay

ax.set_xlabel('Gender')  # x ekseni açıklaması
ax.set_ylabel('Age')     # y ekseni açıklaması
ax.set_title('Age vs Gender\n(Mental_Health_History Count)')  # Başlık
plt.tight_layout()

# İsteğe bağlı: Isı haritasını dosyaya kaydet
plt.savefig('heatmap_age_gender_mhcount.png', dpi=150)
plt.show()


# ───────────────────────────────────────────────────────────────────────────────
# 4) CROSSTAB (Age, Gender) x Growing_Stress → HEATMAP
# ───────────────────────────────────────────────────────────────────────────────
# Amaç: Çoklu indeksli (multi-index) bir crosstab tablosu oluşturarak,
#       "Age" ve "Gender" kombinasyonlarına göre "Growing_Stress" dağılımını sayısal olarak görmek.
#       Ardından bunları bir ısı haritası (heatmap) şeklinde çizmek.
#
# Adımlar:
#   a) pd.crosstab(index=[df['Age'], df['Gender']], columns=df['Growing_Stress'])
#      Bu tablo, satırda (Age, Gender) kombine, sütunda her stres seviyesi (0,1,2,...),
#      hücrede o kombinasyonun kaç defa gerçekleştiği sayısı yer alır.
#   b) Görselleştirme için çoklu indeksli satır etiketlerini düz bir liste haline getiriyoruz:
#      row_labels = [f'{age}-{gen}' for age, gen in ct_age_gender_stress.index]
#      Böylece y ekseninde her satır için "Yaş-Cinsiyet" formatında bir metin yazabiliriz.
#   c) plt.imshow ile crosstab.values dizisini 2D matris olarak çiziyoruz.
#   d) Eksen etiketlerini ayarlayıp, başlık ekliyoruz.
ct_age_gender_stress = pd.crosstab(
    index=[df['Age'], df['Gender']],  # Çoklu indeks: (Age, Gender) kombinasyonları
    columns=df['Growing_Stress']       # Sütun: farklı stres seviyeleri
)

# Çoklu indeksli satır etiketlerini düzleştir:
#   Örneğin, (0, 'Male') → '0-Male'
row_labels = [f'{age}-{gen}' for age, gen in ct_age_gender_stress.index]

# Isı haritası çizimi
fig, ax = plt.subplots(figsize=(8, 6))
im2 = ax.imshow(ct_age_gender_stress.values, aspect='auto', cmap='plasma')
plt.colorbar(im2, ax=ax, label='Count')

# x ekseni: stres seviyeleri
ax.set_xticks(np.arange(len(ct_age_gender_stress.columns)))
ax.set_xticklabels(ct_age_gender_stress.columns, rotation=45, ha='right')

# y ekseni: "Age-Gender" etiketleri
ax.set_yticks(np.arange(len(row_labels)))
ax.set_yticklabels(row_labels, rotation=0, fontsize=7)

ax.set_xlabel('Growing_Stress')              # x ekseni açıklaması
ax.set_ylabel('Age - Gender')                # y ekseni açıklaması
ax.set_title('Age-Gender vs Growing_Stress (Count Heatmap)')  # Başlık
plt.tight_layout()

# İsteğe bağlı: Grafiği dosyaya kaydet
plt.savefig('heatmap_age_gender_vs_stress.png', dpi=150)
plt.show()


# ───────────────────────────────────────────────────────────────────────────────
# 5) Age x Growing_Stress’e Göre Occupation Dağılımı → ÇOKLU BAR GRAFİKLERİ
# ───────────────────────────────────────────────────────────────────────────────
# Amaç: Her bir "Growing_Stress" seviyesi için, "Age" grupları bazında
#       "Occupation" (meslek) dağılımını görmek. Yani stres seviyesi 0,1,2,... ayrı ayrı
#       çizimler oluşturarak, yaş-aralığına göre meslek frekanslarını çubuk grafiği şeklinde gösteriyoruz.
#
# Adımlar:
#   a) unique_stress = df['Growing_Stress'].unique() → veri kümesindeki benzersiz stres seviyelerini al.
#   b) Döngü içinde her bir stres seviyesi için alt küme oluştur: subdf = df[df['Growing_Stress'] == stress]
#   c) pd.crosstab(subdf['Age'], subdf['Occupation']) → Age ve Occupation arasındaki frekans tablosunu oluştur.
#   d) occ_counts.plot(kind='bar', ax=ax) → çoklu bar grafiği. Her çubuk, Age grubundaki her meslek kategorisinin
#      frekansını gösterir.
#   e) Efsaneyi (legend) grafiğin dışına yerleştirerek okunabilirliği arttırıyoruz.
#   f) Başlık, eksen etiketleri, döndürülmüş x tick etiketleri, tight_layout vb. düzenlemeler yapılıyor.
#
# Not: Eğer bazı Age-Occupation kombinasyonlarında hiç kayıt yoksa, crosstab otomatik olarak 0 ekler.
unique_stress = df['Growing_Stress'].unique()
for stress in unique_stress:
    # İlgili stres seviyesine sahip satırları filtrele
    subdf = df[df['Growing_Stress'] == stress]
    # Age x Occupation crosstab tablosu oluştur: satır Age, sütun Occupation, değer sayılar (freq)
    occ_counts = pd.crosstab(subdf['Age'], subdf['Occupation'])
    
    # Yeni bar grafiği için fig ve ax oluştur
    fig, ax = plt.subplots(figsize=(8, 5))
    # crosstab sonucunu pandas'ın plot fonksiyonuyla çiz; kind='bar' → çubuk grafik
    occ_counts.plot(kind='bar', ax=ax)
    
    # Başlık ve eksen etiketleri
    ax.set_title(f'Age vs Occupation\n(Growing_Stress = {stress})', fontsize=10)
    ax.set_xlabel('Age')       # x ekseni: Age grupları
    ax.set_ylabel('Frekans')   # y ekseni: meslek frekansı
    
    # Efsaneyi (legend) grafiğin dışına yerleştir:
    #   - title='Occupation' → legend başlığı
    #   - bbox_to_anchor=(1.02, 1) → legend'i grafiğin sağ üstüne yerleştir
    #   - loc='upper left' → anchor noktasını ayarlar
    #   - fontsize=7 → küçük font
    ax.legend(title='Occupation', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
    
    # X ekseni etiketlerini 45 derece döndür ve sağa hizala
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # İsteğe bağlı: Her grafiği farklı bir dosyaya kaydet
    plt.savefig(f'bar_age_occupation_stress_{stress}.png', dpi=150)
    plt.show()
