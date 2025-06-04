# train_pipeline_strategyB_v4.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    cross_validate,
    GridSearchCV,
    cross_val_predict
)
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

# =============================================================================
# 1) CSV OKUMA ve ÖN İŞLEMLER (Tip Dönüşümleri + Eksik Değer İmputation)
# =============================================================================

# Path nesnesiyle proje kök dizinini belirliyoruz. 
# Path(__file__).parents[1) dosya konumunun bir üst klasörünü işaret eder.
root = Path(__file__).parents[1]

# İşlenmiş verinin bulunduğu dosya yolunu oluşturuyoruz:
data_path = root / 'data' / 'processed' / 'mental_health_finaldata_1.csv'

# Dosya yoksa hata fırlatır. Böylece eksik dosya durumunda erken uyarı alırız.
if not data_path.exists():
    raise FileNotFoundError(f"Veri dosyası bulunamadı: {data_path}")

# CSV dosyasını pandas DataFrame olarak yüklüyoruz.
df = pd.read_csv(data_path)

# ---- 1.1) Önce sütunların dtypes'ını kontrol edelim:
#      Veri okunduktan sonra hangi sütunların hangi tipte olduğunu ekrana yazdırıyoruz.
print("🔍 Okunduktan sonra sütun tipleri:\n", df.dtypes, "\n")

# ---- 1.2) Nümerik olması gereken sütunları sayıya çevirelim (hatayı NaN'a döndür)
#      Bazı sütunlar metin olarak veya karışık tipte gelebilir; pd.to_numeric ile hata durumunda NaN oluşacak.
numeric_cols = [
    'Age',
    'Days_Indoors',
    'Growing_Stress',
    'Quarantine_Frustrations',
    'Changes_Habits',
    'Mental_Health_History',
    'Weight_Change',
    'Mood_Swings'
]
# apply(pd.to_numeric, errors='coerce') → dönüştürülemeyen hücreler NaN yapılır
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# ---- 1.3) Satır bazlı NaN oluşumunu raporla (kaçı NaN oldu?)
before_drop = len(df)
# Her bir numeric sütunda kaç NaN olduğunu sayıyoruz
nan_counts = df[numeric_cols].isnull().sum()
print("🔍 Numeric sütunlardaki NaN sayıları (dönüşüm sonrası):\n", nan_counts, "\n")

# Bu örnekte dropna kullanmak yerine eksik değerleri sütun medyanıyla dolduracağız.
for col in numeric_cols:
    median_val = df[col].median()             # Her sütunun medyanını hesapla
    df[col].fillna(median_val, inplace=True)  # NaN'ları medyanla doldur

# İmpute sonrası toplam NaN sayısını kontrol edelim (hepsi 0 olmalı)
after_impute_nan = df[numeric_cols].isnull().sum().sum()
print(f"🔍 Tüm numeric sütunlarda toplam NaN sayısı (impute sonrası): {after_impute_nan}\n")

# Artık NaN kalmadığı için satır kaybı yaşanmadı. 
# Satır sayısını tekrar kontrol ediyoruz (drop işlemi yapılmadı).
print("ℹ️ İşlem sonrası satır sayısı:", len(df), "\n")


# =============================================================================
# 2) TARGET ve FEATURE AYIRMA (%80 Train+Val – %20 Test)
# =============================================================================

# Target sütununu y olarak atıyoruz
y = df['Coping_Struggles']
# X, yani özellikler kısmı ise 'Coping_Struggles' sütunu düşülmüş DataFrame
X = df.drop(columns=['Coping_Struggles'])

# Stratify özelliğiyle y dağılımına göre bölerek veri kümesini %80 train+val ve %20 test olarak ayırıyoruz
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y,
    test_size=0.20,      # %20 test
    stratify=y,          # Hedef dağılımını koru
    random_state=42      # Tekrarlanabilirlik için sabit seed
)

print(f"🔍 Dataset dağılımı:\n"
      f"  - Train+Val: {len(X_trainval)}/{len(df)} = {len(X_trainval)/len(df):.2f}\n"
      f"  - Test     : {len(X_test)}/{len(df)} = {len(X_test)/len(df):.2f}\n")


# =============================================================================
# 3) FEATURE ENGINEERING (FunctionTransformer İçinde)
# =============================================================================

def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Bu fonksiyon, ham DataFrame'i alıp yeni feature sütunları ekler.
    3.1) Nümerik etkileşim/oran/kuvvet sütunları ekler:
         - Mood_Days_Product: Mood_Swings * Days_Indoors
         - Weight_Med_History_Ratio: Weight_Change / (Mental_Health_History + 1)
         - Frustration_Stress_Diff: Quarantine_Frustrations - Growing_Stress
         - Stress_Sq_Frustration: Growing_Stress^2 + Quarantine_Frustrations
         - Frustration_Stress_SqRatio: (Quarantine_Frustrations + 1) / (Growing_Stress^2 + 1)
         - Weight_x_History: Weight_Change * Mental_Health_History

    3.2) Age'i kategorilere ayır:
         - Age_Group: 0-25, 26-40, 41+ aralıklarında binning yapar.

    3.3) Kategorik sütunları (Gender, Occupation, Work_Interest, Social_Weakness, Age_Group)
         one-hot encode (dummy öznitelik) formuna dönüştür.

    3.4) Hedef sütun (Coping_Struggles) varsa kaldır (train aşamasında zaten yok, test aşamasında da önceden çıkarıldı).

    Dönüş: Yeni eklenmiş sütunlarla birlikte tamamen sayısal/kodlanmış bir DataFrame döner.
    """
    # Orijinal DataFrame'i değiştirmemek için bir kopyasını al
    df2 = df_raw.copy()

    # --- 3.1) Nümerik etkileşim/oran/kuvvet sütunları ekleme:

    # Mood_Days_Product: Mood_Swings ile Days_Indoors çarpımı
    df2['Mood_Days_Product'] = df2['Mood_Swings'] * df2['Days_Indoors']

    # Weight_Med_History_Ratio: Kilo değişimi / (Ruh sağlığı geçmişi + 1)
    # +1 ekleyerek bölme sıfıra bölme hatasını önlüyoruz.
    df2['Weight_Med_History_Ratio'] = df2['Weight_Change'] / (df2['Mental_Health_History'] + 1)

    # Frustration_Stress_Diff: Hayal kırıklığı - stres farkı
    df2['Frustration_Stress_Diff'] = df2['Quarantine_Frustrations'] - df2['Growing_Stress']

    # Stress_Sq_Frustration: Stres karesi + hayal kırıklığı
    df2['Stress_Sq_Frustration'] = df2['Growing_Stress']**2 + df2['Quarantine_Frustrations']

    # Frustration_Stress_SqRatio: (Hayal kırıklığı + 1) / (Stres^2 + 1)
    df2['Frustration_Stress_SqRatio'] = (df2['Quarantine_Frustrations'] + 1) / (df2['Growing_Stress']**2 + 1)

    # Weight_x_History: Kilo değişimi * ruh sağlığı geçmişi
    df2['Weight_x_History'] = df2['Weight_Change'] * df2['Mental_Health_History']

    # --- 3.2) Age’i kategorilere ayıralım:
    #     Binning yapılırken bin sınırları [0,25,40,100] kabul edildi.
    #     labels ile kategorik etiketleri belirledik.
    df2['Age_Group'] = pd.cut(
        df2['Age'],
        bins=[0, 25, 40, 100],
        labels=['0_25', '26_40', '41_plus']
    ).astype(str)  # Kategorik etiketleri string olarak saklamak için .astype(str)

    # --- 3.3) Kategorik sütunlar:
    categorical_cols = [
        'Gender',
        'Occupation',
        'Work_Interest',
        'Social_Weakness',
        'Age_Group'
    ]

    # --- 3.4) One-hot encode:
    #     drop_first=False, çünkü bir kategori referans olarak düşürmek istemiyoruz; 
    #     tüm dummy sütunlarını ayrı ayrı tutacağız.
    df_encoded = pd.get_dummies(df2, columns=categorical_cols, drop_first=False)

    # --- 3.5) Hedef sütun hâlâ varsa kaldıralım
    df_encoded = df_encoded.drop(columns=['Coping_Struggles'], errors='ignore')

    # Yeni eklenen tüm sütunlarla birlikte DataFrame'i döndür
    return df_encoded

# FunctionTransformer: yukarıdaki engineer_features fonksiyonunu pipeline içinde kullanabilmek için sarıyoruz.
fe_transformer = FunctionTransformer(engineer_features, validate=False)


# =============================================================================
# 4) PIPELINE ve HYPERPARAMETER GRID
# =============================================================================

# Pipeline adımları:
#  - 'fe' kısmında: engineer_features fonksiyonu uygulanacak
#  - 'clf' kısmında: DecisionTreeClassifier (sabit random_state ile) 
pipe = Pipeline([
    ('fe', fe_transformer),
    ('clf', DecisionTreeClassifier(random_state=42))
])

# DecisionTreeClassifier için ayarlanacak hiperparametre ızgarası
param_grid = {
    'clf__max_depth': [3, 5, 7],             # Ağacın maksimum derinliği
    'clf__min_samples_leaf': [2, 5, 10],     # Her yaprakta en az kaç örnek olmalı
    'clf__min_samples_split': [2, 5, 10],    # Bir düğüm kaç örnekte split edilmeli
    'clf__ccp_alpha': [0.0, 0.0001],         # Cost-complexity pruning parametresi
    'clf__criterion': ['gini', 'entropy'],   # Bölünme ölçütü
    'clf__class_weight': [None, 'balanced']  # Sınıf dengesizliği varsa dengeli ağırlık
}


# =============================================================================
# 5) NESTED CV: 5-fold Outer, 3-fold Inner
# =============================================================================

# Dış döngü (outer) için StratifiedKFold: 5 katlı, shuffle=True, random_state sabit
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# İç döngü (inner) için StratifiedKFold: 3 katlı, shuffle=True, random_state sabit
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Değerlendirme metrikleri: accuracy, roc_auc, average_precision
scoring = ['accuracy', 'roc_auc', 'average_precision']

# cross_validate ile nested CV gerçekleştiriyoruz.
# Ref: https://scikit-learn.org/stable/modules/cross_validation.html#nested-cross-validation
nested_cv_results = cross_validate(
    estimator=GridSearchCV(
        estimator=pipe,          # Pipeline: fe + DecisionTree
        param_grid=param_grid,   # Hyperparametre ızgarası
        cv=inner_cv,             # İç döngü: 3-fold CV
        scoring='roc_auc',       # İç döngü optimizasyonu ROC-AUC üzerinden
        n_jobs=-1,               # Paralel hesaplama (tüm çekirdekleri kullan)
        refit=False              # Sonuçları doğrudan refit etme (dış döngü sonuçlarına bakacağız)
    ),
    X=X_trainval,                # Train+Val verisi
    y=y_trainval,                # Train+Val hedef
    cv=outer_cv,                 # Dış döngü: 5-fold CV
    scoring=scoring,             # Dış döngüde ölçmek istediğimiz metrikler
    return_estimator=True,       # Her outer katman sonrası estimator nesnelerini geri dön
    n_jobs=-1                    # Paralel hesaplama
)

# Her bir fold için test_accuracy, test_roc_auc, test_average_precision değerlerini yazdır
for i in range(len(nested_cv_results['test_accuracy'])):
    print(f"Fold {i+1}: "
          f"Acc = {nested_cv_results['test_accuracy'][i]:.3f}, "
          f"ROC-AUC = {nested_cv_results['test_roc_auc'][i]:.3f}, "
          f"PR-AUC = {nested_cv_results['test_average_precision'][i]:.3f}"
    )

print("\n🔄 Out-of-Fold Ortalama Sonuçlar:")
print(f"  Accuracy : {np.mean(nested_cv_results['test_accuracy']):.3f}")
print(f"  ROC-AUC   : {np.mean(nested_cv_results['test_roc_auc']):.3f}")
print(f"  PR-AUC    : {np.mean(nested_cv_results['test_average_precision']):.3f}\n")


# =============================================================================
# 6) EN İYİ PARAMETRELER VE FİNAL PIPE
# =============================================================================

# ROC-AUC'ya göre en yüksek skoru veren fold'u buluyoruz
best_fold_index = np.argmax(nested_cv_results['test_roc_auc'])
# İlgili fold'un GridSearchCV nesnesini alıyoruz
best_inner_gs = nested_cv_results['estimator'][best_fold_index]
# Eğer best_inner_gs içinde best_params_ varsa çek, yoksa None
best_params = best_inner_gs.best_params_ if hasattr(best_inner_gs, 'best_params_') else None

if best_params is None:
    print("⚠️ İç CV’den best_params alınamadı (refit=False).")
    print("   → Final aşamada tüm trainval üzerinde yeniden GridSearchCV yapılacak.\n")
else:
    print(f"📌 Nested CV’den elde edilmiş örnek best_params: {best_params}\n")

# Final aşamada Pipeline'ı yeniden oluşturup, trainval verisiyle GridSearchCV yapacağız
final_pipe = Pipeline([
    ('fe', fe_transformer),
    ('clf', DecisionTreeClassifier(random_state=42))
])
final_grid = GridSearchCV(
    estimator=final_pipe,
    param_grid=param_grid,
    cv=5,                    # 5-fold CV (trainval üzerinde)
    scoring='roc_auc',       # ROC-AUC optimizasyonu
    n_jobs=-1                # Paralel hesaplama
)
# Train+Val verisi üzerinde en iyi parametreleri bul
final_grid.fit(X_trainval, y_trainval)

print("📌 Final aşamada (Train+Val tümü) seçilen best_params (StratejiB_v4):")
print(final_grid.best_params_, "\n")

# En iyi tahmin eden pipeline örneğini best_pipe olarak alalım
best_pipe = final_grid.best_estimator_

# =============================================================================
# 7) OUT-OF-FOLD (OOF) ROC & PR EĞRİLERİ
# =============================================================================

# cross_val_predict ile trainval üzerinde out-of-fold predict_proba değerlerini elde ediyoruz.
# method='predict_proba' → tahmin edilen olasılık skorlarını döndürür
y_prob_oof = cross_val_predict(
    estimator=best_pipe,
    X=X_trainval,
    y=y_trainval,
    cv=outer_cv,                # Dış döngü (aynı outer_cv kullanılıyor)
    method='predict_proba',     # Olasılık tahmini
    n_jobs=-1
)[:, 1]                        # Pozitif sınıfın (1) olasılık skorları

print("🔄 Out-of-Fold ROC & PR eğrileri:")
# ROC eğrisini çiziyoruz (trainval OOF sonuçları)
RocCurveDisplay.from_predictions(y_trainval, y_prob_oof).plot()
plt.title("ROC Curve (Out-of-Fold)")
plt.show()

# Precision-Recall eğrisini çiziyoruz (trainval OOF sonuçları)
PrecisionRecallDisplay.from_predictions(y_trainval, y_prob_oof).plot()
plt.title("Precision-Recall Curve (Out-of-Fold)")
plt.show()


# =============================================================================
# 8) TEST SET PERFORMANSI
# =============================================================================

# Test set üzerinde doğrudan predict ve predict_proba ile tahminleri alıyoruz:
y_pred_test = best_pipe.predict(X_test)
y_prob_test = best_pipe.predict_proba(X_test)[:, 1]  # Olasılık skoru (pozitif sınıf)

print("=== TEST SET Performansı ===")
# Sınıflandırma raporu (precision, recall, f1-score, destek sayısı)
print(classification_report(y_test, y_pred_test))
# Test set ROC-AUC skoru
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_prob_test):.3f}")
# Test set PR-AUC skoru (Average Precision)
print(f"Test PR-AUC  : {average_precision_score(y_test, y_prob_test):.3f}\n")

# ROC eğrisini Test set için çiz
RocCurveDisplay.from_predictions(y_test, y_prob_test).plot()
plt.title("ROC Curve (Test Set)")
plt.show()

# Precision-Recall eğrisini Test set için çiz
PrecisionRecallDisplay.from_predictions(y_test, y_prob_test).plot()
plt.title("Precision-Recall Curve (Test Set)")
plt.show()

# Karışıklık matrisi (confusion matrix) çizimi:
cm = confusion_matrix(y_test, y_pred_test)
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.matshow(cm, cmap=plt.cm.Blues)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
plt.title("Confusion Matrix (Test)")
plt.colorbar(im, ax=ax)
plt.show()


# =============================================================================
# 9) FEATURE IMPORTANCES (Önemli Özellikler)
# =============================================================================

# DecisionTreeClassifier adımını çıkartıp feature_importances_ özelliğini alıyoruz
clf_final = best_pipe.named_steps['clf']
# Test seti üzerinde Feature Engineering uygulanmış haliyle X_test'i elde etmeliyiz
X_test_fe = engineer_features(X_test)
# Feature isimleri, DataFrame sütun adlarından alınır
feature_names = X_test_fe.columns.to_list()

# Modelin attribute'u olan feature_importances_ dizisini kullanıyoruz
importances = clf_final.feature_importances_
# En yüksek 10 özelliğin indekslerini alıyoruz (küçükten büyüğe sıralayıp son 10'u dilimliyoruz)
indices = np.argsort(importances)[-10:]

# Çubuk grafik (barh) çiziyoruz: yatay barlar en yüksek 10 önemli feature için
fig, ax = plt.subplots(figsize=(6, 4))
ax.barh([feature_names[i] for i in indices], importances[indices])
ax.set_title("Top 10 Feature Importances")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.show()


# =============================================================================
# 10) MODELİ KAYDET
# =============================================================================

# best_pipe pipeline'ını joblib ile diske kaydediyoruz, böylece yeniden eğitim yapmadan yükleyebiliriz.
model_path = root / 'models' / 'decision_tree_stratB_v4.pkl'
joblib.dump(best_pipe, model_path)
print(f"\n💾 Final pipeline (StratejiB_v4) kaydedildi: {model_path}\n")
