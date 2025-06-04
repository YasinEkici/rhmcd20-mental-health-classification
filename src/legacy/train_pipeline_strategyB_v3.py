# train_pipeline_strategyB_v3.py
# ----------------------------------
# “Strateji B v3” için nested‐CV + OOF metrikleri + grafikler (ROC & PR).
# Sonuçta en iyi parametreler seçilip pipeline diske kaydedilir.

from pathlib import Path
import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# --- 1) Dosya yolları ---
ROOT_DIR       = Path(__file__).parents[1]
RAW_PATH       = ROOT_DIR / 'data' / 'raw' / 'mental_health_finaldata_1.csv'
MODEL_DIR      = ROOT_DIR / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --- 2) Basit Ön İşleme: Label & One-Hot encoding ---
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) Eksik değer kontrolü (eksik yoksa 0 çıktısını yazdırır).
    2) Ordinal sütunları label‐encode eder.
    3) Nominal sütunları one‐hot encode eder.
    4) Hedef değişkeni Coping_Struggles’i label‐encode eder.
    """
    # 2.1) Eksik değer raporu
    missing = df.isnull().sum()
    print("Missing values per column:\n", missing)

    # 2.2) Ordinal kategorileri label encode et
    ord_cols = [
        'Age', 'Days_Indoors', 'Mood_Swings',
        'Growing_Stress', 'Quarantine_Frustrations',
        'Changes_Habits', 'Mental_Health_History', 'Weight_Change'
    ]
    for col in ord_cols:
        df[col] = df[col].astype('category').cat.codes

    # 2.3) Nominal sütunları one‐hot encode et
    nom_cols = ['Gender', 'Occupation', 'Work_Interest', 'Social_Weakness']
    df = pd.get_dummies(df, columns=nom_cols, drop_first=False)

    # 2.4) Hedef değişkeni label encode et
    df['Coping_Struggles'] = df['Coping_Struggles'].astype('category').cat.codes

    return df


# --- 3) Daha İnce Kategorize Etme (Binleme) ---
def finer_ordinal_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aşağıdaki sütunlar üzerinde “daha ince” kategorize etme yapar:
      - Age → Age_Group (4 sınıf: 18-20, 21-25, 26-30, 31_plus)
      - Days_Indoors → Days_In_Bin (0-1, 2-3, 4_plus)
      - Mood_Swings → Mood_Bin (0-1, 2-3, 4_plus)
      - Growing_Stress → Stress_Bin (0-1, 2-3, 4_plus)
      - Quarantine_Frustrations → Frustration_Bin (0-1, 2-3, 4_plus)
    Ardından orijinal sütunları bırakır, sadece yeni sütunlar ekler.
    """
    df2 = df.copy()

    # 3.1) Age_Group
    bins_age = [-0.1, 0.5, 1.5, 2.5, 3.5]  # "df['Age']" kodları 0,1,2,3 olacaktır
    labels_age = ['18-20', '21-25', '26-30', '31_plus']
    df2['Age_Group'] = pd.cut(df2['Age'], bins=bins_age, labels=labels_age)

    # 3.2) Days_In_Bin
    bins_days = [-0.1, 0.5, 2.5, 100]
    labels_days = ['0_1', '2_3', '4_plus']
    df2['Days_In_Bin'] = pd.cut(df2['Days_Indoors'], bins=bins_days, labels=labels_days)

    # 3.3) Mood_Bin
    bins_mood = [-0.1, 0.5, 2.5, 4.5]
    labels_mood = ['0_1', '2_3', '4_plus']
    df2['Mood_Bin'] = pd.cut(df2['Mood_Swings'], bins=bins_mood, labels=labels_mood)

    # 3.4) Stress_Bin
    bins_stress = [-0.1, 0.5, 2.5, 4.5]
    labels_stress = ['0_1', '2_3', '4_plus']
    df2['Stress_Bin'] = pd.cut(df2['Growing_Stress'], bins=bins_stress, labels=labels_stress)

    # 3.5) Frustration_Bin
    bins_frus = [-0.1, 0.5, 2.5, 4.5]
    labels_frus = ['0_1', '2_3', '4_plus']
    df2['Frustration_Bin'] = pd.cut(df2['Quarantine_Frustrations'], bins=bins_frus, labels=labels_frus)

    return df2


# --- 4) İleri Düzey Feature Engineering ---
def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Birbirleriyle etkileşim veya oran ilişkileri olan yeni sütunlar ekler:
      - Frustration_Stress_Ratio    : Quarantine_Frustrations / (Growing_Stress + 1)
      - Frustration_Stress_Diff     : Quarantine_Frustrations - Growing_Stress
      - Mood_Days_Product           : Mood_Swings * Days_Indoors
      - Weight_Med_History_Ratio    : Weight_Change / (Mental_Health_History + 1)
      - Weight_x_History            : Weight_Change * Mental_Health_History
    """
    df2 = df.copy()

    df2['Frustration_Stress_Ratio']    = df2['Quarantine_Frustrations'] / (df2['Growing_Stress'] + 1)
    df2['Frustration_Stress_Diff']     = df2['Quarantine_Frustrations'] - df2['Growing_Stress']
    df2['Mood_Days_Product']           = df2['Mood_Swings'] * df2['Days_Indoors']
    df2['Weight_Med_History_Ratio']    = df2['Weight_Change'] / (df2['Mental_Health_History'] + 1)
    df2['Weight_x_History']            = df2['Weight_Change'] * df2['Mental_Health_History']

    return df2


# --- 5) Preprocessing Pipeline (ColumnTransformer) ---
def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    OrdinalEncoder: “_Bin” sütunları
    OneHotEncoder : dummy sütunlar (pd.get_dummies sonucunda gelen nominal sütunlar)
    Remainder='passthrough': Geriye kalan tüm sayısal sütunlar olduğu gibi kalır.
    """
    # 5.1) “_Bin” sütunlarının olası kategorileri
    age_group_cats       = ['18-20', '21-25', '26-30', '31_plus']
    days_in_bin_cats     = ['0_1', '2_3', '4_plus']
    mood_bin_cats        = ['0_1', '2_3', '4_plus']
    stress_bin_cats      = ['0_1', '2_3', '4_plus']
    frustration_bin_cats = ['0_1', '2_3', '4_plus']

    # 5.2) Yalnızca “_Bin” sütunlarını OrdinalEncoder’a veriyoruz
    ord_cols = [
        'Age_Group',
        'Days_In_Bin',
        'Mood_Bin',
        'Stress_Bin',
        'Frustration_Bin'
    ]
    ord_categories = [
        age_group_cats,
        days_in_bin_cats,
        mood_bin_cats,
        stress_bin_cats,
        frustration_bin_cats
    ]

    # 5.3) OneHotEncoder’a gidecek sütunlar
    nom_cols = [
        'Gender_Female', 'Gender_Male',
        'Occupation_Corporate', 'Occupation_Housewife', 'Occupation_Others', 'Occupation_Student',
        'Work_Interest_No', 'Work_Interest_Yes',
        'Social_Weakness_Maybe', 'Social_Weakness_No', 'Social_Weakness_Yes'
    ]

    transformers = [
        ('ord', OrdinalEncoder(categories=ord_categories, dtype=int), ord_cols),
        ('oh',  OneHotEncoder(sparse_output=False, handle_unknown='ignore'), nom_cols)
    ]

    ct = ColumnTransformer(transformers=transformers, remainder='passthrough')
    return ct


# --- 6) “Ham Veri” → FE Pipeline Adımı ---
def full_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) finer_ordinal_bins
    2) add_advanced_features
    3) ColumnTransformer.fit_transform → numpy array
    4) DataFrame’e geri çevir (sütun adlarını elde ederek)
    """
    # 6.1) İnce binleme
    df_bins = finer_ordinal_bins(df)

    # 6.2) İleri özellik üretme
    df_adv = add_advanced_features(df_bins)

    # 6.3) ColumnTransformer’ı oluştur ve dönüştür
    preprocessor = build_preprocessing_pipeline()
    arr = preprocessor.fit_transform(df_adv)

    # 6.4) FE sonrası sütun isimlerini elde et
    #   a) OneHotEncoder’dan gelen sütun isimleri:
    oh_encoder = preprocessor.named_transformers_['oh']
    oh_cols    = oh_encoder.get_feature_names_out(input_features=oh_encoder.feature_names_in_)

    #   b) OrdinalEncoder’dan gelen sütun isimleri (ord_cols ile aynıdır)
    ord_cols = preprocessor.transformers_[0][2]  # ['Age_Group','Days_In_Bin',...]

    #   c) Geriye kalan “passthrough” sütunlar:
    nom_cols = preprocessor.transformers_[1][2]
    remainder_cols = [c for c in df_adv.columns if (c not in ord_cols) and (c not in nom_cols)]

    # 6.5) Nihai sütun sıralaması
    final_cols = list(oh_cols) + list(ord_cols) + remainder_cols

    # 6.6) DataFrame’e çevirip geri döndür
    df_final = pd.DataFrame(arr, columns=final_cols, index=df_adv.index)
    return df_final


# --- 7) Ana Fonksiyon: Nested CV + OOF Metrikleri + Grafikler + Model Kaydetme ---
def main():
    # 7.1) Ham veriyi yükle ve preprocess et (basit encode)
    df_raw  = pd.read_csv(RAW_PATH)
    df_proc = preprocess_data(df_raw)

    X_all = df_proc.drop('Coping_Struggles', axis=1)
    y_all = df_proc['Coping_Struggles']

    # 7.2) Pipeline: önce full_feature_engineering, sonra DecisionTree
    pipeline = Pipeline([
        ('fe',  FunctionTransformer(full_feature_engineering, validate=False)),
        ('clf', DecisionTreeClassifier(random_state=42))
    ])

    # 7.3) Nested CV ayarları
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 7.4) Hiperparametre ızgarası
    param_grid = {
        'clf__max_depth':        [3, 5, 7],
        'clf__min_samples_leaf': [2, 5, 10],
        'clf__min_samples_split':[5, 10, 20],
        'clf__criterion':        ['gini', 'entropy'],
        'clf__ccp_alpha':        [1e-4, 5e-4, 1e-3],
        'clf__class_weight':     ['balanced']
    }

    # 7.5) OOF dizilerini tanımla
    oof_pred_proba = np.zeros(len(X_all))
    oof_pred       = np.zeros(len(X_all), dtype=int)

    outer_scores_acc    = []
    outer_scores_rocauc = []
    outer_scores_ap     = []

    # 7.6) Dış katman (5 split) içinde GridSearchCV
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_all, y_all), start=1):
        X_train_outer, X_test_outer = X_all.iloc[train_idx], X_all.iloc[test_idx]
        y_train_outer, y_test_outer = y_all.iloc[train_idx], y_all.iloc[test_idx]

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=inner_cv,
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train_outer, y_train_outer)
        best_estimator = grid_search.best_estimator_

        # 7.6.b) Dış katman test seti tahminleri
        proba = best_estimator.predict_proba(X_test_outer)[:, 1]
        pred  = best_estimator.predict(X_test_outer)

        oof_pred_proba[test_idx] = proba
        oof_pred[test_idx]       = pred

        acc_val    = accuracy_score(y_test_outer, pred)
        rocauc_val = roc_auc_score(y_test_outer, proba)
        ap_val     = average_precision_score(y_test_outer, proba)

        outer_scores_acc.append(acc_val)
        outer_scores_rocauc.append(rocauc_val)
        outer_scores_ap.append(ap_val)

        print(f" Fold {fold}: Acc = {acc_val:.3f}, ROC-AUC = {rocauc_val:.3f}, PR-AUC = {ap_val:.3f}")

    # 7.7) OOF skorlarını yazdır
    print("\nOut‐of‐Fold Ortalama Sonuçlar:")
    print(f"  Accuracy : {np.mean(outer_scores_acc):.3f}")
    print(f"  ROC‐AUC   : {np.mean(outer_scores_rocauc):.3f}")
    print(f"  PR‐AUC    : {np.mean(outer_scores_ap):.3f}")

    # 7.8) OOF ROC ve Precision‐Recall Grafikleri
    plt.figure(figsize=(12, 5))
    # 7.8.a) ROC Curve (OOF)
    ax1 = plt.subplot(1, 2, 1)
    RocCurveDisplay.from_predictions(
        y_all, oof_pred_proba, pos_label=1, ax=ax1
    )
    ax1.set_title("ROC Curve (Out‐of‐Fold)")

    # 7.8.b) Precision‐Recall Curve (OOF)
    ax2 = plt.subplot(1, 2, 2)
    PrecisionRecallDisplay.from_predictions(
        y_all, oof_pred_proba, pos_label=1, ax=ax2
    )
    ax2.set_title("Precision‐Recall Curve (Out‐of‐Fold)")

    plt.tight_layout()
    plt.show()

    # 7.9) Tüm veri üzerinde son bir GridSearchCV (en iyi parametrelerin bulunması)
    final_grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=inner_cv,
        n_jobs=-1,
        verbose=0
    )
    final_grid.fit(X_all, y_all)

    best_params = final_grid.best_params_
    print("\n📌 En iyi parametreler (StratejiB_v3):")
    for k, v in best_params.items():
        print(f"    {k} = {v}")

    # 7.10) En iyi pipeline’ı kaydet
    best_pipeline = final_grid.best_estimator_
    model_path = MODEL_DIR / 'decision_tree_stratB_v3.pkl'
    joblib.dump(best_pipeline, model_path)
    print(f"\n💾 Final pipeline (StratejiB_v3) kaydedildi: {model_path}")


if __name__ == '__main__':
    main()
