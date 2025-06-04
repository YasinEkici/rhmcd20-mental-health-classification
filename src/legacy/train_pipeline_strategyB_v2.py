#!/usr/bin/env python3
"""
train_pipeline_strategyB_v2.py

Strateji B (v2, Düzeltilmiş):  
  - Nested CV (iç ve dış katman) kullanarak out‐of‐fold tahminler elde eder.
  - Ancak grid_search.best_params_ hatası nedeniyle, nested CV'den sonra 
    en iyi parametreleri tüm veri üzerinde ayrı bir GridSearchCV ile tekrar bulup 
    final modelimizi oluştururuz.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import FunctionTransformer, OneHotEncoder
from sklearn.compose         import ColumnTransformer
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_val_predict
)
from sklearn.metrics         import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
from sklearn.tree            import DecisionTreeClassifier

from legacy.preprocessing2           import preprocess_data

ROOT_DIR = Path(__file__).parents[1]
RAW_PATH = ROOT_DIR / 'data' / 'raw' / 'mental_health_finaldata_1.csv'
MODEL_DIR = ROOT_DIR / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def finer_ordinal_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daha ince binleme:
      - Age: 0→"18_20", 1→"21_25", 2→"26_30", 3→"31_plus"
      - Days_Indoors: 0, 1, 2-3, 4+
      - Mood_Swings: 0-1, 2, 3, 4+
      - Growing_Stress: 0-1, 2, 3, 4+
    Age → Age_Group sütunu, diğerleri boolean bayraklar.
    """
    df2 = df.copy()

    # Age binning
    age_map = {0: '18_20', 1: '21_25', 2: '26_30', 3: '31_plus'}
    df2['Age_Group'] = df2['Age'].map(age_map)

    # Days_Indoors binning
    df2['Indoor_0']      = (df2['Days_Indoors'] == 0).astype(int)
    df2['Indoor_1']      = (df2['Days_Indoors'] == 1).astype(int)
    df2['Indoor_2_3']    = ((df2['Days_Indoors'] >= 2) & (df2['Days_Indoors'] <= 3)).astype(int)
    df2['Indoor_4_plus'] = (df2['Days_Indoors'] >= 4).astype(int)

    # Mood_Swings binning
    df2['Mood_0_1']    = (df2['Mood_Swings'] <= 1).astype(int)
    df2['Mood_2']      = (df2['Mood_Swings'] == 2).astype(int)
    df2['Mood_3']      = (df2['Mood_Swings'] == 3).astype(int)
    df2['Mood_4_plus'] = (df2['Mood_Swings'] >= 4).astype(int)

    # Growing_Stress binning
    df2['Stress_0_1']    = (df2['Growing_Stress'] <= 1).astype(int)
    df2['Stress_2']      = (df2['Growing_Stress'] == 2).astype(int)
    df2['Stress_3']      = (df2['Growing_Stress'] == 3).astype(int)
    df2['Stress_4_plus'] = (df2['Growing_Stress'] >= 4).astype(int)

    return df2


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    İleri düzey oran/etkileşim özellikleri:
      - Frustration_Stress_Ratio    = Quarantine_Frustrations / (Growing_Stress + 1)
      - Frustration_Stress_SqRatio  = (Quarantine_Frustrations^2) / (Growing_Stress + 1)
      - Stress_Sq_Frustration       = (Growing_Stress^2) / (Quarantine_Frustrations + 1)
      - Frustration_Stress_Diff     = Quarantine_Frustrations - Growing_Stress
      - Mood_Days_Product           = Mood_Swings * Days_Indoors
      - Weight_Med_History_Ratio    = Weight_Change / (Mental_Health_History + 1)
      - Weight_x_History            = Weight_Change * Mental_Health_History
    """
    df2 = df.copy()
    df2['Frustration_Stress_Ratio']    = df2['Quarantine_Frustrations'] / (df2['Growing_Stress'] + 1)
    df2['Frustration_Stress_SqRatio']  = (df2['Quarantine_Frustrations'] ** 2) / (df2['Growing_Stress'] + 1)
    df2['Stress_Sq_Frustration']       = (df2['Growing_Stress'] ** 2) / (df2['Quarantine_Frustrations'] + 1)
    df2['Frustration_Stress_Diff']     = df2['Quarantine_Frustrations'] - df2['Growing_Stress']
    df2['Mood_Days_Product']           = df2['Mood_Swings'] * df2['Days_Indoors']
    df2['Weight_Med_History_Ratio']    = df2['Weight_Change'] / (df2['Mental_Health_History'] + 1)
    df2['Weight_x_History']            = df2['Weight_Change'] * df2['Mental_Health_History']
    return df2


def build_preprocessing_pipeline():
    """
    Age_Group sütunu için one-hot encoder; diğer sütunlar passthrough.
    """
    onehot_age = Pipeline([
        ('select_age', FunctionTransformer(lambda df: df[['Age_Group']], validate=False)),
        ('oh',        OneHotEncoder(sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('age_ohe', onehot_age, ['Age_Group']),
        ],
        remainder='passthrough'
    )
    return preprocessor


def full_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) preprocess_data’dan gelen DataFrame’i alır.
    2) finer_ordinal_bins ile daha ince binler (Age, Days_Indoors, Mood_Swings, Growing_Stress).
    3) add_advanced_features ile yeni oran/etkileşim sütunları ekler.
    4) build_preprocessing_pipeline ile Age_Group’u one-hot, 
       diğer tüm sütunları passthrough ederek nihai bir DataFrame döner.
    """
    df_binned = finer_ordinal_bins(df)
    df_adv    = add_advanced_features(df_binned)

    preprocessor = build_preprocessing_pipeline()
    transformed = preprocessor.fit_transform(df_adv)

    age_ohe_categories = preprocessor.named_transformers_['age_ohe']\
                             .named_steps['oh']\
                             .get_feature_names_out(input_features=['Age_Group'])
    passthrough_cols = [col for col in df_adv.columns if col != 'Age_Group']

    df_final = pd.DataFrame(
        np.hstack([
            transformed[:, : len(age_ohe_categories)],
            transformed[:, len(age_ohe_categories):]
        ]),
        columns=list(age_ohe_categories) + passthrough_cols,
        index=df_adv.index
    )
    return df_final


def plot_roc_pr(y_true, y_prob, title_suffix=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax1)
    ax1.set_title(f"ROC Curve {title_suffix}")
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax2)
    ax2.set_title(f"Precision-Recall Curve {title_suffix}")
    plt.tight_layout()
    plt.show()


def evaluate_classification(y_true, y_pred):
    print("=== Classification Report ===")
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_title("Confusion Matrix")
    plt.show()


def main():
    # --- 1) Veri yükle ve preprocess_data (base encoding) ---
    df_raw  = pd.read_csv(RAW_PATH)
    df_proc = preprocess_data(df_raw)  # categorical→one-hot, ordinal→code, target kodlandı
    X_all   = df_proc.drop('Coping_Struggles', axis=1)
    y_all   = df_proc['Coping_Struggles']

    # --- 2) Pipeline tanımı: FE + DT ---
    pipeline = Pipeline([
        ('fe',  FunctionTransformer(full_feature_engineering, validate=False)),
        ('clf', DecisionTreeClassifier(random_state=42))
    ])

    # --- 3) Hiperparametre ızgarası (v2) ---
    param_grid = {
        'clf__max_depth':        [3, 5, 8],
        'clf__min_samples_leaf': [1, 2, 5],
        'clf__min_samples_split':[2, 5, 10],
        'clf__criterion':        ['gini', 'entropy'],
        'clf__ccp_alpha':        [5e-4, 1e-3, 2e-3],
        'clf__class_weight':     [
            'balanced',
            {0: 0.4, 1: 0.6},
            {0: 0.3, 1: 0.7},
            {0: 0.2, 1: 0.8}
        ]
    }

    # --- 4) Nested CV setupları ---
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # GridSearchCV (iç CV)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    print("🔄 Nested CV başlıyor (iç ve dış katmanlarda)...")
    # cross_val_predict ile “out-of-fold” tahminlerini almak:
    y_oof = cross_val_predict(
        grid_search,
        X_all,
        y_all,
        cv=outer_cv,
        method='predict',
        n_jobs=-1
    )
    y_oof_prob = cross_val_predict(
        grid_search,
        X_all,
        y_all,
        cv=outer_cv,
        method='predict_proba',
        n_jobs=-1
    )[:, 1]

    # --- 5) Out‐of‐fold metrıkler (Nested CV sonuçları) ---
    acc_oof = accuracy_score(y_all, y_oof)
    roc_oof = roc_auc_score(y_all, y_oof_prob)
    pr_oof  = average_precision_score(y_all, y_oof_prob)
    print(f"\nOut‐of‐Fold Accuracy : {acc_oof:.3f}")
    print(f"Out‐of‐Fold ROC‐AUC  : {roc_oof:.3f}")
    print(f"Out‐of‐Fold PR‐AUC   : {pr_oof:.3f}\n")

    plot_roc_pr(y_all, y_oof_prob, title_suffix="(Out‐of‐Fold)")

    # --- 6) Tüm veri üzerinde hiperparametre seçimi için yeni bir GridSearchCV ---
    print("🔧 Final parametre seçimi için tüm veri üzerinde ayrı GridSearchCV yapılıyor...")
    final_gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    final_gs.fit(X_all, y_all)

    best_params = final_gs.best_params_
    print("\n📌 Nested CV’den sonra tüm veri üzerinden seçilen en iyi parametreler:")
    print(best_params)

    # --- 7) Final pipeline’ı oluştur ve tüm veriyle eğit ---
    final_pipeline = Pipeline([
        ('fe',  FunctionTransformer(full_feature_engineering, validate=False)),
        ('clf', DecisionTreeClassifier(
            max_depth        = best_params['clf__max_depth'],
            min_samples_leaf = best_params['clf__min_samples_leaf'],
            min_samples_split= best_params['clf__min_samples_split'],
            criterion        = best_params['clf__criterion'],
            ccp_alpha        = best_params['clf__ccp_alpha'],
            class_weight     = best_params['clf__class_weight'],
            random_state=42
        ))
    ])
    final_pipeline.fit(X_all, y_all)

    # --- 8) Modeli diske kaydet ---
    save_path = MODEL_DIR / 'decision_tree_stratB_v2.pkl'
    joblib.dump(final_pipeline, save_path)
    print(f"\n💾 Final pipeline (StratejiB_v2) kaydedildi: {save_path}\n")

    # --- 9) Değerlendirme için evaluate_model.py’ı kullanın. ---
    # evaluate_model.py, bu yeni modeli (decision_tree_stratB_v2.pkl) yükleyip 
    # test için aşağıdaki adımları yapacak:
    #   - Aynı test split (test_size=0.20, stratify=y_all, random_state=42)
    #   - y_pred, y_prob → classification_report, confusion_matrix, ROC/PR eğrileri, feature importances
    #
    # Tek yapmanız gereken:
    #   python evaluate_model.py
    #
    # ve sonuçları incelemek. 
    # Bu kod satırlarında doğrudan test değerlendirmesi yer almıyor, 
    # çünkü evaluate_model.py zaten bu sorumluluğu alıyor.

if __name__ == '__main__':
    main()
