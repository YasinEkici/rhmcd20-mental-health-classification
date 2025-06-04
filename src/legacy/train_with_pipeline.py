#!/usr/bin/env python3
"""
train_pipeline.py

Bu script, ham mental health verisini alıp sırasıyla:
  1) preprocess_data ile manuel ordinal mapping ve one-hot encoding,
  2) engineer_features ile etkileşim / polinom / oran / bayrak özellikleri,
  3) DecisionTreeClassifier eğitimi (ister CV, ister GridSearchCV ile hiperparametre optimizasyonu),
  4) Son modeli disk’e kaydetme,
  5) (Opsiyonel) Basit testler
adımlarını tek bir komuttan çalıştırmanızı sağlar.

Kullanım:
  # Normal eğitim + 5-fold CV
  python train_pipeline.py

  # Hiperparametre optimizasyonu ile
  python train_pipeline.py --tune

  # Eğitilmiş modeli kaydet
  python train_pipeline.py --save-model

  # Preprocess ve feature_eng fonksiyonları için basit testleri çalıştır
  python train_pipeline.py --test
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.pipeline    import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import (
    KFold, cross_val_score, cross_val_predict,
    GridSearchCV, train_test_split
)
from sklearn.metrics     import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.tree        import DecisionTreeClassifier

# Projemizin modülleri
from legacy.preprocessing2       import preprocess_data
from feature_engineering import engineer_features

# Dosya yolları
ROOT_DIR     = Path(__file__).parents[1]
RAW_PATH     = ROOT_DIR / 'data' / 'raw' / 'mental_health_finaldata_1.csv'
ENCODED_PATH = ROOT_DIR / 'data' / 'processed' / 'mental_health_encoded.csv'
MODEL_DIR    = ROOT_DIR / 'models'
MODEL_PATH   = MODEL_DIR / 'decision_tree_pipeline.pkl'

def run_tests():
    """Preprocessing ve feature engineering fonksiyonları için basit doğrulama testleri."""
    print("🔍 Running basic tests...")
    df_raw = pd.read_csv(RAW_PATH)
    df_proc = preprocess_data(df_raw.copy())
    # 1) Tüm ordinal sütunlar int
    for col in ['Age','Days_Indoors','Mood_Swings',
                'Growing_Stress','Quarantine_Frustrations',
                'Changes_Habits','Mental_Health_History',
                'Weight_Change','Coping_Struggles']:
        assert pd.api.types.is_integer_dtype(df_proc[col]), f"{col} int değil!"
    # 2) engineer_features çıktısında beklenen yeni sütunlar var
    df_fe = engineer_features(df_proc.copy())
    for col in ['Stress_Mood_Interaction','Total_Stress_Frustration',
                'Frac_Frustration','Mood_Squared','Stress_Squared',
                'High_Total_Stress','High_Frustration_Rate']:
        assert col in df_fe.columns, f"{col} bulunamadı!"
    print("✅ All tests passed!")

def plot_curves(y_true, y_prob):
    """ROC ve Precision-Recall eğrilerini birlikte göster."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax1)
    ax1.set_title("ROC Curve")
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax2)
    ax2.set_title("Precision-Recall Curve")
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune',       action='store_true',
                        help='GridSearchCV ile hiperparametre optimizasyonu yap')
    parser.add_argument('--save-model', action='store_true',
                        help='Eğitilmiş pipeline’ı diske kaydet')
    parser.add_argument('--test',       action='store_true',
                        help='Preprocess ve FE fonksiyonları için testleri çalıştır')
    args = parser.parse_args()

    if args.test:
        run_tests()
        sys.exit(0)

    # 1) Ham veriyi yükle & preprocess
    df_raw  = pd.read_csv(RAW_PATH)
    df_proc = preprocess_data(df_raw)
    X       = df_proc.drop('Coping_Struggles', axis=1)
    y       = df_proc['Coping_Struggles']

    # 2) Pipeline: sadece feature engineering + classifier
    pipe = Pipeline([
        ('fe',  FunctionTransformer(engineer_features, validate=False)),
        ('clf', DecisionTreeClassifier(
                    random_state=42,
                    class_weight='balanced'
                ))
    ])

    # 3) CV veya GridSearch
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    if args.tune:
        param_grid = {
            'clf__max_depth':        [5, 10, 15],
            'clf__min_samples_leaf': [10, 30, 50],
            'clf__criterion':        ['gini','entropy'],
            'clf__ccp_alpha':        [0.0, 0.001, 0.01]
        }
        grid = GridSearchCV(
            pipe, param_grid,
            cv=kf, scoring='accuracy',
            n_jobs=-1, verbose=1
        )
        grid.fit(X, y)
        print("🔧 Best hyperparameters:", grid.best_params_)
        pipe = grid.best_estimator_
    else:
        print("🔍 5-fold CV sonuçları:")
        scores = cross_val_score(pipe, X, y, cv=kf, scoring='accuracy')
        print("  Fold accuracies:", scores)
        print("  Mean CV accuracy:", scores.mean())

        y_pred = cross_val_predict(pipe, X, y, cv=kf)
        print("\n=== Classification Report ===")
        print(classification_report(y, y_pred))
        cm = confusion_matrix(y, y_pred)
        print("Confusion matrix:\n", cm)

    # 4) Final fit ve opsiyonel kaydetme
    pipe.fit(X, y)
    y_prob = pipe.predict_proba(X)[:, 1]
    print(f"\nOverall training ROC-AUC: {roc_auc_score(y, y_prob):.3f}")
    print(f"Overall training PR-AUC : {average_precision_score(y, y_prob):.3f}")
    plot_curves(y, y_prob)

    if args.save_model:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, MODEL_PATH)
        print(f"💾 Trained pipeline saved to: {MODEL_PATH}")

if __name__ == '__main__':
    main()
