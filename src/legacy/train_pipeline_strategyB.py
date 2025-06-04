#!/usr/bin/env python3
"""
train_pipeline_strategyB.py

Strateji B: Nested Cross-Validation (iç‐dış CV) ile hem model seçimi/pruning (iç CV) 
hem de genelleme performansı tahmini (dış CV) yapar. 
Tek bir hold‐out test set’i ayırmak yerine, tüm veri üzerinde nested CV gerçekleştirir.

Adım adım:
 1) Ham veri yüklenir, preprocess_data ile ön işleme (ordinal mapping, one-hot, target encode).
 2) X/y ayrılır.
 3) Pipeline: engineer_features + DecisionTree (henüz parametre yok).
 4) İç CV: GridSearchCV ile pruning ve parametre seçimi (ör. max_depth, min_samples_leaf, ccp_alpha).
 5) Dış CV: cross_val_score kullanarak her fold’da “içinde GridSearch” işlemi yapılır ve o fold test edilir.
 6) Elde edilen nested CV doğruluk skorları, gerçekçi genelleme performansı verir.
 7) (Opsiyonel) En iyi parametre kombinasyonu tek seferde seçip, tüm veriyle final model eğitilebilir ve kaydedilebilir.

Kullanım:
  python train_pipeline_strategyB.py
  python train_pipeline_strategyB.py --save-model
"""

import argparse
from pathlib import Path

import pandas as pd
import joblib
import numpy as np

from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import (
    KFold,
    GridSearchCV,
    cross_val_score
)
from sklearn.metrics       import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
from sklearn.tree          import DecisionTreeClassifier
import matplotlib.pyplot as plt

from legacy.preprocessing2       import preprocess_data
from feature_engineering import engineer_features

# Sabitler
ROOT_DIR = Path(__file__).parents[1]
RAW_PATH = ROOT_DIR / 'data' / 'raw' / 'mental_health_finaldata_1.csv'
MODEL_DIR = ROOT_DIR / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def plot_roc_pr(y_true, y_prob, title_suffix=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax1)
    ax1.set_title(f"ROC Curve {title_suffix}")
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax2)
    ax2.set_title(f"Precision-Recall Curve {title_suffix}")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save-model',
        action='store_true',
        help='Tüm veriyle en iyi parametreleri bulup fit edilen final modeli kaydet'
    )
    args = parser.parse_args()

    # ========== 1) Veri yükle & preprocess ==========
    df_raw = pd.read_csv(RAW_PATH)
    df_proc = preprocess_data(df_raw)  # ordinal mapping, one-hot, target map
    X = df_proc.drop('Coping_Struggles', axis=1)
    y = df_proc['Coping_Struggles']

    # ========== 2) Pipeline tanımı ==========
    base_pipeline = Pipeline([
        ('fe',  FunctionTransformer(engineer_features, validate=False)),
        ('clf', DecisionTreeClassifier(random_state=42, class_weight='balanced'))
    ])

    # ========== 3) Parametre ızgarası ==========
    param_grid = {
        'clf__max_depth':        [5, 10, 15, None],
        'clf__min_samples_leaf': [5, 10, 20, 50],
        'clf__min_samples_split':[10, 20, 50, 100],
        'clf__criterion':        ['gini', 'entropy'],
        'clf__ccp_alpha':        [0.0, 0.001, 0.01]
    }

    # ========== 4) Nested CV ==========
    #  - cv_inner: parametre seçimi (GridSearch) için
    #  - cv_outer: gerçek genelleme performansı için
    cv_inner = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_outer = KFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=cv_inner,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    print("🔄 Nested CV başlıyor (iç CV ile GridSearch ve dış CV ile skor)...")
    # cross_val_score, her outer fold’da:
    #   1) İç fold’larda grid_search.fit(...) çalıştırır
    #   2) En iyi estimator ile outer fold’un test kısmında predict eder
    nested_scores = cross_val_score(
        grid_search, X, y,
        cv=cv_outer,
        scoring='accuracy',
        n_jobs=-1
    )
    print(f"\nNested CV accuracies (5 outer folds): {nested_scores}")
    print(f"Nested CV mean accuracy             : {nested_scores.mean():.4f}")

    # ========== 5) (Opsiyonel) Tüm veriyle en iyi parametre seçimi ve eğitim ==========
    if args.save_model:
        print("\n🔧 Tüm veriyle en iyi parametreleri bulmak için grid search yapılıyor...")
        # Tekrar grid search, tüm veri üzerinde:
        grid_search.fit(X, y)
        print(f"En iyi parametreler (tüm veri): {grid_search.best_params_}")
        best_pipeline = grid_search.best_estimator_
        # Best pipeline zaten içinde FE ve best DT parametrelerini tutuyor
        # Tekrar tüm veriyle fit edilmiş durumda:
        best_pipeline.fit(X, y)

        # Modeli kaydet
        save_path = MODEL_DIR / 'decision_tree_stratB.pkl'
        joblib.dump(best_pipeline, save_path)
        print(f"💾 Final pipeline (StratejiB) kaydedildi --> {save_path}")

        # Ayrıca tüm veri üzerindeki ROC/PR skorlarını da gösterebiliriz:
        y_prob_all = best_pipeline.predict_proba(X)[:, 1]
        roc_all = roc_auc_score(y, y_prob_all)
        pr_all  = average_precision_score(y, y_prob_all)
        print(f"Tüm veri ROC-AUC (overfit check): {roc_all:.3f}")
        print(f"Tüm veri PR-AUC  (overfit check): {pr_all:.3f}")
        plot_roc_pr(y, y_prob_all, title_suffix="(Tüm Veri)")

    # Nested CV’nun dışında tekrar classification report vs. yazmaya gerek yok,
    # çünkü nested_scores zaten “gerçekçi genelleme hatasının” bir özetini verdi.
    # Eğer istersek, cross_val_predict ile out‐of‐fold tahminleri alıp ROC/PR çizebiliriz:
    from sklearn.model_selection import cross_val_predict

    print("\n🔄 Out-of-Fold tahminleri ile ROC/PR çiziliyor...")
    oof_probs = cross_val_predict(
        base_pipeline, X, y,
        cv=cv_outer,
        method='predict_proba',
        n_jobs=-1
    )[:, 1]

    roc_oof = roc_auc_score(y, oof_probs)
    pr_oof  = average_precision_score(y, oof_probs)
    print(f"Out-of-Fold ROC-AUC: {roc_oof:.3f}")
    print(f"Out-of-Fold PR-AUC : {pr_oof:.3f}")
    plot_roc_pr(y, oof_probs, title_suffix="(Out-of-Fold)")

    # Opsiyonel: Eğer istersen, tüm veri üzerinde parametre seçip final eğitim yaptıktan sonra,
    # test etmek için farklı bir veri geldiğinde (production), doğrudan   best_pipeline.predict(new_X)   diyebilirsin.
    # Burada test set ayrı olmadığı için “production test” örneği yok.

if __name__ == '__main__':
    main()
