#!/usr/bin/env python3
"""
train_pipeline_strategyA.py

Strateji A: Veriyi üçe bölerek (Train / Validation / Test), 
iç CV (GridSearchCV) ile Decision Tree için pruning ve hyperparametre seçimi yapar. 
En iyi parametreler validation aşamasında test edildikten sonra, 
son olarak test set’i üzerinde finalize edilmiş modelin performansını raporlar.

Adım adım:
 1) Veri yüklenir, preprocess_data ile dönüştürülür. (ordinal mapping, one-hot, target encode)
 2) X ve y ayrılır.
 3) X_trainval / X_test olarak %80/%20 split yapılır (stratify=y).
 4) X_trainval’i %60/%20 olacak şekilde X_train / X_val olarak ikiye böler.
 5) Sadece X_train / y_train üzerinde GridSearchCV (iç CV) ile pruning ve parametre taraması yapılır.
 6) Bulunan en iyi pipeline, X_train üzerinde 5-fold CV ile test edilir, 
    sonra validation set (X_val) üzerinde evaluate edilir.
 7) En iyi parametrelerle X_train+X_val birleşik (%80) set üzerinde yeniden fit edilir.
 8) Son olarak X_test üzerinde performans (classification_report, confusion_matrix, ROC/PR curve) ölçülür.
 9) İstersen --save-model argümanıyla final pipeline’ı kaydedebilirsin.

Kullanım:
  python train_pipeline_strategyA.py
  python train_pipeline_strategyA.py --save-model
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import (
    train_test_split,
    KFold,
    GridSearchCV
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

# Bizim modüller (aynı src klasöründe olduklarını varsayıyoruz)
from legacy.preprocessing2       import preprocess_data
from feature_engineering import engineer_features

# Sabitler
ROOT_DIR = Path(__file__).parents[1]
RAW_PATH = ROOT_DIR / 'data' / 'raw' / 'mental_health_finaldata_1.csv'
MODEL_DIR = ROOT_DIR / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def plot_roc_pr(y_true, y_prob, title_suffix=""):
    """
    Verilen gerçek etiketler (y_true) ve pozitif sınıf olasılıkları (y_prob) ile
    ROC ve Precision-Recall eğrilerini yan yana çizer.
    """
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
        help='Final pipeline’ı disk’e kaydet'
    )
    args = parser.parse_args()

    # === 1) Ham veriyi yükle ve ön işleme ===
    df_raw = pd.read_csv(RAW_PATH)
    # preprocess_data içindeki mapping ve one-hot işlemleri:
    df_proc = preprocess_data(df_raw)

    # X / y ayrımı
    X = df_proc.drop('Coping_Struggles', axis=1)
    y = df_proc['Coping_Struggles']

    # === 2) Hold-out split: %80 trainval, %20 test ===
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    # === 3) Trainval kümesini %60/~20 olacak şekilde train / val olarak böl ===
    # "trainval" toplamın %80’i idi, şimdi bunun %75’i train, %25’i val → sonuç %60/%20/%20
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.25,  # 0.25 * 0.80 = 0.20
        stratify=y_trainval,
        random_state=42
    )

    print(f"Dataset dağılımı:")
    print(f"  - Train    (total %): {len(X_train)}/{len(X)} = {len(X_train)/len(X):.2f}")
    print(f"  - Val      (total %): {len(X_val)}/{len(X)} = {len(X_val)/len(X):.2f}")
    print(f"  - Test     (total %): {len(X_test)}/{len(X)} = {len(X_test)/len(X):.2f}")
    print()

    # === 4) Pipeline tanımı (sadece FE + DecisionTree) ===
    base_pipeline = Pipeline([
        ('fe',  FunctionTransformer(engineer_features, validate=False)),
        ('clf', DecisionTreeClassifier(random_state=42, class_weight='balanced'))
    ])

    # === 5) GridSearchCV için parametre ızgarası ===
    param_grid = {
        'clf__max_depth':        [5, 10, 15, None],
        'clf__min_samples_leaf': [5, 10, 20, 50],
        'clf__min_samples_split':[10, 20, 50, 100],
        'clf__criterion':        ['gini', 'entropy'],
        'clf__ccp_alpha':        [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    }

    cv_inner = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=cv_inner,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    # === 6) Sadece X_train / y_train üzerinde GridSearchCV (pruning ve parametre seçimi) ===
    print("🔧 GridSearchCV başlatılıyor (Training set üzerinde)...")
    grid_search.fit(X_train, y_train)

    print("\n📌 En iyi parametreler (pruned & optimized):")
    print(grid_search.best_params_)
    best_pipeline = grid_search.best_estimator_

    # === 7) X_train üzerinde CV skorları (en iyi model ile) ===
    from sklearn.model_selection import cross_val_score
    scores_traincv = cross_val_score(
        best_pipeline, X_train, y_train,
        cv=cv_inner, scoring='accuracy', n_jobs=-1
    )
    print(f"\n🔍 En iyi model ile Train‐CV skorları: {scores_traincv}")
    print(f"🔍 Ortalama Train‐CV Doğruluk: {scores_traincv.mean():.4f}")

    # === 8) Validation set (X_val) üzerinde performans kontrolü ===
    print("\n=== Validation Set Performansı ===")
    # En iyi modeli X_train üzerinde eğit (kendi bünyesinde FE ve DT)
    best_pipeline.fit(X_train, y_train)

    # Validation tahminleri
    y_val_pred = best_pipeline.predict(X_val)
    y_val_prob = best_pipeline.predict_proba(X_val)[:, 1]

    print(classification_report(y_val, y_val_pred))
    cm_val = confusion_matrix(y_val, y_val_pred)
    print("Confusion Matrix (Val):\n", cm_val)

    roc_val = roc_auc_score(y_val, y_val_prob)
    pr_val  = average_precision_score(y_val, y_val_prob)
    print(f"Val ROC-AUC: {roc_val:.3f}")
    print(f"Val PR-AUC : {pr_val:.3f}")
    plot_roc_pr(y_val, y_val_prob, title_suffix="(Validation Set)")


    # === 9) Final Model: X_train+X_val birleşik set üzerinde yeniden fit ve test ===
    print("\n🔄 Final eğitim: X_train+X_val birleşik set’i kullanıyoruz.")
    X_trainval_all = pd.concat([X_train, X_val], axis=0)
    y_trainval_all = pd.concat([y_train, y_val], axis=0)

    # Aynı pipeline yapısını (özellik ve en iyi DT parametreleri) kullanıyoruz:
    final_pipeline = Pipeline([
        ('fe',  FunctionTransformer(engineer_features, validate=False)),
        ('clf', DecisionTreeClassifier(
                    max_depth      = grid_search.best_params_['clf__max_depth'],
                    min_samples_leaf  = grid_search.best_params_['clf__min_samples_leaf'],
                    min_samples_split = grid_search.best_params_['clf__min_samples_split'],
                    criterion         = grid_search.best_params_['clf__criterion'],
                    ccp_alpha         = grid_search.best_params_['clf__ccp_alpha'],
                    random_state=42,
                    class_weight='balanced'
                ))
    ])

    final_pipeline.fit(X_trainval_all, y_trainval_all)

    # === 10) Test set üzerinde performans raporu ===
    print("\n=== TEST SET Performansı ===")
    y_test_pred = final_pipeline.predict(X_test)
    y_test_prob = final_pipeline.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_test_pred))
    cm_test = confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix (Test):\n", cm_test)

    roc_test = roc_auc_score(y_test, y_test_prob)
    pr_test  = average_precision_score(y_test, y_test_prob)
    print(f"Test ROC-AUC: {roc_test:.3f}")
    print(f"Test PR-AUC : {pr_test:.3f}")
    plot_roc_pr(y_test, y_test_prob, title_suffix="(Test Set)")

    # === 11) Modeli kaydet ===
    if args.save_model:
        save_path = MODEL_DIR / 'decision_tree_stratA.pkl'
        joblib.dump(final_pipeline, save_path)
        print(f"\n💾 Final pipeline (StratejiA) kaydedildi --> {save_path}")


if __name__ == '__main__':
    main()
