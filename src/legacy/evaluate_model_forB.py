# evaluate_model.py

from pathlib import Path
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

# “Strateji B v3” eğitim kodunda yer alan preprocess ve FE fonksiyonlarını import ediyoruz.
#from train_pipeline_strategyB_v3 import preprocess_data, full_feature_engineering
from train_pipeline_strategyB_v4 import preprocess_data, full_feature_engineering


def load_data(path: Path) -> pd.DataFrame:
    """
    Ham CSV’den pandas DataFrame’e yükler.
    (Henüz FE uygulanmamış, sadece kategorik encode edilmemiş ham veridir.)
    """
    return pd.read_csv(path)


def load_model(path: Path):
    """
    joblib ile diske kaydedilmiş sklearn Pipeline modelini yükler.
    """
    return joblib.load(path)


def evaluate_classification(y_true, y_pred):
    """
    - classification_report: precision/recall/f1 skoru hesaplar ve yazdırır.
    - confusion_matrix: gerçek vs tahmin matrisini oluşturur ve görselleştirir.
    """
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


def plot_roc_pr_curves(y_true, y_prob, pos_label=1, title_suffix=""):
    """
    ROC eğrisi ve Precision-Recall eğrisini yan yana çizer.
    - y_prob: pozitif sınıfın tahmin olasılıkları.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC Curve
    RocCurveDisplay.from_predictions(
        y_true, y_prob, pos_label=pos_label, ax=ax1
    )
    ax1.set_title(f"ROC Curve {title_suffix}")

    # Precision-Recall Curve
    PrecisionRecallDisplay.from_predictions(
        y_true, y_prob, pos_label=pos_label, ax=ax2
    )
    ax2.set_title(f"Precision-Recall Curve {title_suffix}")

    plt.tight_layout()
    plt.show()


def main():
    # ----------------------------
    # 1) Dosya Yolları
    # ----------------------------
    root          = Path(__file__).parents[1]
    RAW_PATH      = root / 'data' / 'raw' / 'mental_health_finaldata_1.csv'
    MODEL_FILENAME = 'decision_tree_stratB_v3.pkl'
    model_path    = root / 'models' / MODEL_FILENAME

    # ----------------------------
    # 2) Model ve Ham Veriyi Yükle
    # ----------------------------
    if not model_path.exists():
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Ham veri dosyası bulunamadı: {RAW_PATH}")

    # a) Pipeline’ı yükle
    model = load_model(model_path)  # Bu bir Pipeline nesnesi (FE + DT) olmalı

    # b) Ham CSV’yi yükle
    df_raw = load_data(RAW_PATH)

    # c) Eğitim aşamasındaki preprocess adımını aynen burada tekrar uygula
    #    (One-Hot & Ordinal encode, Coping_Struggles label-encode vs.)
    #df_proc = preprocess_data(df_raw)

    # ----------------------------
    # 3) X / y Ayrımı & Hold-out Test Set ayırma
    # ----------------------------
    X = df_proc.drop('Coping_Struggles', axis=1)
    y = df_proc['Coping_Struggles']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    # ----------------------------
    # 4) Test Üzerinde Tahminler
    # ----------------------------
    # Pipeline: önce full_feature_engineering → sonra DecisionTreeClassifier
    y_pred = model.predict(X_test)               # sınıf tahminleri
    try:
        y_prob = model.predict_proba(X_test)[:, 1]   # pozitif sınıf olasılıkları
    except AttributeError:
        y_prob = None

    # ----------------------------
    # 5) Sınıflandırma Metrikleri
    # ----------------------------
    evaluate_classification(y_test, y_pred)

    if y_prob is not None:
        roc = roc_auc_score(y_test, y_prob)
        pr  = average_precision_score(y_test, y_prob)
        print(f"\nTest ROC-AUC: {roc:.3f}")
        print(f"Test PR-AUC : {pr:.3f}\n")

        plot_roc_pr_curves(y_test, y_prob, pos_label=1, title_suffix="(Test Set)")

    # ----------------------------
    # 6) Feature Importances
    # ----------------------------
    try:
        clf = model.named_steps['clf']
    except (KeyError, AttributeError):
        raise ValueError("Pipeline içinde 'clf' adında bir adım bulunamadı veya model bir Pipeline değil.")

    if not hasattr(clf, 'feature_importances_'):
        print("Bu modelde feature_importances_ özelliği yok.")
        return

    # a) FE adımını çalıştırarak X_test’i dönüştür
    try:
        fe_step = model.named_steps['fe']  # FunctionTransformer(full_feature_engineering)
        X_test_fe = fe_step.transform(X_test.copy())
    except Exception as e:
        raise RuntimeError(f"FE adımını (full_feature_engineering) uygularken hata: {e}")

    # b) FE sonrası sütun isimlerini al
    feature_names = list(X_test_fe.columns)

    importances = clf.feature_importances_
    idxs = np.argsort(importances)[-10:]  # En yüksek 10 özelliği seç

    plt.figure(figsize=(6, 4))
    plt.barh([feature_names[i] for i in idxs], importances[idxs])
    plt.title("Top 10 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
