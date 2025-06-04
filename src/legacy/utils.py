# src/utils.py

from pathlib import Path
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    RocCurveDisplay,
    precision_recall_curve,
    PrecisionRecallDisplay
)


def load_data(path: Path) -> pd.DataFrame:
    """
    CSV’den DataFrame’e yükleme.
    Tek satırda veriyi alıp sütun isimlerini, tipleri vs. korur.
    """
    return pd.read_csv(path)


def save_model(model, path: Path):
    """
    Modeli disk’e seri hale getirerek kaydeder.
    :param model: sklearn objesi (örneğin DecisionTreeClassifier)
    :param path: .pkl olarak kaydedilecek dosya yolu
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path):
    """
    Daha önce kaydedilmiş bir modeli yükler.
    """
    return joblib.load(path)


def evaluate_classification(y_true, y_pred, labels=None):
    """
    - classification_report yazdırır.
    - Confusion Matrix’i görselleştirir.
    """
    print("=== Classification Report ===")
    print(classification_report(y_true, y_pred, labels=labels))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.show()


def plot_roc(y_true, y_score, pos_label=1):
    """
    ROC eğrisi ve AUC puanını çizdirir.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.title("ROC Curve")
    plt.show()


def plot_precision_recall(y_true, y_score, pos_label=1):
    """
    Precision-Recall eğrisi çizdirir.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=pos_label)
    PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    plt.title("Precision-Recall Curve")
    plt.show()


def plot_feature_importances(importances: np.ndarray, feature_names: list, top_n=10):
    """
    Feature importance’ları barplot olarak görselleştirir.
    :param importances: model.feature_importances_
    :param feature_names: X.columns list
    :param top_n: en yüksek N özelliği göster
    """
    indices = np.argsort(importances)[-top_n:]
    names = [feature_names[i] for i in indices]
    vals = importances[indices]

    plt.figure(figsize=(6, top_n * 0.4))
    plt.barh(names, vals)
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
