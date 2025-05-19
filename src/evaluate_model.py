# src/evaluate_model.py

from pathlib import Path
import pandas as pd
import joblib                                # Model yüklemek için
import matplotlib.pyplot as plt             # Grafik çizmek için
import numpy as np                          # İndeks/sıralama işleri için

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

def load_data(path: Path) -> pd.DataFrame:
    """
    CSV’den pandas DataFrame’e yükler.
    - Başlıkları, veri tiplerini ve eksikleri doğru alır.
    """
    return pd.read_csv(path)

def load_model(path: Path):
    """
    joblib ile diske kaydedilmiş sklearn modelini yükler.
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
    fig, ax = plt.subplots()
    im = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.title("Confusion Matrix")
    plt.show()

def plot_roc_pr_curves(y_true, y_prob, pos_label=1):
    """
    ROC eğrisi ve Precision-Recall eğrisini yan yana çizer.
    - y_prob: pozitif sınıfın tahmin olasılıkları.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC Curve
    RocCurveDisplay.from_predictions(
        y_true, y_prob, pos_label=pos_label, ax=ax1
    )
    ax1.set_title("ROC Curve")

    # Precision-Recall Curve
    PrecisionRecallDisplay.from_predictions(
        y_true, y_prob, pos_label=pos_label, ax=ax2
    )
    ax2.set_title("Precision-Recall Curve")

    plt.tight_layout()
    plt.show()

def main():
    # --- 1) Dosya yolları ---
    root       = Path(__file__).parents[1]
    model_path = root / 'models' / 'decision_tree.pkl'
    data_path  = root / 'data'  / 'processed' / 'mental_health_fe.csv'

    # --- 2) Model ve veri yükle ---
    model = load_model(model_path)
    df    = load_data(data_path)

    # --- 3) Test setini ayır ---
    #    Aynı random_state ve test_size kullanılarak gerçek test kümesini oluşturuyoruz.
    X = df.drop('Coping_Struggles', axis=1)
    y = df['Coping_Struggles']
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # --- 4) Tahminler ---
    y_pred = model.predict(X_test)               # Sınıf tahminleri
    y_prob = model.predict_proba(X_test)[:, 1]   # Pozitif sınıf olasılıkları

    # --- 5) Sınıflandırma metrikleri ---
    evaluate_classification(y_test, y_pred)

    # --- 6) ROC ve Precision-Recall eğrileri ---
    plot_roc_pr_curves(y_test, y_prob, pos_label=1)

    # --- 7) Feature Importances (isteğe bağlı) ---
    importances = model.feature_importances_
    feature_names = X.columns.to_list()
    idxs = np.argsort(importances)[-10:]  # En yüksek 10 özelliği seç
    plt.figure(figsize=(6, 4))
    plt.barh(
        [feature_names[i] for i in idxs],
        importances[idxs]
    )
    plt.title("Top 10 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
