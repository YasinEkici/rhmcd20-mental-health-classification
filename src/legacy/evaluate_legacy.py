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

def load_data(path: Path) -> pd.DataFrame:
    """
    CSV’den pandas DataFrame’e yükler.
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
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
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
    root        = Path(__file__).parents[1]
    # Burada MODEL_FILENAME’ı hangisini kullandığınıza göre değiştirin:
    #   - Strateji A için: 'decision_tree_stratA.pkl'
    #   - Strateji B için: 'decision_tree_stratB.pkl'
    MODEL_FILENAME = 'decision_tree_stratA.pkl'
    model_path  = root / 'models' / MODEL_FILENAME

    data_path   = root / 'data' / 'processed' / 'mental_health_encoded.csv'
    # (ÖNEMLİ) Burada FE’yi biz pipeline içinde uygulayacağımız için
    # veri mental_health_encoded.csv olmalı. Eğer “fe önceden uygulanmış” bir CSV’niz varsa,
    # pipeline FE’yi ikinci kez uygulayacağından doğru olmaz.

    # ----------------------------
    # 2) Model ve Veri Yükle
    # ----------------------------
    if not model_path.exists():
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {data_path}")

    model = load_model(model_path)  # Bu bir Pipeline nesnesi olmalı
    df    = load_data(data_path)    # Bu, sadece encoding’li, FE uygulanmamış DataFrame

    # ----------------------------
    # 3) X / y Ayrımı & Hold-out Test Set ayırma
    # ----------------------------
    # X: tüm input sütunları (Coping_Struggles dışındaki)
    X = df.drop('Coping_Struggles', axis=1)
    y = df['Coping_Struggles']

    # Aynı random_state ve stratify ile veri %80 train, %20 test olarak bölünsün
    # (Pipeline içindeki FE adımı, X_test üzerinde otomatik uygulanacak.)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    # ----------------------------
    # 4) Test Üzerinde Tahminler
    # ----------------------------
    # (Pipeline: önce engineer_features → sonra DecisionTreeClassifier)
    y_pred = model.predict(X_test)               # sınıf tahminleri
    try:
        y_prob = model.predict_proba(X_test)[:, 1]   # pozitif sınıf olasılıkları
    except AttributeError:
        # Eğer model.predict_proba yoksa (örneğin DecisionTreeClassifier değilse),
        # y_prob = None yapabilirsiniz. Ancak bizim durumda projemiz DT olduğu için olması gerekir.
        y_prob = None

    # ----------------------------
    # 5) Sınıflandırma Metrikleri
    # ----------------------------
    evaluate_classification(y_test, y_pred)

    if y_prob is not None:
        # Test set ROC-AUC & PR-AUC
        roc = roc_auc_score(y_test, y_prob)
        pr  = average_precision_score(y_test, y_prob)
        print(f"\nTest ROC-AUC: {roc:.3f}")
        print(f"Test PR-AUC : {pr:.3f}\n")

        plot_roc_pr_curves(y_test, y_prob, pos_label=1, title_suffix="(Test Set)")

    # ----------------------------
    # 6) Feature Importances
    # ----------------------------
    # Pipeline içindeki classifier’ı alıyoruz:
    try:
        clf = model.named_steps['clf']
    except (KeyError, AttributeError):
        raise ValueError("Pipeline içinde 'clf' adında bir adım bulunamadı veya model bir Pipeline değil.")

    # “Feature Importances” yalnızca karar ağaçlarında/ensemble’da bulunur
    if not hasattr(clf, 'feature_importances_'):
        print("Bu modelde feature_importances_ özelliği yok.")
        return

    # 1) İlk olarak FE adımının transform() ettiği X_test’i elde edelim:
    #    model.named_steps['fe'] bir FunctionTransformer ise, validate=False ile engineer_features
    #    doğrudan DataFrame alıp dönüştürüyor. O yüzden X_test’i direkt dönüştürebiliriz.
    try:
        fe_step = model.named_steps['fe']
        X_test_fe = fe_step.transform(X_test.copy())
    except Exception as e:
        raise RuntimeError(f"FE adımını (engineer_features) uygularken hata: {e}")

    # 2) FE’dan çıkan sütun isimlerini alın
    #    engineer_features DataFrame döndüreceği için columns özniteliği olacak.
    feature_names = list(X_test_fe.columns)

    importances = clf.feature_importances_
    # En yüksek 10 sütunu seç
    idxs = np.argsort(importances)[-10:]

    plt.figure(figsize=(6, 4))
    plt.barh([feature_names[i] for i in idxs], importances[idxs])
    plt.title("Top 10 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
