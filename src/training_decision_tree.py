from pathlib import Path
import joblib
import pandas as pd

# Model, CV, metrikler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Görselleştirme
import matplotlib.pyplot as plt
import numpy as np


def load_data(path: Path) -> pd.DataFrame:
    """CSV’den DataFrame yükleme."""
    return pd.read_csv(path)


def evaluate_classification(y_true, y_pred):
    """
    - classification_report
    - Confusion Matrix görselleştirme
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


def plot_feature_importances(importances: np.ndarray, feature_names: list, top_n=10):
    """
    En önemli top_n özelliği yatay barplot ile çiz.
    """
    idxs = np.argsort(importances)[-top_n:]
    names = [feature_names[i] for i in idxs]
    vals  = importances[idxs]

    plt.figure(figsize=(6, top_n * 0.4))
    plt.barh(names, vals)
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


def main():
    # 1) Veriyi yükle
    fe_path = Path(__file__).parents[1] / 'data' / 'processed' / 'mental_health_fe.csv'
    df = load_data(fe_path)

    # 2) X/y ayrımı
    X = df.drop('Coping_Struggles', axis=1)
    y = df['Coping_Struggles']

    # 3) 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    clf = DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=30,
        random_state=42
    )

    # 4) Fold bazlı doğruluk
    scores = cross_val_score(clf, X, y, cv=kf, scoring='accuracy')
    print("Fold accuracies:", scores)
    print("Mean CV Accuracy:", scores.mean())

    # 5) Tüm veri için CV tahminleri → rapor + CM
    y_pred = cross_val_predict(clf, X, y, cv=kf)
    evaluate_classification(y, y_pred)

    # 6) Son olarak full-data ile eğit, feature importance
    clf.fit(X, y)
    plot_feature_importances(clf.feature_importances_, X.columns, top_n=10)

    # 7) Full data ile eğitildikten sonra modeli diske kaydet
    model_dir = Path(__file__).parents[1] / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'decision_tree.pkl'
    joblib.dump(clf, model_path)
    print(f"Trained model saved to: {model_path}")

if __name__ == '__main__':
    main()
