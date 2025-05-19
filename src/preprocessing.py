import pandas as pd
from pathlib import Path

# Define file paths
RAW_PATH = Path(__file__).parents[1] / 'data' / 'raw' / 'mental_health_finaldata_1.csv'
PROCESSED_PATH = Path(__file__).parents[1] / 'data' / 'processed' / 'mental_health_encoded.csv'


def load_data(path: Path) -> pd.DataFrame:
    """
    Load CSV data from a given path.
    """
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data preprocessing:
      - Check and report missing values
      - Label encode all categorical columns
    """
    # 1) Report missing values
    missing = df.isnull().sum()
    print("Missing values per column:\n", missing)

# 3) Ordinal kategorileri label encode et
    ord_cols = [
        'Age', 'Days_Indoors', 'Mood_Swings',
        'Growing_Stress', 'Quarantine_Frustrations',
        'Changes_Habits', 'Mental_Health_History', 'Weight_Change'
    ]
    for col in ord_cols:
        df[col] = df[col].astype('category').cat.codes

    # 4) Nominal kategorileri one-hot encode et
    nom_cols = ['Gender', 'Occupation', 'Work_Interest', 'Social_Weakness']
    df = pd.get_dummies(df, columns=nom_cols, drop_first=False)

    # 5) Hedef değişkeni (Coping_Struggles) label encode et
    df['Coping_Struggles'] = df['Coping_Struggles'].astype('category').cat.codes

    return df 

def save_data(df: pd.DataFrame, path: Path):
    """
    Save DataFrame to CSV at the given path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Processed data saved to: {path}")


if __name__ == '__main__':
    # Load
    df_raw = load_data(RAW_PATH)
    # Preprocess
    df_clean = preprocess_data(df_raw)
    # Save
    save_data(df_clean, PROCESSED_PATH)
