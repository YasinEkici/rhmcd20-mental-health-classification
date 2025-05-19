from pathlib import Path
import pandas as pd

def add_high_risk_stress_frustration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Growing_Stress ve Mood_Swings eşik değerinin üzerinde olduğu satırları işaretler.
    - Eğer Growing_Stress >= 2 ve Mood_Swings >= 2 ise 1, değilse 0.
    """
    df['High_Stress_Mood_Swings'] = (
        (df['Growing_Stress'] >= 2) & 
        (df['Mood_Swings'] >= 2)
    ).astype(int)
    """
    Days_Indoors ve Quarantine_Frustrations eşik değerinin üzerinde olduğu satırları işaretler.
    - Eğer Days_Indoors >= 2 ve Quarantine_Frustrations >= 2 ise 1, değilse 0.
    """
    df['Days_In_Frustration'] = (
        (df['Days_Indoors'] >= 2) & 
        (df['Quarantine_Frustrations'] >= 2)
    ).astype(int)

    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Yalnızca hedef (Coping_Struggles) kullanmadan feature engineering uygulayan wrapper.
    """
    # High risk bayrağı ekle
    df = add_high_risk_stress_frustration(df)
    return df

if __name__ == '__main__':
    # İşlenmiş CSV'i yükle
    root = Path(__file__).parents[1]
    processed_path = root / 'data' / 'processed' / 'mental_health_encoded.csv'
    df = pd.read_csv(processed_path)

    # Özellikleri ekle
    df_fe = engineer_features(df)

    # Yeni hali kaydet
    out_path = root / 'data' / 'processed' / 'mental_health_fe.csv'
    df_fe.to_csv(out_path, index=False)
    print(f"Feature-engineered data saved to: {out_path}")
