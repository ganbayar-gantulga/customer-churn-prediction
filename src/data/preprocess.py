"""
Өгөгдөл цэвэрлэх, хувиргах модуль
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Түүхий өгөгдлийг уншина."""
    df = pd.read_csv(filepath)
    print(f"Өгөгдлийн хэмжээ: {df.shape[0]} мөр, {df.shape[1]} багана")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Өгөгдлийг цэвэрлэнэ:
    - Хоосон утгуудыг арилгах
    - Өгөгдлийн төрлийг засах
    - Давхардлыг арилгах
    """
    df = df.copy()

    # TotalCharges баганыг тоон төрөлд хөрвүүлэх
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Хоосон утгуудыг median-аар бөглөх
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # customerID баганыг хасах (шинж чанар биш)
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Давхардсан мөрүүдийг арилгах
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"  {duplicates} давхардсан мөр арилгагдлаа")
        df.drop_duplicates(inplace=True)

    print(f"Цэвэрлэсний дараа: {df.shape[0]} мөр, {df.shape[1]} багана")
    return df


def encode_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Categorical шинж чанаруудыг тоон утгад хувиргана.

    Returns:
        df: Хувиргасан DataFrame
        encoders: LabelEncoder-үүдийн dict (дараа нь decode хийхэд хэрэглэнэ)
    """
    df = df.copy()
    encoders = {}

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"  '{col}' -> {len(le.classes_)} ангилал")

    return df, encoders


def scale_features(
    df: pd.DataFrame, target_col: str = "Churn"
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Тоон шинж чанаруудыг StandardScaler-аар масштаблана.
    """
    df = df.copy()
    scaler = StandardScaler()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target_col]

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, scaler


def split_data(
    df: pd.DataFrame,
    target_col: str = "Churn",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Өгөгдлийг train/test болгон хуваана.
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print(f"Churn rate - Train: {y_train.mean():.2%} | Test: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test


def run_preprocessing_pipeline(filepath: str) -> dict:
    """
    Бүх preprocessing алхмуудыг дараалуулан ажиллуулна.
    """
    print("=" * 50)
    print("PREPROCESSING PIPELINE")
    print("=" * 50)

    # 1. Өгөгдөл унших
    print("\n[1/5] Өгөгдөл уншиж байна...")
    df = load_raw_data(filepath)

    # 2. Цэвэрлэх
    print("\n[2/5] Өгөгдөл цэвэрлэж байна...")
    df = clean_data(df)

    # 3. Encode
    print("\n[3/5] Categorical шинж чанаруудыг хувиргаж байна...")
    df, encoders = encode_features(df)

    # 4. Scale
    print("\n[4/5] Тоон шинж чанаруудыг масштаблаж байна...")
    df, scaler = scale_features(df)

    # 5. Хуваах
    print("\n[5/5] Train/Test хуваалт хийж байна...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("\n" + "=" * 50)
    print("PREPROCESSING ДУУСЛАА!")
    print("=" * 50)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "encoders": encoders,
        "scaler": scaler,
    }


if __name__ == "__main__":
    result = run_preprocessing_pipeline("data/raw/telco_churn.csv")
