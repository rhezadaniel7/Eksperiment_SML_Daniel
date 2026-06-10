"""
Automated preprocessing script untuk Telco Customer Churn.
Usage: python automate_Daniel.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import sys


def preprocess(input_path: str, output_dir: str) -> None:
    """Load raw Telco churn CSV, preprocess, dan simpan train/test split."""
    print(f"[INFO] Loading data from {input_path}")
    df = pd.read_csv(input_path)
    print(f"[INFO] Raw shape: {df.shape}")

    # 1. Drop customerID
    df = df.drop(columns=['customerID'])

    # 2. Convert TotalCharges ke numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # 3. Drop duplikat
    df = df.drop_duplicates()

    # 4. Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # 5. One-hot encode kategorikal
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 6. Split
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. Scaling numerik
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # 8. Save
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)

    train_combined = X_train.copy()
    train_combined['Churn'] = y_train.values
    train_combined.to_csv(f'{output_dir}/train_processed.csv', index=False)

    test_combined = X_test.copy()
    test_combined['Churn'] = y_test.values
    test_combined.to_csv(f'{output_dir}/test_processed.csv', index=False)

    print(f"[INFO] Preprocessed data saved to {output_dir}/")
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")


if __name__ == '__main__':
    input_path = sys.argv[1] if len(sys.argv) > 1 else '../telco_raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'telco_preprocessing'
    preprocess(input_path, output_dir)
