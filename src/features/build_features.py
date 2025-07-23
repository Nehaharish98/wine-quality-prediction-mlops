import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_data(red_path, white_path):
    red = pd.read_csv(red_path, delimiter=';')
    white = pd.read_csv(white_path, delimiter=';')
    return red, white

def initial_inspection(df, name="Dataset"):
    #Print summary of the DataFrame.
    print(f"--- {name} INFO ---")
    print(df.info())
    print(df.describe())
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("-" * 40)

def clean_data(df):
    
    #Clean the data: 
    #- Drop duplicate rows
    #- Fill missing values (mean imputation for numerics, mode for categoricals
    df_clean = df.drop_duplicates().copy()
    
    # Numeric columns: fill missing with mean
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    
    # Categorical columns: fill missing with mode (if any)
    cat_cols = df_clean.select_dtypes(include=[object]).columns
    for col in cat_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0])
    return df_clean

def preprocess(df, quality_threshold=6):
    if 'quality' not in df.columns:
        print("ERROR: 'quality' column not found in DataFrame. Columns are:", df.columns.tolist())
        raise KeyError("'quality' column missing from DataFrame!")

    X = df.drop('quality', axis=1)
    y = (df['quality'] >= quality_threshold).astype(int)  # 1 if good wine, 0 if not

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Balance classes with SMOTE (works only for classification)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)
    return X_res, y_res, scaler

if __name__ == "__main__":
    # Adjust these paths as needed
    red_path = "data/raw/wine-quality-red.csv"
    white_path = "data/raw/wine-quality-white.csv"
    
    red_df, white_df = load_data(red_path, white_path)
    
    initial_inspection(red_df, "Red Wine")
    initial_inspection(white_df, "White Wine")
    
    red_df = clean_data(red_df)
    white_df = clean_data(white_df)
    
    X_red, y_red, scaler_red = preprocess(red_df)
    X_white, y_white, scaler_white = preprocess(white_df)

    print(f"Red: X shape: {X_red.shape}, y counts: {np.bincount(y_red)}")
    print(f"White: X shape: {X_white.shape}, y counts: {np.bincount(y_white)}")