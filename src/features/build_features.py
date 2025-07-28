import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(red_path: str, white_path: str):
    red = pd.read_csv(red_path, sep=";")
    white = pd.read_csv(white_path, sep=";")
    red["wine_type"]   = "red"
    white["wine_type"] = "white"
    return red, white

def preprocess(df: pd.DataFrame, target="quality", problem_type="regression"):
    features = [c for c in df.columns if c not in (target, "wine_type")]
    X = df[features].fillna(df[features].median())
    y = df[target]

    # Optional re-label for classification
    if problem_type == "binary_classification":
        y = (y >= 7).astype(int)
    elif problem_type == "classification":
        y = y.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=features), y, scaler