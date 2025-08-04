"""
Preprocess wine quality data.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Utility: load params.yaml
def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from params.yaml."""
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


# Main preprocessing function
def preprocess_data(input_path: str, output_path: str, params: dict) -> None:
    """Preprocess wine quality data and write parquet + scaler."""
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)

    # Basic cleaning
    print(f"Original dataset shape: {df.shape}")
    df = df.drop_duplicates()
    df = df.dropna()
    print(f"After dedup & drop-na:  {df.shape}")

    # Remove the non-numeric 'wine_type' column completely
    if "wine_type" in df.columns:
        df = df.drop(columns=["wine_type"])
        print("Removed column: wine_type")

    # Feature scaling (numeric columns only, target excluded)
    target_col = params["base"]["target_col"]
    feature_columns = [col for col in df.columns if col != target_col]

    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    # Train / test split
    test_size = params["data"]["test_size"]
    random_state = params["base"]["random_state"]

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col],
    )

    # Save outputs
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(output_path, index=False)
    test_df.to_parquet(
        output_path.replace("train.parquet", "test.parquet"), index=False
    )

    # Save scaler for inference
    scaler_path = Path(output_path).parent / "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    print(f"Train saved to {output_path}  → shape {train_df.shape}")
    print(
        f"Test  saved to {output_path.replace('train', 'test')}  → shape {test_df.shape}"
    )
    print(f"Scaler saved to {scaler_path}")
# CLI wrapper
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", help="Input CSV path")
    parser.add_argument("--out", dest="output_path", help="Output parquet path")
    parser.add_argument("--params", default="params.yaml", help="Params file path")
    args = parser.parse_args()

    # Load params and resolve paths
    params = load_params(args.params)
    input_path = args.input_path or params["paths"]["data"]["raw"]
    output_path = args.output_path or params["paths"]["data"]["processed"]

    preprocess_data(input_path, output_path, params)
