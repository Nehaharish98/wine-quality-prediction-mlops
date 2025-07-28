"""Preprocess wine quality data."""
import pandas as pd
import argparse
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path: str, output_path: str):
    """Preprocess wine quality data."""
    df = pd.read_csv(input_path)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.dropna()
    
    # Feature scaling for numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns = [col for col in numeric_columns if col != 'quality']
    
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    # Save processed data
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path", required=True)
    args = parser.parse_args()
    preprocess_data(args.input_path, args.output_path)
