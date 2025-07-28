"""Download wine quality dataset."""
import pandas as pd
import requests
from pathlib import Path
import argparse

def download_wine_data(output_path: str):
    """Download wine quality dataset from UCI repository."""
    red_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    white_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    
    # Download datasets
    red_wine = pd.read_csv(red_wine_url, sep=';')
    white_wine = pd.read_csv(white_wine_url, sep=';')
    
    # Add wine type column
    red_wine['wine_type'] = 'red'
    white_wine['wine_type'] = 'white'
    
    # Combine datasets
    combined = pd.concat([red_wine, white_wine], ignore_index=True)
    
    # Save to output path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/wine.csv")
    args = parser.parse_args()
    download_wine_data(args.out)