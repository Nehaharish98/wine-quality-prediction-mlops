"""Download wine quality dataset."""
import pandas as pd
import requests
from pathlib import Path
import argparse
import yaml

def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from params.yaml."""
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)

def download_wine_data(output_path: str):
    """Download wine quality dataset from UCI repository."""
    red_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    white_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    
    try:
        # Download datasets
        print("Downloading red wine data...")
        red_wine = pd.read_csv(red_wine_url, sep=';')
        
        print("Downloading white wine data...")
        white_wine = pd.read_csv(white_wine_url, sep=';')
        
        # Add wine type column
        red_wine['wine_type'] = 'red'
        white_wine['wine_type'] = 'white'
        
        # Combine datasets
        combined = pd.concat([red_wine, white_wine], ignore_index=True)
        
        # Save to output path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)
        
        print(f"Combined dataset saved to {output_path}")
        print(f"Dataset shape: {combined.shape}")
        print(f"Wine types: {combined['wine_type'].value_counts().to_dict()}")
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="Output path for dataset")
    parser.add_argument("--params", default="params.yaml", help="Parameters file")
    
    args = parser.parse_args()
    
    # Load output path from params if not provided
    if not args.out:
        params = load_params(args.params)
        output_path = params['paths']['data']['raw']
    else:
        output_path = args.out
    
    download_wine_data(output_path)