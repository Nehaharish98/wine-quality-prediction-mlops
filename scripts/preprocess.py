"""Preprocess wine quality data."""
import pandas as pd
import argparse
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from params.yaml."""
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_data(input_path: str, output_path: str, params: dict):
    """Preprocess wine quality data."""
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"Original dataset shape: {df.shape}")
    
    # Remove duplicates
    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape}")
    
    # Handle missing values
    df = df.dropna()
    print(f"After removing null values: {df.shape}")
    
    # Feature engineering
    feature_columns = [col for col in df.columns 
                      if col not in [params['base']['target_col'], 'wine_type']]
    
    # Feature scaling for numeric columns
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    # Split data
    target_col = params['base']['target_col']
    test_size = params['data']['test_size']
    random_state = params['base']['random_state']
    
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df[target_col]
    )
    
    # Save processed data
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(output_path, index=False)
    
    # Save test set separately
    test_output_path = output_path.replace('train.parquet', 'test.parquet')
    test_df.to_parquet(test_output_path, index=False)
    
    print(f"Train data saved to {output_path}, shape: {train_df.shape}")
    print(f"Test data saved to {test_output_path}, shape: {test_df.shape}")
    
    # Save scaler for later use
    import joblib
    scaler_path = Path(output_path).parent / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", help="Input data path")
    parser.add_argument("--out", dest="output_path", help="Output data path")
    parser.add_argument("--params", default="params.yaml", help="Parameters file")
    
    args = parser.parse_args()
    
    # Load parameters
    params = load_params(args.params)
    
    # Use paths from params if not provided
    input_path = args.input_path or params['paths']['data']['raw']
    output_path = args.output_path or params['paths']['data']['processed']
    
    preprocess_data(input_path, output_path, params)