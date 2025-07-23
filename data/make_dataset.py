import pandas as pd
from pathlib import Path
import os

def load_wine_data():
    red = pd.read_csv(Path("data/raw/wine-quality-red.csv"), sep=';')
    white = pd.read_csv(Path("data/raw/wine-quality-white.csv"), sep=';')
    red["wine_type"] = "red"
    white["wine_type"] = "white"
    df = pd.concat([red, white], ignore_index=True)
    return df


#def save_dataset(df: pd.DataFrame):
   # df.to_csv("data/processed/clean_wine.csv", index=False)

def save_dataset(df: pd.DataFrame):
    os.makedirs("data/processed", exist_ok=True)  # <-- Ensures directory exists
    df.to_csv("data/processed/clean_wine.csv", index=False)    

if __name__ == '__main__':
    df = load_wine_data()
    save_dataset(df)