import pandas as pd

def load_and_merge_data(red_path, white_path):
    red = pd.read_csv(red_path)
    white = pd.read_csv(white_path)

    red['type'] = 0
    white['type'] = 1

    df = pd.concat([red, white], axis=0)
    return df