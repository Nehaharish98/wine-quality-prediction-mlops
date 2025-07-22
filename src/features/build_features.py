from sklearn.model_selection import train_test_split

def build_features(df: pd.DataFrame):
    df["wine_type"] = df["wine_type"].map({"red": 0, "white": 1})
    X = df.drop("quality", axis=1)
    y = df["quality"]
    return train_test_split(X, y, test_size=0.2, random_state=42)