from src.utils.config_loader import load_config, load_params
from src.features.build_features import load_and_merge_data
from src.models.train_evaluate import train_models

def main():
    config = load_config()
    params = load_params()

    df = load_and_merge_data(
        config["data"]["raw_data_red"],
        config["data"]["raw_data_white"]
    )

    train_models(df, config, params)

if __name__ == "__main__":
    main()