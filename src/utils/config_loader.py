import yaml

def load_config(config_path="config/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_params(params_path="config/params.yaml"):
    with open(params_path) as f:
        return yaml.safe_load(f)