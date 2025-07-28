#!/usr/bin/env python
import yaml, logging, sys
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.features.build_features import load_data, preprocess
from src.models.train_evaluate import train
from src.utils.mlflow_logger import log_run, setup as ml_setup

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    cfg = yaml.safe_load(Path("config/config.yaml").read_text())
    if not ml_setup(cfg["mlflow"]):
        sys.exit("MLflow not reachable")

    red, white = load_data(**cfg["data"])
    prob = cfg["problem"]["type"]

    for wine, df in [("red", red), ("white", white)]:
        X, y, _ = preprocess(df, problem_type=prob)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42,
                                              stratify=y if prob!="regression" else None)
        for algo in ["rf", "xgb"]:
            model, params = train(algo, Xtr, ytr, prob)
            run_name = f"{wine.capitalize()}Wine-{algo.upper()}"
            log_run(model, run_name, Xte, yte, params, X.columns.tolist(), cfg["mlflow"])
            log.info("✅ finished %s", run_name)

if __name__ == "__main__":
    main()