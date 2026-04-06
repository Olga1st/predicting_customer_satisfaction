import json
from datetime import datetime
from pathlib import Path
from turtle import mode

TRACKING_PATH = Path("models/runs.json")


def log_experiment(model_name: str, metrics: dict, params: dict,mode: str, use_tuning: bool):
    run = {
        "timestamp": str(datetime.now()),
        "model": model_name,
        "metrics": metrics,
        "params": params,
        "mode": mode,
        "use_tuning": use_tuning
    }

    TRACKING_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(TRACKING_PATH, "a") as f:
        f.write(json.dumps(run) + "\n")

    print("✅ Experiment logged.")