import random
import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from loguru import logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)  # For use with torch.device("mps") (on Mac)


def remove_checkpoints(model_dir: str) -> None:
    logger.info("Removing checkpoints")
    for path in Path(model_dir).glob("checkpoint-*"):
        if path.is_dir():
            shutil.rmtree(path)


def add_values(
    prefix: str, row: Dict[str, Any], name: str, value: Any
) -> Dict[str, Any]:
    new_name = name.replace(prefix, "")
    row[new_name] = value
    row["stage"] = prefix.replace("_", "")
    return row


def create_train_log(logs: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for log in logs:
        row: Dict[str, Any] = {}
        for name, value in log.items():
            if name.startswith("eval_"):
                row = add_values("eval_", row, name, value)
            elif name.startswith("train_"):
                row = add_values("train_", row, name, value)
            else:
                row[name] = value
        rows.append(row)

    log_data = pd.DataFrame(rows)
    log_data = log_data[
        [
            "stage",
            "epoch",
            "step",
            "loss",
            "runtime",
            "samples_per_second",
            "steps_per_second",
            "total_flos",
        ]
    ]
    return log_data


def save_train_log(train_log_data: List[Dict[str, Any]], train_log_file: str) -> None:
    Path(train_log_file).parent.mkdir(exist_ok=True, parents=True)
    log_data = create_train_log(train_log_data)
    log_data.to_csv(train_log_file)
