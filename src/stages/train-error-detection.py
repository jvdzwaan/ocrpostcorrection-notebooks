import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Text

import pandas as pd
import yaml
from datasets import load_from_disk
from loguru import logger
from ocrpostcorrection.token_classification import tokenize_and_align_labels
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)


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


def train_error_detection(config_path: Text) -> None:
    config = yaml.safe_load(open(config_path))

    logger.remove()
    logger.add(sys.stderr, level=config["base"]["loglevel"])

    set_seed(config["base"]["seed"])

    dataset = load_from_disk(config["create-error-detection-dataset"]["dataset"])

    num_train = len(dataset["train"])
    num_val = len(dataset["val"])
    logger.info(f"Dataset loaded: # samples train: {num_train}, val: {num_val}")

    model_name = config["train-error-detection"]["pretrained-model-name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = dataset.map(tokenize_and_align_labels(tokenizer), batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=config["train-error-detection"]["output-dir"],
        evaluation_strategy=config["train-error-detection"]["evaluation-strategy"],
        num_train_epochs=config["train-error-detection"]["num-train-epochs"],
        load_best_model_at_end=config["train-error-detection"][
            "load-best-model-at-end"
        ],
        save_strategy=config["train-error-detection"]["save-strategy"],
        per_device_train_batch_size=config["train-error-detection"][
            "per-device-train-batch-size"
        ],
    )

    num_labels = dataset["train"].features["tags"].feature.num_classes
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model()

    if config["train-error-detection"]["delete-checkpoints"]:
        logger.info("Removing checkpoints")
        for path in Path(config["train-error-detection"]["output-dir"]).glob(
            "checkpoint-*"
        ):
            if path.is_dir():
                shutil.rmtree(path)

    log_file = config["train-error-detection"]["train-log"]
    Path(log_file).parent.mkdir(exist_ok=True, parents=True)
    log_data = create_train_log(trainer.state.log_history)
    log_data.to_csv(log_file)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    train_error_detection(args.config)
