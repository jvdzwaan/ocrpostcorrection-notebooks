import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Text

import pandas as pd
import yaml
from datasets import load_from_disk
from loguru import logger
from ocrpostcorrection.icdar_data import extract_icdar_data, generate_data
from ocrpostcorrection.token_classification import tokenize_and_align_labels
from ocrpostcorrection.utils import (
    predictions2icdar_output,
    predictions_to_labels,
)
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)


def save_test_log(metrics: Dict[str, Any], out_file: str) -> None:
    test_log = pd.DataFrame([metrics])
    test_log.columns = [col.replace("test_", "") for col in test_log.columns]

    Path(out_file).parent.mkdir(exist_ok=True, parents=True)
    test_log.to_csv(out_file)


def predict_test_set_error_detection(config_path: Text) -> None:
    config = yaml.safe_load(open(config_path))

    logger.remove()
    logger.add(sys.stderr, level=config["base"]["loglevel"])

    set_seed(config["base"]["seed"])

    dataset = load_from_disk(config["create-error-detection-dataset"]["dataset"])

    logger.info(f'Dataset loaded: # samples test: {len(dataset["test"])}')

    model_dir = config["train-error-detection"]["output-dir"]
    model_name = config["train-error-detection"]["pretrained-model-name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = dataset.map(tokenize_and_align_labels(tokenizer), batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy=config["train-error-detection"]["evaluation-strategy"],
        num_train_epochs=3,
    )

    num_labels = dataset["train"].features["tags"].feature.num_classes
    model = AutoModelForTokenClassification.from_pretrained(
        model_dir, num_labels=num_labels
    )

    eval_batch_size = config["predict-test-set-error-detection"]["per-device-eval-batch-size"]
    training_args = TrainingArguments(
        output_dir=config["train-error-detection"]["output-dir"],
        evaluation_strategy=config["train-error-detection"]["evaluation-strategy"],
        num_train_epochs=config["train-error-detection"]["num-train-epochs"],
        per_device_eval_batch_size=eval_batch_size,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    pred = trainer.predict(tokenized_dataset["test"])
    save_test_log(pred.metrics, config["predict-test-set-error-detection"]["test-log"])

    logger.info("Converting predictions to icdar output format")
    with tempfile.TemporaryDirectory() as tmp_dir:
        _, test_path = extract_icdar_data(tmp_dir, config["base"]["raw-data-zip"])

        data_test, _ = generate_data(test_path)

        icdar_output = predictions2icdar_output(
            tokenized_dataset["test"],
            predictions_to_labels(pred.predictions),
            tokenizer,
            data_test,
        )

    json_file = config["predict-test-set-error-detection"]["icdar-output-json"]
    with open(json_file, "w") as f:
        json.dump(icdar_output, f)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    predict_test_set_error_detection(args.config)
