import argparse
import json
import sys
import tempfile
from typing import Text

import numpy as np
import yaml
from datasets import load_from_disk
from loguru import logger
from ocrpostcorrection.icdar_data import extract_icdar_data, generate_data
from ocrpostcorrection.token_classification import tokenize_and_align_labels
from ocrpostcorrection.utils import (
    predictions2icdar_output,
    predictions_to_labels,
)
from transformers import AutoTokenizer


def convert_predictions_to_icdar_output(config_path: Text) -> None:
    config = yaml.safe_load(open(config_path))

    logger.remove()
    logger.add(sys.stderr, level=config["base"]["loglevel"])

    dataset = load_from_disk(config["create-error-detection-dataset"]["dataset"])

    logger.info(f'Dataset loaded: # samples test: {len(dataset["test"])}')

    model_name = config["train-error-detection"]["pretrained-model-name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = dataset.map(tokenize_and_align_labels(tokenizer), batched=True)

    predictions = np.load(config["predict-test-set-error-detection"]["predictions"])

    logger.info("Converting predictions to icdar output format")
    with tempfile.TemporaryDirectory() as tmp_dir:
        _, test_path = extract_icdar_data(tmp_dir, config["base"]["raw-data-zip"])

        data_test, _ = generate_data(test_path)

        icdar_output = predictions2icdar_output(
            tokenized_dataset["test"],
            predictions_to_labels(predictions),
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

    convert_predictions_to_icdar_output(args.config)
