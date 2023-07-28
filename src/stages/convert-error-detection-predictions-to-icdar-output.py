import json
import tempfile
from pathlib import Path

import numpy as np
import typer
from datasets import load_from_disk
from loguru import logger
from ocrpostcorrection.icdar_data import extract_icdar_data, generate_data
from ocrpostcorrection.token_classification import tokenize_and_align_labels
from ocrpostcorrection.utils import predictions2icdar_output, predictions_to_labels
from transformers import AutoTokenizer
from typing_extensions import Annotated

from common.option_types import dir_in_option, file_in_option, file_out_option


def convert_error_detection_predictions_to_icdar_output(
    dataset_in: Annotated[Path, dir_in_option],
    model_name: Annotated[str, typer.Option()],
    predictions_in: Annotated[Path, file_in_option],
    raw_dataset: Annotated[Path, file_in_option],
    json_out: Annotated[Path, file_out_option],
) -> None:
    dataset = load_from_disk(dataset_in)

    logger.info(f'Dataset loaded: # samples test: {len(dataset["test"])}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = dataset.map(tokenize_and_align_labels(tokenizer), batched=True)

    predictions = np.load(predictions_in)

    logger.info("Converting predictions to icdar output format")
    with tempfile.TemporaryDirectory() as tmp_dir:
        _, test_path = extract_icdar_data(tmp_dir, raw_dataset)

        data_test, _ = generate_data(test_path)

        icdar_output = predictions2icdar_output(
            tokenized_dataset["test"],
            predictions_to_labels(predictions),
            tokenizer,
            data_test,
        )

    with open(json_out, "w") as f:
        json.dump(icdar_output, f)


if __name__ == "__main__":
    typer.run(convert_error_detection_predictions_to_icdar_output)
