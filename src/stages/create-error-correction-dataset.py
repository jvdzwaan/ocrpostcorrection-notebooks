import argparse
import sys
from pathlib import Path
from typing import Text

import pandas as pd
import yaml
from loguru import logger
from ocrpostcorrection.error_correction import get_tokens_with_OCR_mistakes
from ocrpostcorrection.icdar_data import get_intermediate_data


def create_error_correction_dataset(config_path: Text) -> None:
    config = yaml.safe_load(open(config_path))

    logger.remove()
    logger.add(sys.stderr, level=config["base"]["loglevel"])

    logger.info("Creating intermediary data")
    data, _, data_test, _ = get_intermediate_data(config["base"]["raw-data-zip"])

    X_val = pd.read_csv(config["data-split"]["val-split"], index_col=0)

    logger.info("Getting tokens with OCR mistakes")
    tdata = get_tokens_with_OCR_mistakes(data, data_test, list(X_val.file_name))
    tdata.drop_duplicates(subset=["ocr", "gs", "dataset"], inplace=True)
    tdata.reset_index(drop=True, inplace=True)

    counts = tdata.dataset.value_counts()
    logger.info(
        f"Found {tdata.shape[0]} unique samples (train: {counts['train']},"
        + f" val: {counts['val']}, test: {counts['test']})"
    )

    logger.info("Saving dataset")
    out_file = Path(config["create-error-correction-dataset"]["dataset"])
    out_dir = out_file.parent
    out_dir.mkdir(exist_ok=True, parents=True)

    tdata.to_csv(out_file)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    create_error_correction_dataset(args.config)
