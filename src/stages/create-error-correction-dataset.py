from pathlib import Path

import pandas as pd
import typer
from loguru import logger
from ocrpostcorrection.error_correction import get_tokens_with_OCR_mistakes
from ocrpostcorrection.icdar_data import get_intermediate_data
from typing_extensions import Annotated

from common.option_types import file_in_option, file_out_option


def create_error_correction_dataset(
    raw_dataset: Annotated[Path, file_in_option],
    val_split: Annotated[Path, file_in_option],
    dataset_out: Annotated[Path, file_out_option],
) -> None:
    logger.info("Creating intermediary data")
    data, _, data_test, _ = get_intermediate_data(raw_dataset)

    X_val = pd.read_csv(val_split, index_col=0)

    logger.info("Getting tokens with OCR mistakes")
    tdata = get_tokens_with_OCR_mistakes(data, data_test, list(X_val.file_name))
    tdata.drop_duplicates(subset=["ocr", "gs", "language", "dataset"], inplace=True)
    tdata.reset_index(drop=True, inplace=True)

    counts = tdata.dataset.value_counts()
    logger.info(
        f"Found {tdata.shape[0]} unique samples (train: {counts['train']},"
        + f" val: {counts['val']}, test: {counts['test']})"
    )

    logger.info("Saving dataset")
    out_dir = dataset_out.parent
    out_dir.mkdir(exist_ok=True, parents=True)

    tdata.to_csv(dataset_out)


if __name__ == "__main__":
    typer.run(create_error_correction_dataset)
