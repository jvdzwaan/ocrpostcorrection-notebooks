from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from loguru import logger
from ocrpostcorrection.error_correction import (
    get_OCR_mistakes_in_context,
    get_tokens_with_OCR_mistakes,
)
from ocrpostcorrection.icdar_data import get_intermediate_data
from typing_extensions import Annotated

from common.option_types import file_in_option, file_out_option


def create_error_correction_dataset(
    raw_dataset: Annotated[Path, file_in_option],
    val_split: Annotated[Path, file_in_option],
    dataset_out: Annotated[Path, file_out_option],
    context_offset: Annotated[int, typer.Option()],
    include_language: Annotated[
        Optional[bool], typer.Option("--include-language/--exclude-language")
    ] = False,
) -> None:
    logger.info("Creating intermediary data")
    data, _, data_test, _ = get_intermediate_data(raw_dataset)

    X_val = pd.read_csv(val_split, index_col=0)

    logger.info("Getting tokens with OCR mistakes")
    tdata = get_tokens_with_OCR_mistakes(data, data_test, list(X_val.file_name))
    logger.info(f"Found {tdata.shape[0]} samples.")
    if context_offset > 0:
        logger.info(f"Adding context of ~{context_offset} characters")
        cdata = get_OCR_mistakes_in_context(
            data,
            data_test,
            tdata,
            context_offset,
        )
        tdata = tdata.merge(cdata)

    subset = ["ocr", "gs", "dataset"]
    if include_language:
        subset.append("language")
    if context_offset > 0:
        subset.append("context_before")
        subset.append("context_after")
    logger.info(f"Subset for identyfying duplicate samples: {subset}")

    tdata.drop_duplicates(subset=subset, inplace=True)
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
