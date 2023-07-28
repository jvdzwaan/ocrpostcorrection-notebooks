from pathlib import Path

import pandas as pd
import typer
from datasets import ClassLabel, Dataset, DatasetDict, Sequence
from loguru import logger
from ocrpostcorrection.icdar_data import generate_sentences, get_intermediate_data
from typing_extensions import Annotated

from common.option_types import dir_out_option, file_in_option


def create_error_detection_dataset(
    raw_dataset: Annotated[Path, file_in_option],
    train_split: Annotated[Path, file_in_option],
    val_split: Annotated[Path, file_in_option],
    test_split: Annotated[Path, file_in_option],
    size: Annotated[int, typer.Option()],
    step: Annotated[int, typer.Option()],
    max_ed: Annotated[float, typer.Option()],
    dataset_out: Annotated[Path, dir_out_option],
) -> None:
    data, _, data_test, _ = get_intermediate_data(raw_dataset)

    X_train = pd.read_csv(train_split, index_col=0)
    X_val = pd.read_csv(val_split, index_col=0)
    X_test = pd.read_csv(test_split, index_col=0)

    logger.info(f"Generating sentences (size: {size}, step: {step})")

    train_data = generate_sentences(X_train, data, size=size, step=step)
    val_data = generate_sentences(X_val, data, size=size, step=step)
    test_data = generate_sentences(X_test, data_test, size=size, step=size)

    num_train = train_data.shape[0]
    num_val = val_data.shape[0]
    num_test = test_data.shape[0]
    logger.info(f"# samples train: {num_train}, val: {num_val}, test: {num_test})")

    logger.info(f"Filtering train and val based on maximum edit distance of {max_ed}")
    train_data = train_data[train_data.score < max_ed]
    val_data = val_data[val_data.score < max_ed]

    for df in (train_data, val_data, test_data):
        df.drop(columns=["score"], inplace=True)

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_data),
            "val": Dataset.from_pandas(val_data),
            "test": Dataset.from_pandas(test_data),
        }
    )

    logger.info("Adding class labels")
    dataset = dataset.cast_column(
        "tags",
        Sequence(
            feature=ClassLabel(
                num_classes=3, names=["O", "OCR-Mistake-B", "OCR-Mistake-I"]
            ),
            length=-1,
        ),
    )

    logger.info("Saving dataset")
    dataset.save_to_disk(dataset_out)
    num_train = len(dataset["train"])
    num_val = len(dataset["val"])
    num_test = len(dataset["test"])
    logger.info(f"# samples train: {num_train}, val: {num_val}, test: {num_test})")


if __name__ == "__main__":
    typer.run(create_error_detection_dataset)
