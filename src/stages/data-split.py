from pathlib import Path

import typer
from loguru import logger
from ocrpostcorrection.icdar_data import get_intermediate_data
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated

from common.option_types import file_in_option, file_out_option


def data_split(
    raw_dataset: Annotated[Path, file_in_option],
    train_split: Annotated[Path, file_out_option],
    val_split: Annotated[Path, file_out_option],
    test_split: Annotated[Path, file_out_option],
    val_size: Annotated[float, typer.Option()],
    seed: Annotated[int, typer.Option()],
) -> None:
    train_split.parent.mkdir(exist_ok=True, parents=True)

    _, md, _, md_test = get_intermediate_data(raw_dataset)

    X_train, X_val, _, _ = train_test_split(
        md,
        md["file_name"],
        test_size=val_size,
        random_state=seed,
        stratify=md["subset"],
    )

    X_train.to_csv(train_split)
    X_val.to_csv(val_split)

    logger.info(f"Train set contains {X_train.shape[0]} texts")
    logger.info(f"Val set contains {X_val.shape[0]} texts")

    md_test.to_csv(test_split)

    logger.info(f"Test set contains {md_test.shape[0]} texts")


if __name__ == "__main__":
    typer.run(data_split)
