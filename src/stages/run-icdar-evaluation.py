import tempfile
from pathlib import Path

import typer
from loguru import logger
from ocrpostcorrection.icdar_data import extract_icdar_data
from ocrpostcorrection.utils import runEvaluation
from typing_extensions import Annotated

from common.option_types import file_in_option, file_out_option


def run_icdar_evaluation(
    raw_dataset: Annotated[Path, file_in_option],
    json_file: Annotated[Path, file_in_option],
    csv_file: Annotated[Path, file_out_option],
) -> None:
    logger.info("Running evaluation")
    with tempfile.TemporaryDirectory() as tmp_dir:
        _, test_path = extract_icdar_data(tmp_dir, raw_dataset)

        runEvaluation(test_path, json_file, csv_file)


if __name__ == "__main__":
    typer.run(run_icdar_evaluation)
