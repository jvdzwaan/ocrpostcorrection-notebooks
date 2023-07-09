import sys
import tempfile
from typing import Text

import typer
from loguru import logger
from ocrpostcorrection.icdar_data import extract_icdar_data
from ocrpostcorrection.utils import runEvaluation


def run_icdar_evaluation(
    raw_data_zip: Text, json_file: Text, csv_file: Text, loglevel: str
) -> None:
    logger.remove()
    logger.add(sys.stderr, level=loglevel)

    logger.info("Running evaluation")
    with tempfile.TemporaryDirectory() as tmp_dir:
        _, test_path = extract_icdar_data(tmp_dir, raw_data_zip)

        runEvaluation(test_path, json_file, csv_file)


if __name__ == "__main__":
    typer.run(run_icdar_evaluation)
