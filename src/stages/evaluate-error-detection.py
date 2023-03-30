import argparse
import sys
import tempfile
from typing import Text

import yaml
from loguru import logger
from ocrpostcorrection.icdar_data import extract_icdar_data
from ocrpostcorrection.utils import runEvaluation


def evaluate_error_detection(config_path: Text) -> None:
    config = yaml.safe_load(open(config_path))

    logger.remove()
    logger.add(sys.stderr, level=config["base"]["loglevel"])

    logger.info("Running evaluation")
    with tempfile.TemporaryDirectory() as tmp_dir:
        _, test_path = extract_icdar_data(tmp_dir, config["base"]["raw-data-zip"])
        json_file = config["predict-test-set-error-detection"]["icdar-output-json"]
        csv_file = config["evaluate-error-detection"]["icdar-output-csv"]

        runEvaluation(test_path, json_file, csv_file)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    evaluate_error_detection(args.config)
