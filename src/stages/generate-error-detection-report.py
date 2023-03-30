import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Text

from jinja2 import Environment, FileSystemLoader
import pandas as pd
import yaml
from loguru import logger
from ocrpostcorrection.utils import aggregate_results


def generate_error_detection_report(config_path: Text) -> None:
    config = yaml.safe_load(open(config_path))

    logger.remove()
    logger.add(sys.stderr, level=config["base"]["loglevel"])

    here = Path(__file__).parent
    environment = Environment(loader=FileSystemLoader(here/'..'/'..'/"templates/"))
    template = environment.get_template("report-error-detection.md")

    csv_file = config["evaluate-error-detection"]["icdar-output-csv"]
    results = aggregate_results(csv_file)

    train_log = pd.read_csv(
        config["train-error-detection"]["train-log"], index_col=0
    )
    idx_min = train_log.query('stage == "eval"')["loss"].idxmin()
    val_loss = train_log.loc[idx_min].loss
    train_loss = train_log.loc[idx_min - 1].loss

    test_log = pd.read_csv(
        config["predict-test-set-error-detection"]["test-log"], index_col=0
    )
    test_loss = test_log.loc[0].loss

    content = template.render(
        today=date.today(),
        seed=config['base']['seed'],
        val_size=config['data-split']['val-size'],
        max_edit_distance=config['create-error-detection-dataset']['max-edit-distance'],
        size=config['create-error-detection-dataset']['size'],
        step=config['create-error-detection-dataset']['step'],
        model_name=config["train-error-detection"]["pretrained-model-name"],
        train_loss=train_loss,
        val_loss=val_loss,
        test_loss=test_loss,
        results_table=results.round(2).to_markdown(),
        summarized_results=results[["T1_Fmesure"]].round(2).transpose().to_markdown(),
    )

    out_file = config["generate-error-detection-report"]["report-file"]
    logger.info(f'Writing report "{out_file}"')
    with open(out_file, mode="w", encoding="utf-8") as report:
        report.write(content)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    generate_error_detection_report(args.config)
