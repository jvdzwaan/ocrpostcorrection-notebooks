import json
from datetime import date
from pathlib import Path

import pandas as pd
import typer
from jinja2 import Environment, FileSystemLoader
from loguru import logger
from ocrpostcorrection.utils import aggregate_results
from typing_extensions import Annotated

from common.option_types import file_in_option, file_out_option


def generate_error_detection_report(
    evaluation_csv: Annotated[Path, file_in_option],
    train_log_file: Annotated[Path, file_in_option],
    test_log_file: Annotated[Path, file_in_option],
    seed: Annotated[int, typer.Option()],
    val_size: Annotated[float, typer.Option()],
    max_ed: Annotated[int, typer.Option()],
    size: Annotated[int, typer.Option()],
    step: Annotated[int, typer.Option()],
    model_name: Annotated[str, typer.Option()],
    metrics_file: Annotated[Path, file_out_option],
    report_file: Annotated[Path, file_out_option],
) -> None:
    here = Path(__file__).parent
    environment = Environment(
        loader=FileSystemLoader(here / ".." / ".." / "templates/")
    )
    template = environment.get_template("report-error-detection.md")

    results = aggregate_results(evaluation_csv)

    train_log = pd.read_csv(train_log_file, index_col=0)
    idx_min = train_log.query('stage == "eval"')["loss"].idxmin()
    val_loss = train_log.loc[idx_min].loss
    train_loss = train_log.loc[idx_min - 1].loss

    test_log = pd.read_csv(test_log_file, index_col=0)
    test_loss = test_log.loc[0].loss

    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
    }
    for lang, f1 in results.T1_Fmesure.to_dict().items():
        metrics[f"{lang}_F1"] = f1

    logger.info(f'Writing metrics "{metrics_file}"')
    with open(metrics_file, mode="w", encoding="utf-8") as f:
        json.dump(metrics, f)

    content = template.render(
        today=date.today(),
        seed=seed,
        val_size=val_size,
        max_edit_distance=max_ed,
        size=size,
        step=step,
        model_name=model_name,
        train_loss=train_loss,
        val_loss=val_loss,
        test_loss=test_loss,
        results_table=results.round(2).to_markdown(),
        summarized_results=results[["T1_Fmesure"]].round(2).transpose().to_markdown(),
    )

    logger.info(f'Writing report "{report_file}"')
    with open(report_file, mode="w", encoding="utf-8") as report:
        report.write(content)


if __name__ == "__main__":
    typer.run(generate_error_detection_report)
