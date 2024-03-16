import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd
import typer
from jinja2 import Environment, FileSystemLoader
from loguru import logger
from ocrpostcorrection.utils import aggregate_ed_results
from typing_extensions import Annotated

from common.option_types import file_in_option, file_out_option


def add_metrics(
    df: pd.DataFrame, metrics: Dict[str, Union[int, float]], name: str
) -> Dict[str, Union[int, float]]:
    for lang, improvement in df["%ed_improvement"].to_dict().items():
        metrics[f"{name}_correction_{lang}_%ed_impr"] = improvement
    return metrics


def to_markdown_table(df: pd.DataFrame) -> Any:
    return df[["%ed_improvement"]].round(0).transpose().to_markdown()


def generate_error_correction_report(
    train_log: Annotated[Path, file_in_option],
    test_log: Annotated[Path, file_in_option],
    eval_perfect_detection: Annotated[Path, file_in_option],
    eval_predicted_detection: Annotated[Path, file_in_option],
    metrics_file: Annotated[Path, file_out_option],
    seed: Annotated[int, typer.Option()],
    val_size: Annotated[float, typer.Option()],
    max_len: Annotated[int, typer.Option()],
    model_name: Annotated[str, typer.Option()],
    num_epochs: Annotated[int, typer.Option()],
    report_file: Annotated[Path, file_out_option],
) -> None:
    here = Path(__file__).parent
    environment = Environment(
        loader=FileSystemLoader(here / ".." / ".." / "templates/")
    )
    template = environment.get_template("report-error-correction.md")

    train_log = pd.read_csv(train_log, index_col=0)
    try:
        idx_min = train_log.query('stage == "eval"')["loss"].idxmin()
        val_loss = train_log.loc[idx_min].loss
        train_loss = train_log.loc[idx_min - 1].loss
    except ValueError:
        logger.info("Train log does not contain all expected losses. Loss is set to 0.")
        val_loss = 0.0
        train_loss = 0.0

    test_log = pd.read_csv(test_log, index_col=0)
    test_loss = test_log.loc[0].test_loss

    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
    }

    perfect_detection_results = aggregate_ed_results(eval_perfect_detection)
    metrics = add_metrics(perfect_detection_results, metrics, "perfect")
    table_perfect = to_markdown_table(perfect_detection_results)
    predicted_detection_results = aggregate_ed_results(eval_predicted_detection)
    metrics = add_metrics(predicted_detection_results, metrics, "predicted")
    table_predicted = to_markdown_table(predicted_detection_results)

    logger.info(f'Writing metrics "{metrics_file}"')
    (Path(metrics_file).parent).mkdir(exist_ok=True, parents=True)
    with open(metrics_file, mode="w", encoding="utf-8") as f:
        json.dump(metrics, f)

    content = template.render(
        today=date.today(),
        seed=seed,
        val_size=val_size,
        max_len=max_len,
        train_loss=train_loss,
        val_loss=val_loss,
        test_loss=test_loss,
        model_name=model_name,
        num_epochs=num_epochs,
        table_perfect=table_perfect,
        table_predicted=table_predicted,
    )

    logger.info(f'Writing report "{report_file}"')
    with open(report_file, mode="w", encoding="utf-8") as report:
        report.write(content)


if __name__ == "__main__":
    typer.run(generate_error_correction_report)
