import json
import tempfile
from pathlib import Path
from typing import Dict, Text

import pandas as pd
import typer
from loguru import logger
from ocrpostcorrection.icdar_data import extract_icdar_data, generate_data
from ocrpostcorrection.utils import (
    create_perfect_icdar_output,
    icdar_output2simple_correction_dataset_df,
)
from typing_extensions import Annotated


def save_json(obj: Dict[str, Dict[str, float]], out_file: Text) -> None:
    logger.info(f"Saving ICDAR json file to '{out_file}'")
    (Path(out_file).parent).mkdir(exist_ok=True, parents=True)
    with open(out_file, "w") as f:
        json.dump(obj, f)


def generate_perfect_icdar_json(
    raw_dataset: Annotated[Text, typer.Option()],
    out_file_detection: Annotated[Text, typer.Option()],
    out_file_correction: Annotated[Text, typer.Option()],
) -> None:
    logger.info("Reading test data from raw dataset")
    with tempfile.TemporaryDirectory() as tmp_dir:
        _train_path, test_path = extract_icdar_data(tmp_dir, raw_dataset)
        logger.debug(f"Test data is in {test_path}")

        data_test, _md_test = generate_data(test_path)
        logger.info(f"Found {len(data_test)} texts in test data")

    logger.info("Extracting perfect detection results from test data")
    output = create_perfect_icdar_output(data_test)

    save_json(output, out_file_detection)

    logger.info("Extracting perfect correction results from test data")
    task2_input_df = icdar_output2simple_correction_dataset_df(output, data_test)
    task2_output = task2_input_df.copy().set_index("text")
    task2_output["pred"] = task2_output["gs"]

    for key, mistakes in output.items():
        samples = task2_output.loc[key]
        # If there is only 1 erronous token, samples is not a DataFrame but a Series
        if isinstance(samples, pd.DataFrame):
            for token, (i, row) in zip(mistakes, samples.iterrows()):
                output[key][token][row.gs] = 1.0
        else:
            token = list(mistakes.keys())[0]
            output[key][token][samples.gs] = 1.0

    save_json(output, out_file_correction)


if __name__ == "__main__":
    typer.run(generate_perfect_icdar_json)
