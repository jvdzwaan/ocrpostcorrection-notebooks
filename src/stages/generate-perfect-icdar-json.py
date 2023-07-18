import json
import tempfile
from pathlib import Path
from typing import Text

import typer
from loguru import logger
from ocrpostcorrection.icdar_data import extract_icdar_data, generate_data
from ocrpostcorrection.utils import create_perfect_icdar_output
from typing_extensions import Annotated


def generate_perfect_detection_icdar_json(
    raw_dataset: Annotated[Text, typer.Option()],
    out_file: Annotated[Text, typer.Option()],
) -> None:
    logger.info("Reading test data from raw dataset")
    with tempfile.TemporaryDirectory() as tmp_dir:
        _train_path, test_path = extract_icdar_data(tmp_dir, raw_dataset)
        logger.debug(f"Test data is in {test_path}")

        data_test, _md_test = generate_data(test_path)
        logger.info(f"Found {len(data_test)} texts in test data")

    logger.info("Extracting perfect detection results from test data")
    output = create_perfect_icdar_output(data_test)

    logger.info(f"Saving ICDAR json file to '{out_file}'")
    (Path(out_file).parent).mkdir(exist_ok=True, parents=True)
    with open(out_file, "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    typer.run(generate_perfect_detection_icdar_json)
