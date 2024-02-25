import json
import tempfile
from itertools import chain
from pathlib import Path
from typing import Dict, Text

import edlib
import pandas as pd
import torch
import typer
from datasets import Dataset
from loguru import logger
from ocrpostcorrection.icdar_data import Text as ICDARText
from ocrpostcorrection.icdar_data import (
    extract_icdar_data,
    generate_data,
    normalized_ed,
)
from ocrpostcorrection.utils import icdar_output2simple_correction_dataset_df
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from typing_extensions import Annotated

from common.option_types import file_in_option


def create_predictions_csv(
    test: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    test_results = test.copy()
    test_results["pred"] = predictions

    test_results["ed"] = test_results.apply(
        lambda row: edlib.align(row.ocr, row.gs)["editDistance"], axis=1
    )
    test_results["ed_norm"] = test_results.apply(
        lambda row: normalized_ed(row.ed, row.ocr, row.gs), axis=1
    )

    test_results["ed_pred"] = test_results.apply(
        lambda row: edlib.align(row.pred, row.gs)["editDistance"], axis=1
    )
    test_results["ed_norm_pred"] = test_results.apply(
        lambda row: normalized_ed(row.ed_pred, row.pred, row.gs), axis=1
    )

    return test_results


def predict_and_save(
    in_file: Text,
    data_test: Dict[str, ICDARText],
    max_len: int,
    pipe,
    batch_size: int,
    out_file: Text,
    dev: bool = False,
) -> None:
    if in_file and out_file:
        logger.info(f"Generating predictions for '{in_file}'")
        with open(in_file) as f:
            output = json.load(f)
        test = icdar_output2simple_correction_dataset_df(output, data_test)

        test = test.query(f"len_ocr <= {max_len}").query(f"len_gs <= {max_len}").copy()
        if dev:
            test = test.head(5)
        dataset = Dataset.from_pandas(test)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        predictions_data = []
        for batch in tqdm(dataloader):
            r = pipe.predict(batch["ocr"])
            predictions_data.append(r)

        predictions = [p["generated_text"] for p in chain(*predictions_data)]
        results = create_predictions_csv(test, predictions)
        results.to_csv(out_file)
        logger.info(f"Saved predictions to '{out_file}'")
    elif in_file or out_file:
        logger.info(
            "To generate predictions for an icdar json file, please specify both --*_in"
            + " and --*_out. (Got '{in_file}' (--*_in) and '{out_file}' (--*_out))"
        )


def predict_error_correction(
    max_len: Annotated[int, typer.Option()],
    model_name: Annotated[Text, typer.Option()],
    raw_dataset: Annotated[Path, file_in_option],
    batch_size: Annotated[int, typer.Option()],
    perfect_detection_in: Annotated[Text, typer.Option()] = "",
    perfect_detection_out: Annotated[Text, typer.Option()] = "",
    predicted_detection_in: Annotated[Text, typer.Option()] = "",
    predicted_detection_out: Annotated[Text, typer.Option()] = "",
    dev: Annotated[bool, typer.Option()] = False,
) -> None:
    logger.info("Loading the test dataset")
    with tempfile.TemporaryDirectory() as tmp_dir:
        _train_path, test_path = extract_icdar_data(tmp_dir, raw_dataset)
        logger.debug(f"Test data is in {test_path}")

        data_test, _md_test = generate_data(test_path)
        logger.info(f"Found {len(data_test)} texts in test data")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline(
        model=model, tokenizer=tokenizer, task="text2text-generation", device_map="auto"
    )

    predict_and_save(
        perfect_detection_in,
        data_test,
        max_len,
        pipe,
        batch_size,
        perfect_detection_out,
        dev,
    )

    predict_and_save(
        predicted_detection_in,
        data_test,
        max_len,
        pipe,
        batch_size,
        predicted_detection_out,
        dev,
    )


if __name__ == "__main__":
    typer.run(predict_error_correction)
