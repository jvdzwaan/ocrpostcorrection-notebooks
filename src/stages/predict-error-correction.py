import json
import tempfile
from pathlib import Path
from typing import Dict, Text

import edlib
import pandas as pd
import typer
from datasets import Dataset
from loguru import logger
from ocrpostcorrection.error_correction_t5 import filter_max_len, preprocess_function
from ocrpostcorrection.icdar_data import Text as ICDARText
from ocrpostcorrection.icdar_data import (
    extract_icdar_data,
    generate_data,
    normalized_ed,
)
from ocrpostcorrection.utils import icdar_output2simple_correction_dataset_df
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from typing_extensions import Annotated

from common.option_types import file_in_option


def create_predictions_csv(
    test: pd.DataFrame, max_len: int, predictions: pd.DataFrame, dev: bool = False
) -> pd.DataFrame:
    test_results = (
        test.query(f"len_ocr <= {max_len}").query(f"len_gs <= {max_len}").copy()
    )
    if dev:
        test_results = test_results[:5]
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
    trainer,
    tokenizer,
    out_file: Text,
    dev: bool = False,
) -> None:
    logger.info(f"Generating predictions for '{in_file}'")
    if in_file and out_file:
        with open(in_file) as f:
            output = json.load(f)
        test = icdar_output2simple_correction_dataset_df(output, data_test)

        dataset = Dataset.from_pandas(test)
        if dev:
            dataset = dataset.select(range(5))
        tokenized_dataset = dataset.map(
            preprocess_function, fn_kwargs={"tokenizer": tokenizer}, batched=True
        )

        pred = trainer.predict(tokenized_dataset)
        predictions = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)

        results = create_predictions_csv(test, max_len, predictions, dev)
        results.to_csv(out_file)
        logger.info(f"Saved predictions to '{out_file}'")
    elif in_file or out_file:
        logger.info(
            "To generate predictions for an icdar json file, please specify both --*_in"
            + " and --*_out. (Got '{in_file}' (--*_in) and '{out_file}' (--*_out))"
        )


def predict_error_correction(
    error_correction_dataset: Annotated[Path, file_in_option],
    max_len: Annotated[int, typer.Option()],
    model_name: Annotated[Text, typer.Option()],
    calculate_loss: Annotated[bool, typer.Option()],
    test_log_file: Annotated[Path, typer.Option()],
    raw_dataset: Annotated[Path, file_in_option],
    seed: Annotated[int, typer.Option()],
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
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    training_args = Seq2SeqTrainingArguments(
        seed=seed,
        output_dir=model_name,
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if calculate_loss:
        logger.info("Calculating loss on dataset without duplicates")
        data = pd.read_csv(error_correction_dataset, index_col=0)
        data.fillna("", inplace=True)

        dataset = Dataset.from_pandas(data.query('dataset == "test"'))

        dataset = dataset.filter(
            filter_max_len, fn_kwargs={"max_len": max_len}, batched=False
        )

        if dev:
            dataset = dataset.select(range(5))

        tokenized_dataset = dataset.map(
            preprocess_function, fn_kwargs={"tokenizer": tokenizer}, batched=True
        )

        pred = trainer.predict(tokenized_dataset)

        logs = pd.DataFrame([pred.metrics])
        (test_log_file.parent).mkdir(exist_ok=True, parents=True)
        logs.to_csv(test_log_file)

    predict_and_save(
        perfect_detection_in,
        data_test,
        max_len,
        trainer,
        tokenizer,
        perfect_detection_out,
        dev,
    )

    predict_and_save(
        predicted_detection_in,
        data_test,
        max_len,
        trainer,
        tokenizer,
        predicted_detection_out,
        dev,
    )


if __name__ == "__main__":
    typer.run(predict_error_correction)
