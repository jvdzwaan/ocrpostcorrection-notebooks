import json
import tempfile
from pathlib import Path
from typing import Dict, Optional, Text

import edlib
import pandas as pd
import typer
from datasets import Dataset
from loguru import logger
from ocrpostcorrection.error_correction import get_context_for_dataset
from ocrpostcorrection.error_correction_t5 import preprocess_function
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
    test: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    logger.info("Creating predictions csv")

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
    trainer,
    tokenizer,
    out_file: Text,
    include_language: bool,
    context_offset: int,
    marker: str,
    dev: bool = False,
) -> None:
    if in_file and out_file:
        logger.info(f"Generating predictions for '{in_file}'")
        with open(in_file) as f:
            output = json.load(f)
        test = icdar_output2simple_correction_dataset_df(output, data_test)
        logger.info(f"Number of samples in the dataset: {test.shape[0]}")

        if context_offset > 0:
            logger.info(f"Adding context of ~{context_offset} characters")
            context_data = get_context_for_dataset(
                data_test,
                test,
                context_offset,
            )
            test = test.merge(context_data)

        test = test.query(f"len_ocr <= {max_len}").query(f"len_gs <= {max_len}").copy()
        test = test.sort_values(by=["len_ocr"], ascending=False)
        logger.info(
            f"Number of samples after filtering on ocr and gs length: {test.shape[0]}"
        )
        if dev:
            test = test.head(5)

        dataset = Dataset.from_pandas(test)
        tokenized_dataset = dataset.map(
            preprocess_function,
            fn_kwargs={
                "tokenizer": tokenizer,
                "add_task_prefix": include_language,
                "context_marker": marker,
            },
            batched=True,
        )

        ocr_in = tokenizer.decode(tokenized_dataset[0]["input_ids"])
        logger.info(f"Input for first sample: {ocr_in}")

        pred = trainer.predict(tokenized_dataset)
        predictions = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)

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
    seed: Annotated[int, typer.Option()],
    batch_size: Annotated[int, typer.Option()],
    perfect_detection_in: Annotated[Text, typer.Option()] = "",
    perfect_detection_out: Annotated[Text, typer.Option()] = "",
    predicted_detection_in: Annotated[Text, typer.Option()] = "",
    predicted_detection_out: Annotated[Text, typer.Option()] = "",
    include_language: Annotated[
        Optional[bool], typer.Option("--include-language/--exclude-language")
    ] = False,
    context_offset: Annotated[int, typer.Option()] = 0,
    marker: Annotated[str, typer.Option()] = "",
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
        per_device_eval_batch_size=batch_size,
        generation_max_length=max_len,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    predict_and_save(
        perfect_detection_in,
        data_test,
        max_len,
        trainer,
        tokenizer,
        perfect_detection_out,
        include_language,
        context_offset,
        marker,
        dev,
    )

    predict_and_save(
        predicted_detection_in,
        data_test,
        max_len,
        trainer,
        tokenizer,
        predicted_detection_out,
        include_language,
        context_offset,
        marker,
        dev,
    )


if __name__ == "__main__":
    typer.run(predict_error_correction)
