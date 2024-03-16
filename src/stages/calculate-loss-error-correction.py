from pathlib import Path
from typing import Text, Optional

import pandas as pd
import typer
from datasets import Dataset
from loguru import logger
from ocrpostcorrection.error_correction_t5 import filter_max_len, preprocess_function
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from typing_extensions import Annotated

from common.option_types import file_in_option


def calculate_loss_error_correction(
    error_correction_dataset: Annotated[Path, file_in_option],
    max_len: Annotated[int, typer.Option()],
    model_name: Annotated[Text, typer.Option()],
    seed: Annotated[int, typer.Option()],
    batch_size: Annotated[int, typer.Option()],
    test_log_file: Annotated[Path, typer.Option()],
    include_language: Annotated[
        Optional[bool], typer.Option("--include-language/--exclude-language")
    ] = False,
    dev: Annotated[bool, typer.Option()] = False,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    training_args = Seq2SeqTrainingArguments(
        seed=seed,
        output_dir=model_name,
        predict_with_generate=True,
        per_device_eval_batch_size=batch_size,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

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
        preprocess_function,
        fn_kwargs={"tokenizer": tokenizer, "add_task_prefix": include_language},
        batched=True,
    )

    pred = trainer.predict(tokenized_dataset)

    logs = pd.DataFrame([pred.metrics])
    (test_log_file.parent).mkdir(exist_ok=True, parents=True)
    logs.to_csv(test_log_file)


if __name__ == "__main__":
    typer.run(calculate_loss_error_correction)
