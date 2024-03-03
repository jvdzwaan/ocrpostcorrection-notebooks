from pathlib import Path

import pandas as pd
import typer
from datasets import Dataset, DatasetDict
from loguru import logger
from ocrpostcorrection.error_correction_t5 import filter_max_len, preprocess_function
from ocrpostcorrection.utils import reduce_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from typing_extensions import Annotated

from common.common import remove_checkpoints, save_train_log, set_seed
from common.option_types import dir_out_option, file_in_option, file_out_option


def train_error_correction(
    seed: Annotated[int, typer.Option()],
    dataset: Annotated[Path, file_in_option],
    max_len: Annotated[int, typer.Option()],
    model_name: Annotated[str, typer.Option()],
    batch_size: Annotated[int, typer.Option()],
    num_epochs: Annotated[int, typer.Option()],
    valid_niter: Annotated[int, typer.Option()],
    model_dir: Annotated[Path, dir_out_option],
    train_log: Annotated[Path, file_out_option],
    delete_checkpoints: Annotated[bool, typer.Option()],
    dev: Annotated[bool, typer.Option()] = False,
) -> None:
    set_seed(seed)

    data = pd.read_csv(dataset, index_col=0)
    data.fillna("", inplace=True)

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(data.query('dataset == "train"')),
            "val": Dataset.from_pandas(data.query('dataset == "val"')),
        }
    )

    if dev:
        dataset = reduce_dataset(dataset)

    dataset = dataset.filter(
        filter_max_len, fn_kwargs={"max_len": max_len}, batched=False
    )

    logger.info(f"# train samples: {len(dataset['train'])}")
    logger.info(f"# val samples: {len(dataset['val'])}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_dataset = dataset.map(
        preprocess_function, fn_kwargs={"tokenizer": tokenizer}, batched=True
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    training_args = Seq2SeqTrainingArguments(
        seed=seed,
        output_dir=model_dir,
        evaluation_strategy="steps",
        eval_steps=valid_niter,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=3,
        num_train_epochs=num_epochs,
        predict_with_generate=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        save_strategy="steps",
        save_steps=valid_niter,
        adafactor=True,
        learning_rate=1e-3,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()

    if delete_checkpoints:
        remove_checkpoints(model_dir)

    save_train_log(train_log_data=trainer.state.log_history, train_log_file=train_log)


if __name__ == "__main__":
    typer.run(train_error_correction)
