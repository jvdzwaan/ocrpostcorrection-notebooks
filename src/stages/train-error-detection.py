import shutil
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import typer
from datasets import load_from_disk
from loguru import logger
from ocrpostcorrection.token_classification import tokenize_and_align_labels
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
from typing_extensions import Annotated

from common.option_types import dir_in_option, dir_out_option, file_out_option


def add_values(
    prefix: str, row: Dict[str, Any], name: str, value: Any
) -> Dict[str, Any]:
    new_name = name.replace(prefix, "")
    row[new_name] = value
    row["stage"] = prefix.replace("_", "")
    return row


def create_train_log(logs: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for log in logs:
        row: Dict[str, Any] = {}
        for name, value in log.items():
            if name.startswith("eval_"):
                row = add_values("eval_", row, name, value)
            elif name.startswith("train_"):
                row = add_values("train_", row, name, value)
            else:
                row[name] = value
        rows.append(row)

    log_data = pd.DataFrame(rows)
    log_data = log_data[
        [
            "stage",
            "epoch",
            "step",
            "loss",
            "runtime",
            "samples_per_second",
            "steps_per_second",
            "total_flos",
        ]
    ]
    return log_data


def train_error_detection(
    seed: Annotated[int, typer.Option()],
    dataset_in: Annotated[Path, dir_in_option],
    model_name: Annotated[str, typer.Option()],
    evaluation_strategy: Annotated[str, typer.Option()],
    per_device_train_batch_size: Annotated[int, typer.Option()],
    num_train_epochs: Annotated[int, typer.Option()],
    load_best_model_at_end: Annotated[bool, typer.Option()],
    save_strategy: Annotated[str, typer.Option()],
    delete_checkpoints: Annotated[bool, typer.Option()],
    model_dir: Annotated[Path, dir_out_option],
    train_log: Annotated[Path, file_out_option],
) -> None:
    set_seed(seed)

    dataset = load_from_disk(dataset_in)

    num_train = len(dataset["train"])
    num_val = len(dataset["val"])
    logger.info(f"Dataset loaded: # samples train: {num_train}, val: {num_val}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = dataset.map(tokenize_and_align_labels(tokenizer), batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy=evaluation_strategy,
        num_train_epochs=num_train_epochs,
        load_best_model_at_end=load_best_model_at_end,
        save_strategy=save_strategy,
        per_device_train_batch_size=per_device_train_batch_size,
    )

    num_labels = dataset["train"].features["tags"].feature.num_classes
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model()

    if delete_checkpoints:
        logger.info("Removing checkpoints")
        for path in Path(model_dir).glob("checkpoint-*"):
            if path.is_dir():
                shutil.rmtree(path)

    Path(train_log).parent.mkdir(exist_ok=True, parents=True)
    log_data = create_train_log(trainer.state.log_history)
    log_data.to_csv(train_log)


if __name__ == "__main__":
    typer.run(train_error_detection)
