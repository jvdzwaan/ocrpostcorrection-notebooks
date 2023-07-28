from pathlib import Path
from typing import Any, Dict

import numpy as np
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

from common.option_types import dir_in_option, file_out_option


def save_test_log(metrics: Dict[str, Any], out_file: Path) -> None:
    test_log = pd.DataFrame([metrics])
    test_log.columns = [col.replace("test_", "") for col in test_log.columns]

    Path(out_file).parent.mkdir(exist_ok=True, parents=True)
    test_log.to_csv(out_file)


def predict_test_set_error_detection(
    seed: Annotated[int, typer.Option()],
    dataset_in: Annotated[Path, dir_in_option],
    model_dir: Annotated[Path, dir_in_option],
    model_name: Annotated[str, typer.Option()],
    evaluation_strategy: Annotated[str, typer.Option()],
    eval_batch_size: Annotated[int, typer.Option()],
    num_train_epochs: Annotated[int, typer.Option()],
    predictions_out: Annotated[Path, file_out_option],
    test_log: Annotated[Path, file_out_option],
) -> None:
    set_seed(seed)

    dataset = load_from_disk(dataset_in)

    logger.info(f'Dataset loaded: # samples test: {len(dataset["test"])}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = dataset.map(tokenize_and_align_labels(tokenizer), batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy=evaluation_strategy,
        num_train_epochs=3,
    )

    num_labels = dataset["train"].features["tags"].feature.num_classes
    model = AutoModelForTokenClassification.from_pretrained(
        model_dir, num_labels=num_labels
    )

    training_args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy=evaluation_strategy,
        num_train_epochs=num_train_epochs,
        per_device_eval_batch_size=eval_batch_size,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    pred = trainer.predict(tokenized_dataset["test"])
    np.save(predictions_out, pred.predictions)
    save_test_log(pred.metrics, test_log)


if __name__ == "__main__":
    typer.run(predict_test_set_error_detection)
