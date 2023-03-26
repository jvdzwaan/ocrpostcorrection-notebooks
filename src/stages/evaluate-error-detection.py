import argparse
import json
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Dict, Text

import pandas as pd
import yaml
from datasets import load_from_disk
from loguru import logger
from ocrpostcorrection.icdar_data import extract_icdar_data, generate_data
from ocrpostcorrection.token_classification import tokenize_and_align_labels
from ocrpostcorrection.utils import (
    aggregate_results,
    predictions2icdar_output,
    predictions_to_labels,
    runEvaluation,
)
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)


def save_test_log(metrics: Dict[str, Any], out_file: str) -> None:
    test_log = pd.DataFrame([metrics])
    test_log.columns = [col.replace("test_", "") for col in test_log.columns]

    Path(out_file).parent.mkdir(exist_ok=True, parents=True)
    test_log.to_csv(out_file)


def evaluate_error_detection(config_path: Text) -> None:
    config = yaml.safe_load(open(config_path))

    logger.remove()
    logger.add(sys.stderr, level=config["base"]["loglevel"])

    set_seed(config["base"]["seed"])

    dataset = load_from_disk(config["create-error-detection-dataset"]["dataset"])

    logger.info(f'Dataset loaded: # samples test: {len(dataset["test"])}')

    model_dir = config["train-error-detection"]["output-dir"]
    model_name = config["train-error-detection"]["pretrained-model-name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = dataset.map(tokenize_and_align_labels(tokenizer), batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy=config["train-error-detection"]["evaluation-strategy"],
        num_train_epochs=3,
    )

    num_labels = dataset["train"].features["tags"].feature.num_classes
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    training_args = TrainingArguments(
        output_dir=config["train-error-detection"]["output-dir"],
        evaluation_strategy=config["train-error-detection"]["evaluation-strategy"],
        num_train_epochs=config["train-error-detection"]["num-train-epochs"],
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
    save_test_log(pred.metrics, config["evaluate-error-detection"]["test-log"])

    logger.info("Generating predictions in icdar output format")
    with tempfile.TemporaryDirectory() as tmp_dir:
        _, test_path = extract_icdar_data(tmp_dir, config["base"]["raw-data-zip"])

        data_test, _ = generate_data(test_path)

        icdar_output = predictions2icdar_output(
            tokenized_dataset["test"],
            predictions_to_labels(pred.predictions),
            tokenizer,
            data_test,
        )

        json_file = config["evaluate-error-detection"]["icdar-output-json"]
        with open(json_file, "w") as f:
            json.dump(icdar_output, f)

        logger.info("Running evaluation")

        csv_file = config["evaluate-error-detection"]["icdar-output-csv"]
        runEvaluation(test_path, json_file, csv_file)

        out_file = config["evaluate-error-detection"]["report-file"]
        logger.info(f'Writing report "{out_file}"')

        results = aggregate_results(csv_file)

        train_log = pd.read_csv(
            config["train-error-detection"]["train-log"], index_col=0
        )
        idx_min = train_log.query('stage == "eval"')["loss"].idxmin()
        val_loss = train_log.loc[idx_min].loss
        train_loss = train_log.loc[idx_min - 1].loss

        Path(out_file).parent.mkdir(exist_ok=True, parents=True)
        with open(out_file, "w") as f:
            f.write(f"## Results Error Detection Experiment ({date.today()})\n\n")
            f.write("* [ocrpostcorrection-notebooks commit: ]()\n")
            f.write("* Dataset\n")
            f.write(f"\t* Split seed: {config['base']['seed']} ")
            f.write(f"({config['data-split']['val-size']*100}% for validation)\n")
            f.write("\t* Normalized editdistance threshold for 'sentences': ")
            f.write(f"{config['create-error-detection-dataset']['max-edit-distance']} ")
            f.write("(only for train and val)\n")
            f.write("\t* Sequence (sentence) length: ")
            f.write(f"size: {config['create-error-detection-dataset']['size']}, ")
            f.write(f"step: {config['create-error-detection-dataset']['step']}\n")
            f.write("* Pretrained model: ")
            f.write(f"[{model_name}](https://huggingface.co/{model_name})\n")
            f.write("* Loss\n")
            f.write(f"\t* Train: {train_loss}\n")
            f.write(f"\t* Val: {val_loss}\n")
            f.write(f"\t* Test: {pred.metrics['test_loss']}\n\n")
            f.write(results.round(2).to_markdown())
            f.write("\n\n### Summarized results\n\n")
            f.write(results[["T1_Fmesure"]].round(2).transpose().to_markdown())
            f.write("\n")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    evaluate_error_detection(args.config)
