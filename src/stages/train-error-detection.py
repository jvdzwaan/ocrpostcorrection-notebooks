import argparse
import sys
from typing import Any, Text

import pandas as pd
import yaml
from datasets import load_from_disk
from loguru import logger
from transformers import AutoTokenizer
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    set_seed,
    Trainer,
    TrainingArguments,
)

from ocrpostcorrection.token_classification import tokenize_and_align_labels
from ocrpostcorrection.utils import reduce_dataset


def add_values(prefix: str, row: dict[str, Any], name: str, value: Any) -> dict[str, Any]:
    new_name = name.replace(prefix, '')
    row[new_name] = value
    row['stage'] = prefix.replace('_', '')
    return row


def create_train_log(logs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for log in logs:
        row: dict[str, Any] = {}
        for name, value in log.items():
            if name.startswith('eval_'):
                row = add_values('eval_', row, name, value)
            elif name.startswith('train_'):
                row = add_values('train_', row, name, value)
            else:
                row[name] = value
        rows.append(row)

    log_data = pd.DataFrame(rows)
    log_data = log_data[['stage', 'epoch', 'step', 'loss', 'runtime',
                         'samples_per_second', 'steps_per_second', 'total_flos']]
    return log_data


def train_error_detection(config_path: Text) -> None:
    config = yaml.safe_load(open(config_path))

    logger.remove()
    logger.add(sys.stderr, level=config['base']['loglevel'])

    set_seed(config['base']['seed'])

    dataset = load_from_disk(config['create-error-detection-dataset']['dataset'])
    dataset = reduce_dataset(dataset)

    logger.info(f'Dataset loaded: # samples train: {len(dataset["train"])}, val: {len(dataset["val"])}')

    model_name = config['train-error-detection']['pretrained-model-name']

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = dataset.map(tokenize_and_align_labels(tokenizer), batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=config['train-error-detection']['output_dir'],
        evaluation_strategy=config['train-error-detection']['evaluation_strategy'],
        num_train_epochs=config['train-error-detection']['num_train_epochs'],
        load_best_model_at_end=config['train-error-detection']['load_best_model_at_end'],
        save_strategy=config['train-error-detection']['save_strategy'],
        per_device_train_batch_size=config['train-error-detection']['per_device_train_batch_size']
    )

    num_labels = dataset['train'].features['tags'].feature.num_classes
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['val'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model()

    log_data = create_train_log(trainer.state.log_history)
    log_data.to_csv(config['train-error-detection']['train_log'])


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train_error_detection(args.config)
