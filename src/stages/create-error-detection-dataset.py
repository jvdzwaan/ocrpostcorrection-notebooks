import argparse
import json
import sys

from pathlib import Path
from typing import Text

import pandas as pd
import yaml

from datasets import Sequence, ClassLabel, DatasetDict, Dataset
from loguru import logger

from ocrpostcorrection.icdar_data import generate_sentences, get_intermediate_data


def create_error_detection_dataset(config_path: Text) -> None:
    config = yaml.safe_load(open(config_path))

    logger.remove()
    logger.add(sys.stderr, level=config['base']['loglevel'])

    data, _, data_test, _ = get_intermediate_data(config['base']['raw-data-zip'])

    X_train = pd.read_csv(config['data-split']['train-split'], index_col=0)
    X_val = pd.read_csv(config['data-split']['val-split'], index_col=0)
    X_test = pd.read_csv(config['data-split']['test-split'], index_col=0)

    size = config['create-error-detection-dataset']['size']
    step = config['create-error-detection-dataset']['step']

    logger.info(f'Generating sentences (size: {size}, step: {step})')

    train_data = generate_sentences(X_train, data, size=size, step=step)
    val_data = generate_sentences(X_val, data, size=size, step=step)
    test_data = generate_sentences(X_test, data_test, size=size, step=size)

    logger.info(f'# samples train: {train_data.shape[0]}, val: {val_data.shape[0]}, test: {test_data.shape[0]})')

    max_ed = config['create-error-detection-dataset']['max-edit-distance']
    logger.info(f"Filtering train and val based on maximum edit distance of {max_ed}")
    train_data = train_data[train_data.score < max_ed]
    val_data = val_data[val_data.score < max_ed]

    for df in (train_data, val_data, test_data):
        df.drop(columns=['score'], inplace=True)

    dataset = DatasetDict(
        {
            'train': Dataset.from_pandas(train_data),
            'val': Dataset.from_pandas(val_data),
            'test': Dataset.from_pandas(test_data),
        }
    )

    logger.info('Adding class labels')
    dataset = dataset.cast_column(
        'tags',
        Sequence(feature=ClassLabel(num_classes=3, names=['O', 'OCR-Mistake-B', 'OCR-Mistake-I']), length=-1)
    )

    logger.info('Saving dataset')
    dataset.save_to_disk(config['create-error-detection-dataset']['dataset'])
    logger.info(f'# samples train: {len(dataset["train"])}, val: {len(dataset["val"])}, test: {len(dataset["test"])})')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    create_error_detection_dataset(args.config)
