import argparse
import shutil
import sys

from pathlib import Path
from typing import Text

import pandas as pd
import yaml

from loguru import logger
from sklearn.model_selection import train_test_split

from ocrpostcorrection.icdar_data import get_intermediate_data


def data_split(config_path: Text) -> None:
    config = yaml.safe_load(open(config_path))

    logger.remove()
    logger.add(sys.stderr, level=config['base']['loglevel'])

    Path(config['data-split']['train-split']).parent.mkdir(exist_ok=True, parents=True)

    _, md, _, md_test = get_intermediate_data(config['base']['raw-data-zip'])

    X_train, X_val, _, _ = train_test_split(
        md,
        md['file_name'],
        test_size=config['data-split']['val-size'],
        random_state=config['base']['seed'],
        stratify=md['language']
    )

    X_train.to_csv(config['data-split']['train-split'])
    X_val.to_csv(config['data-split']['val-split'])

    logger.info(f'Train set contains {X_train.shape[0]} texts')
    logger.info(f'Val set contains {X_val.shape[0]} texts')

    md_test.to_csv(config['data-split']['test-split'])

    logger.info(f'Test set contains {md_test.shape[0]} texts')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_split(args.config)
