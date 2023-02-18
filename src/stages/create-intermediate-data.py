import argparse
import pickle

from pathlib import Path
from typing import Text

import yaml

from loguru import logger

from ocrpostcorrection.icdar_data import generate_data


def create_intermediate_data(config_path: Text) -> None:
    config = yaml.safe_load(open(config_path))

    Path(config['create-intermediate-data']['intermediate-train-md-csv']).parent.mkdir(exist_ok=True, parents=True)

    data, md = generate_data(Path(config['create-intermediate-data']['raw-data-dir-train']))
    md.to_csv(config['create-intermediate-data']['intermediate-train-md-csv'])
    pickle.dump(data, open(config['create-intermediate-data']['intermediate-train-texts'], 'wb'))
    logger.info('Saved intermediate train data')

    data_test, md_test = generate_data(Path(config['create-intermediate-data']['raw-data-dir-test']))
    md_test.to_csv(config['create-intermediate-data']['intermediate-test-md-csv'])
    pickle.dump(data_test, open(config['create-intermediate-data']['intermediate-test-texts'], 'wb'))
    logger.info('Saved intermediate test data')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    create_intermediate_data(args.config)
