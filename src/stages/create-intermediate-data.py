import argparse
import pickle
import shutil
import sys
import tempfile

from pathlib import Path
from typing import Text
from zipfile import ZipFile

import yaml

from loguru import logger

from ocrpostcorrection.icdar_data import generate_data


def create_intermediate_data(config_path: Text) -> None:
    config = yaml.safe_load(open(config_path))

    logger.remove()
    logger.add(sys.stderr, level=config['base']['loglevel'])

    Path(config['create-intermediate-data']['intermediate-train-md-csv']).parent.mkdir(exist_ok=True, parents=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        with ZipFile(config['create-intermediate-data']['raw-data-zip'], 'r') as zip_object:
            zip_object.extractall(path=tmp_dir)

        # Copy Finnish data
        path = Path(tmp_dir)
        in_dir = path/'TOOLS_for_Finnish_data'
        inputs = {
            'evaluation': 'ICDAR2019_POCR_competition_evaluation_4M_without_Finnish',
            'full': 'ICDAR2019_POCR_competition_full_22M_without_Finnish',
            'training': 'ICDAR2019_POCR_competition_training_18M_without_Finnish',
        }
        for from_dir, to_dir in inputs.items():
            for in_file in (in_dir/'output'/from_dir).iterdir():
                if in_file.is_file:
                    out_file = path/to_dir/'FI'/'FI1'/in_file.name
                    shutil.copy2(in_file, out_file)

        # Get paths for train and test data
        train_path = Path(tmp_dir)/'ICDAR2019_POCR_competition_training_18M_without_Finnish'
        test_path = Path(tmp_dir)/'ICDAR2019_POCR_competition_evaluation_4M_without_Finnish'

        data, md = generate_data(train_path)
        md.to_csv(config['create-intermediate-data']['intermediate-train-md-csv'])
        pickle.dump(data, open(config['create-intermediate-data']['intermediate-train-texts'], 'wb'))
        logger.info('Saved intermediate train data')

        data_test, md_test = generate_data(test_path)
        md_test.to_csv(config['create-intermediate-data']['intermediate-test-md-csv'])
        pickle.dump(data_test, open(config['create-intermediate-data']['intermediate-test-texts'], 'wb'))
        logger.info('Saved intermediate test data')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    create_intermediate_data(args.config)
