#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import os
from io import StringIO

from utils import load_config, clean_data, create_log


def etl(config_file):
    '''Read datasets, preprocess, and save the processed data'''
    # Load the settings
    config = load_config(config_file)

    # Create a log file for etl
    logger = create_log(log_file=config['etl_log'])

    ###################
    # Load the datasets
    ###################
    logger.info(f'Loading the datasets')

    raw_data_dir = config['data_dir'] + config['original_dir']
    train_features = pd.read_csv(
        os.path.join(raw_data_dir, config['train_features']),
        header=0)
    target = pd.read_csv(os.path.join(raw_data_dir, config['target']))
    test = pd.read_csv(os.path.join(raw_data_dir, config['test']))

    ###################
    # Data preparation
    ###################
    logger.info(f'Data preparation')

    train = pd.merge(train_features, target, how='inner', on='jobId')

    logger.info(f'\nTrain set:\n')
    buf = StringIO()
    train.info(buf=buf)
    logger.info(buf.getvalue())

    clean_data(train)
    clean_data(test)
    train = train[train['salary'] > 0]  # Remove invalid values for salary

    logger.info(f'Train data dimesions: {train.shape}')
    logger.info(f'Test data dimesions: {test.shape}')

    #######################
    # Save clean  data
    #######################
    derived_dir = config['data_dir'] + config['derived_dir']
    logger.info(f'writing the cleaned data to {derived_dir}')

    train.to_csv(
        os.path.join(derived_dir, config['train_derived']),
        index=False)

    test.to_csv(
        os.path.join(derived_dir, config['test_derived']),
        index=False)


if __name__ == '__main__':
    etl('scripts/config.yaml')
