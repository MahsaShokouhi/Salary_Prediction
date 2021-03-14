#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import os
import yaml
import logging

from sklearn.preprocessing import (StandardScaler,
                                   PolynomialFeatures, 
                                   OneHotEncoder)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import LabelEncoder


def load_config(config_name):
    '''Load the config file containing all the configurations'''
    with open(config_name, 'r') as file:
        config = yaml.safe_load(file)
    return config


def clean_data(df):
    '''Remove duplicates rows and redundant columns'''
    df.drop(['companyId'], axis=1, inplace=True)
    df.drop_duplicates(subset='jobId', inplace=True)  # remove duplicates
    df.drop(['jobId'], axis=1, inplace=True)


def preprocessor(numeric_cols, categorical_cols, interaction_term=True):
    '''Create a preprocessing pipeline'''
    # Process categorical and numerical features separately

    # Define different transformations for categorical and numerical features
    # Add Scaling and PCA (and interaction if required) for numeric features
    numeric_trans = [('scale', StandardScaler())]
    if interaction_term:
        numeric_trans.append(
            ('interaction', PolynomialFeatures(interaction_only=True)))
    numeric_trans.append(('pca', PCA()))  
    numeric_pipe = Pipeline(steps=numeric_trans)
    # One-hot encode categorical features
    categorical_pipe = OneHotEncoder()

    transformer = ColumnTransformer(transformers=[('categorical_preproc',
                                                   categorical_pipe,
                                                   categorical_cols),
                                                  ('numeric_preproc',
                                                   numeric_pipe,
                                                   numeric_cols)],
                                    remainder='passthrough')

    return transformer


# def encode_categoricals(features, le=False):
#     '''Encode Categorical Features using label-encoding or one-hot encoding'''

#     if le:  # label encoding
#         numeric_features = features.select_dtypes(exclude=['object'])
#         categorical_features = features.select_dtypes(include=['object'])
#         encoder = LabelEncoder()
#         categorical_features = categorical_features.apply(
#             encoder.fit_transform)
#         features_encoded = pd.concat(
#             [numeric_features, categorical_features], axis=1)
#     else:  # one-hot encoding
#         features_encoded = pd.get_dummies(features)
#     return features_encoded

def models_validation(x_train, y_train, models, k_cv=5,
                      score='neg_mean_squared_error'):
    '''Evaluate and compare models using cross-validation'''
    mse_means = []
    mse_stdevs = []
    for model in models:
        crossval = cross_val_score(
            model, x_train, y_train, cv=k_cv, scoring=score)        
        mse_means.append(-1.0*crossval.mean())
        mse_stdevs.append(crossval.std())
    return mse_means, mse_stdevs


def create_log(log_file=None):
    '''Create log file for each module'''
    # Get the log file info
    config = load_config('scripts/config.yaml')
    log_dir = config['log_dir']

    # create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # create file handler and set level to debug
    fh = logging.FileHandler(os.path.join(log_dir, log_file), mode="w")
    fh.setLevel(logging.DEBUG)

    # create formatter and add it to the file handler
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    fh.setFormatter(formatter)

    # add the file handler to logger
    logger.addHandler(fh)

    return logger