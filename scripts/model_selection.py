#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import os

from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from utils import load_config, create_log, preprocessor, models_validation



def model_selection(config_file):
    '''Build and compare various models'''
    # Load the settings
    config = load_config(config_file)
        
    # Create a log file for model selection
    logger = create_log(log_file = config['model_selection_log'])
    
    ############################
    # Read cleaned training data
    ############################    
    logger.info(f'Load clean data')
    
    derived_dir = config['data_dir'] + config['derived_dir']
    train = pd.read_csv(os.path.join(derived_dir, config['train_derived']))

    x_train = train.drop(['salary'], axis=1)
    y_train = train['salary']
    

    ###################
    # Model validation
    ###################
    logger.info(f'Build and compare models')
    # Define and compare linear and non-linear models
    

    # Extract numerical and categorical columns from x_train
    features_num = x_train.select_dtypes(exclude='object')
    features_cat = x_train.select_dtypes(include='object')

    num_cols = features_num.columns
    cat_cols = features_cat.columns
    
    prepoc = preprocessor(num_cols, cat_cols, interaction_term=False)
    prepoc_interac = preprocessor(num_cols, cat_cols, interaction_term=True)
    
    # Baseline model
    baseline = Pipeline(steps=[
    ('Preprocess', prepoc),
    ('model', DummyRegressor(strategy="mean"))])
    
    # Linear models
    lr = Pipeline(steps=[
        ('Preprocess', prepoc),
        ('model', LinearRegression())])
    lr_interaction = Pipeline(steps=[
        ('Preprocess', prepoc_interac),
        ('model', LinearRegression())])
    
    # Random Forest
    rf = Pipeline(steps=[
        ('Preprocess', prepoc),
        ('model', RandomForestRegressor(n_estimators=200, max_depth=15, max_features=10))])

    # Evaluate and compare models
    models = list([baseline, lr, lr_interaction, rf])
    model_names = list(['baseline', 
                        'linear regression',
                        'inear regression with interaction', 
                        'random forest'])
    
    mse_means, mse_stdevs = models_validation(x_train, y_train, models)
    for i in range(len(models)):
        logger.info(f'Model: {model_names[i]}')
        logger.info(f'Mean Squared Error = {mse_means[i]:.2f} , ' 
                    f'Standard deviation = {mse_stdevs[i]:.2f}')


if __name__ == '__main__':
    model_selection('scripts/config.yaml')
    