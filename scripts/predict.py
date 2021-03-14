#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import os
import pickle

from utils import load_config, create_log


def predict(config_file):
    '''Make salary predicttion on test set'''
    # Load the settings
    config = load_config(config_file)

    # Create a log file for model training
    logger = create_log(log_file = config['predict_log'])

    ############################################
    # Load the test data and trained model
    ############################################ 
    logger.info(f'Load the test data and trained model')
    
    derived_dir = config['data_dir'] + config['derived_dir']
    test = pd.read_csv(os.path.join(derived_dir, config['test_derived']))

    model_dir = config['model_dir']
    model_trained = os.path.join(model_dir, config['model_trained'])
    model = pickle.load(open(model_trained, 'rb'))

    ######################################
    # Predict and save predictions
    ###################################### 
    prediction_file = config['predictions']
    logger.info(f'Predict and save predictions as {prediction_file}')
    
    predictions = model.predict(test)
    np.savetxt(prediction_file, predictions, delimiter=',')                

                
if __name__ == '__main__':
    predict('scripts/config.yaml')

