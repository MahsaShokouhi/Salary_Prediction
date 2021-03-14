#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import os
import pickle
from sklearn.externals import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from utils import load_config, create_log, preprocessor
import shap


def train(config_file):
    '''Train best model (with lowest mse) on train set'''
    # Load the settings
    config = load_config(config_file)
    
    # Create a log file for model training
    logger = create_log(log_file = config['train_log'])
    
    #####################
    # Load cleaned training data
    #####################  
    logger.info(f'Load clean data')
    
    derived_dir = config['data_dir'] + config['derived_dir']
    train = pd.read_csv(os.path.join(derived_dir, config['train_derived']))

    x_train = train.drop(['salary'], axis=1)
    y_train = train['salary']

    ################
    # Model training
    ################ 
    logger.info(f'Train the selected model')
    
    # Preprocessing step
    
    # Extract numerical and categorical columns from x_train
    features_num = x_train.select_dtypes(exclude='object')
    features_cat = x_train.select_dtypes(include='object')
    num_cols = features_num.columns
    cat_cols = features_cat.columns
    prepoc = preprocessor(num_cols, cat_cols, interaction_term=False)
    
    # Train the selected model (random forest)
    model = Pipeline(steps=[
        ('Preprocess', prepoc),
        ('model', RandomForestRegressor(n_estimators=200, max_depth=15, max_features=10))])    

    model.fit(x_train, y_train)
    
     ###################
    # Add SHAP explainer
    ################### 
    
    def model_predict(data_asarray):
    data_asframe =  pd.DataFrame(data_asarray, columns=test.columns)
    return model.predict(data_asframe)

    # shap.initjs()
    X_train_summary = shap.sample(X_train, 10)
    explainer = shap.KernelExplainer(model_predict, X_train_summary)
    
    
    ################
    # Save the model
    ################ 
    model_dir = config['model_dir']
    model_trained = os.path.join(model_dir, config['model_trained'])
    model_parameters = os.path.join(model_dir, config['model_parameters'])
    
    logger.info(f'writing the trained model to {model_dir}')
                
    pickle.dump(model, open(model_trained, 'wb'))

    with open(model_parameters, 'w') as file:
        file.write(str(model))

    # Save explainer
    ex_file = os.path.join(model_dir,'explainer.bz2')
    joblib.dump(explainer, filename=ex_file, compress=('bz2', 9))
    
if __name__ == '__main__':
    train('scripts/config.yaml')
    