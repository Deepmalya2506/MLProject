# Helper functions that we can treate as utilities ..these are common functions which we write here to promote reusability

import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.exception import CustomException

from sklearn.metrics import r2_score

import dill

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as f:
            dill.dump(obj, f)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train_data,y_train_data,x_test_data,y_test_data,models):
    
    try:
        report={}

        for i in range(len(list(models))):
            model=list(models.values())[i] # from the dictionary of models with their objects first select one model, then take its value i.e object
            model.fit(X_train_data,y_train_data)

            y_train_pred=model.predict(X_train_data) # prediction during training
            y_test_pred=model.predict(x_test_data)   # prediction during testing

            train_model_score=r2_score(y_train_pred,y_train_data) # accuracy during training
            test_model_score=r2_score(y_test_pred,y_test_data)    # accuraxy with test data

            report[list(models.keys()[i])]=test_model_score

            return report

    except Exception as e:
        raise CustomException(e,sys)
