# Helper functions that we can treate as utilities ..these are common functions which we write here to promote reusability

import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.exception import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

import dill

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as f:
            dill.dump(obj, f)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train_data,y_train_data,x_test_data,y_test_data,models,param):
    
    try:
        report={}

        for i in range(len(list(models))): # returns a key-value pair of items form the dictionary 

            model=list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train_data,y_train_data)

            model.set_params(**gs.best_params_)
            model.fit(X_train_data,y_train_data)

            # performance prediction on testing data 
            y_test_pred = model.predict(x_test_data)  

            # test performance evaluation
            test_model_score = r2_score(y_test_data, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e,sys)
