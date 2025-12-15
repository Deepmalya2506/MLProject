import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file=os.path.join('artifacts','model.pkl') # store the trained model objects in the artifacts folder named model.pkl

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig() # creating an object of the config class to access it
    
    def initiate_training(self,transformed_train_x,transformed_test_x,train_y,test_y,preprocessor_path):
        try:
            logging.info("Testing with different models...")
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report=evaluate_model(transformed_train_x,train_y,transformed_test_x,test_y,models) 
            # evaluate function is defined in the utils as a helper function 
            # The function returns a report of the performance of the list of models for test_data

            # To get best_score model form the report
            best_model_score:float=max(sorted(model_report.values()))

            # To get best model form the report dict
            best_model_name:str=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]

            if(best_model_score<0.6):
                logging.error("No Best Model found")
            else:
                logging.info("Best Model Found")

                save_object(file_path=self.model_trainer_config,obj= best_model)

                predicted=best_model.predict(transformed_test_x)
                r2score=r2_score(predicted,test_y)

                logging.info(f"Best Model accuray score came out to be(on test data):{r2score}")

                return r2score

        except Exception as e:
            raise CustomException(e,sys)
