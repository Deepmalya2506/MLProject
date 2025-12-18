import os
import sys
from dataclasses import dataclass

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
    
    def initiate_training(self, transformed_train_x, transformed_test_x, train_y, test_y):
        try:
            logging.info("Testing with different models...")

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(verbosity=0),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # For hyperparameter tuning we are uisng GridSearchCV that not only finds the best model, but also selects th best parameter for optimization
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            # Evaluate all models
            model_report:dict = evaluate_model(
                transformed_train_x, train_y,
                transformed_test_x, test_y,
                models,
                param=params
            )

            # Best model score
            best_model_score = max(model_report.values())

            # Best model name
            best_model_name = next(model_name for model_name,model_score in model_report.items() if model_score==best_model_score)

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.error("No Best Model found")
                return None

            logging.info(f"Best Model Found: {best_model_name} with score {best_model_score}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file,
                obj=best_model
            )

            # Evaluate best model on test data
            predicted = best_model.predict(transformed_test_x)
            final_r2 = r2_score(test_y, predicted)

            logging.info(f"Best Model Found: {best_model_name} with score {best_model_score}")

            return final_r2
        
        except Exception as e:
            raise CustomException(e, sys)

