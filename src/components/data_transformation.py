import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransfConfig:
    preprocessor_obj_file=os.path.join('artifacts','preprocessor.pkl') # It simply defines where the preprocessor will be saved later.

class DataTransformation:
    
    def __init__(self):
        self.data_transformation_config=DataTransfConfig()

    def preprocessor(self):
        """
        This function is responsible for Data Transformation and covers the recipe for transforming data 
        """
        try:
            ordinal_features = ['parental_level_of_education']
            nominal_features = ['gender', 'race_ethnicity', 'lunch', 'test_preparation_course']
            numerical_features=['writing_score','reading_score'] # math_score is ot considered since it is the target variable

            education_order = [
            "some high school",
            "high school",
            "some college",
            "associate's degree",
            "bachelor's degree",
            "master's degree"
        ]



            numerical_pipeline=Pipeline(
                steps=[
                    ('scaler',StandardScaler())
                ]
            )
        
            cat_ord_pipeline=Pipeline(
                steps=[
                    ('ordinal',OrdinalEncoder(categories=[education_order]))
                ]
            )
            cat_nom_pipeline=Pipeline(
                steps=[
                    ('nominal',OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            pre_processor=ColumnTransformer(
                [
                    ('num_pipeline',numerical_pipeline,numerical_features),
                    ('cat_ord_pipelin',cat_ord_pipeline,ordinal_features),
                    ('cat_nom_pipelin',cat_nom_pipeline,nominal_features)
                ]
            )
            
            logging.info("Categorical columns normalizing and encoding completed ")
            logging.info("Numerical columns scaling completed ")

            return pre_processor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_tranformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path) # in data_ingestion file we splitted the main data into entire train and test set
            test_df=pd.read_csv(test_path)

            logging.info("CSV Data succesfully loaded as dataframe")

            logging.info("Preprocessing object...")
  
            target_col='math_score'

            # splitting the train_data & test_data we loaded into x and y so that we can separate out the training (independent) & target(target) variable
            train_x=train_df.drop(target_col,axis=1) # x data used for training
            train_y=train_df[target_col] # y or corresponding result data used for training  

            test_x=test_df.drop(target_col,axis=1) # x data used for testing
            test_y=test_df[target_col] # y or results used for evaluating model's perfromance

            preprocess=self.preprocessor() # object of the DataTransformation class and preprocessor function 
            logging.info("Applying preprocessor on training data")

            train_x_transformed=preprocess.fit_transform(train_x) # preprocessing training data 
            test_x_transformed=preprocess.transform(test_x)  # preprocessing testing data 

            logging.info("Data transformation completed successfully.")

            logging.info(f"Saved preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file, 
                obj=preprocess # saves the preprocessor object as a file.pkl and serializes everything using dill(specified in utils)
            )

            return (
                train_x_transformed,
                test_x_transformed,
                train_y,
                test_y,
            )

        except Exception as e:
            raise CustomException(e,sys)
