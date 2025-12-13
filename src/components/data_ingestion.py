import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass # __init__ -> this makes up fot the __init__
class DataIngestionConfig: # stores only the configuration like where the files will be stored 
    train_data_path:str=os.path.join('artifacts','train.csv') # must be present in a folder named artifacts with the respective file name
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')


class DataIngestion: 

    def __init__(self):
        self.ingestion_config=DataIngestionConfig() # creates an instance of your config class and stores it inside the ingestion object.

    def initiate_data_ingestion(self):
        logging.info("Entered the data Ingestion Section...")
        try:
            df=pd.read_csv(r'C:\Users\DEEPMALYA\OneDrive\Desktop\pip_Malya\Python\ML\ML_End2End\notebooks\data\stud.csv')
            logging.info("Read datasets as dataframe.")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True) # creates folder only if it doesn’t exist.
            #folder name, file name retrived from the DataIngestionConfig class through the object ingestion_config created
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Splitting of data into train and test set initiated...")

            train_set, test_set=train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header= True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header= True)

            logging.info("Ingestion of data completed")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path, self.ingestion_config.raw_data_path

        except Exception as e:
            raise CustomException(e,sys)
        