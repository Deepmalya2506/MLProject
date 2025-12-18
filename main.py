from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if (__name__=="__main__"):

    data_ing_obj=DataIngestion()
    data_transf_obj=DataTransformation()
    model_train_obj=ModelTrainer()

    train_data,test_data=data_ing_obj.initiate_data_ingestion()
    transformed_x_train, transformed_x_test, y_train, y_test=data_transf_obj.initiate_data_tranformation(train_data, test_data)

    accuracy = model_train_obj.initiate_training(transformed_x_train, transformed_x_test, y_train, y_test)

    print(f"Accuracy of Best Model: {accuracy}")
