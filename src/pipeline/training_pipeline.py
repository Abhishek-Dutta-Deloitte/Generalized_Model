import os
import sys
from src.logger import logger


from src.components.data_ingetion import DataIngetion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import InitiateModelTraining

if __name__ == '__main__':
    data_ingetion_obj = DataIngetion()
    train_data_path, test_data_path = data_ingetion_obj.inititate_data_ingetion()
    data_transformation_obj = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation_obj.inititate_data_transformation(train_data_path, test_data_path)
    model_trainer_obj = InitiateModelTraining()
    model_trainer_obj.initiate_model_training(train_arr, test_arr)
    logger.info("### Model Training is done and both model and preprocessing file saved to artifacts directory###")

