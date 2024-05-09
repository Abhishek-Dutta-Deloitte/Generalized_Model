import os
import numpy as np
import pandas as pd
from src.exceptions import CustomException
from src.logger import logger
from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor

from src.logger import logger
from src.exceptions import CustomException
import json
from dataclasses import dataclass
import yaml
from src.utils import *


@dataclass
class ModelTrainerConfig:

    # Get parent Directory path :
    current_directory =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  

    # Define artifacts path
    artifact_path = os.path.join(current_directory, "artifacts")

    # #Define model.pkl path:
    # trained_model_path = os.path.join(artifact_path, "regression_model.pkl")
    
    # #Getting path of params.yaml
    config_path = os.path.join(current_directory,"models_regression.json")
    

class InitiateModelTraining:
    def __init__(self) :
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_training(self, train_array, test_array, user_model_name=None):
        """
        This function is used to initiate the model training process.
        """
        logger.info("Initiating model training process...")
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                )
            logger.info("######### %s", self.model_trainer_config.config_path)
            with open(self.model_trainer_config.config_path, 'r') as f:
                data_model = json.load(f)
            
            # Use eval to convert strings to Python objects
            models = {key: eval(value) for key, value in data_model.items()}
            
            
            # models = {
            # # 'LinearRegression': LinearRegression(),
            # 'Lasso': Lasso(),
            # 'Ridge': Ridge(),
            # 'ElasticNet': ElasticNet(),
            # 'SVR': SVR(),
            # 'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
            # 'RandomForestRegressor': RandomForestRegressor(random_state=42),
            # 'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
            # 'BaggingRegressor': BaggingRegressor(random_state=42)
            # }
            
            model_report: dict = evaluate_model_regression(X_train, y_train, X_test, y_test, models=models)

            logger.info('\n====================================================================================\n')
            logger.info(f'Model Report : {model_report}')
            logger.info('\n====================================================================================\n')

            # Get the model names and their scores
            model_scores = {model_name: model_info['score'] for model_name, model_info in model_report.items()}

            # Find the model with the best score
            best_model_name = max(model_scores, key=model_scores.get)
            best_model_score = model_scores[best_model_name]

            # Get the best parameters for the best model
            best_model_params = model_report[best_model_name]['best_params']
            best_model = models[best_model_name]
            
            # Create a new instance of the model using the best parameters
            best_model = models[best_model_name].set_params(**best_model_params)

            # Fit the model with the best parameters
            best_model.fit(X_train, y_train)
            
            logger.info(f"Model Name is:{best_model_name} and the best parameters are:{best_model_params}, with score {best_model_score}")
            #Define model.pkl path:
            final_model_name = f"{user_model_name}_regression_model.pkl"
            trained_model_path = os.path.join(self.model_trainer_config.artifact_path, final_model_name)
            
            #Save object:
            save_objects(trained_model_path, best_model)
            logger.info("The model is saved successfully to {}".format(trained_model_path))
        except Exception as e:
            logger.error("Error initiating model training process", e)
            raise CustomException(e)
            
            
    