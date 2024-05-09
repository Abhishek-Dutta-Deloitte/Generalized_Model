import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.logger import logger
from src.utils import *
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass 
import os
from src.exceptions import CustomException



raw_file_name = 'heart_statlog_cleveland_hungary_final.csv'
@dataclass
class DataIngetionConfig:

    # Get parent Directory path :
    current_directory =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  
    logger.info(f"The current directory for data Ingetion is {current_directory}")
    #Define artifacts path
    artifact_path = os.path.join(current_directory, "artifacts")

    #Define raw, training and test path
    train_data_path: str = os.path.join(artifact_path, "train.csv")
    test_data_path: str = os.path.join(artifact_path, "test.csv")
    raw_data_path: str = os.path.join(artifact_path, "raw.csv")
    

class DataIngetion:
    def __init__(self) :
        self.ingetion_config = DataIngetionConfig()

    def inititate_data_ingetion(self, df):
        """
        This function will save the training and testing data as a .CSV file in artifacts directory
        ##This can be changed as per the requirement or if the file size is huge##
        """
        logger.info("Initiating data ingetion process...")
        try:


            os.makedirs(os.path.dirname(self.ingetion_config.raw_data_path), exist_ok=True)
            
            df.to_csv(self.ingetion_config.raw_data_path, header= True, index=False)
            logger.info("Raw data saved at {}".format(self.ingetion_config.raw_data_path))

            logger.info("Data ingetion process completed...")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logger.info("Train data shape: {}".format(train_set.shape))
            logger.info("Test data shape: {}".format(test_set.shape))


            train_data_path = self.ingetion_config.train_data_path
            train_set.to_csv(train_data_path, header= True, index=False)
            logger.info("Train data saved at {}".format(self.ingetion_config.train_data_path))

            test_data_path = self.ingetion_config.test_data_path
            test_set.to_csv(test_data_path, header= True, index=False)
            logger.info("Test data saved at {}".format(self.ingetion_config.test_data_path))

            logger.info("Data Ingetion process completed...")
            
            return train_data_path, test_data_path

        except Exception as e:
            logger.error("Error in data ingetion process")
            raise CustomException(e)

# if __name__ == "__main__":
#     data_ingetion = DataIngetion()
#     data_ingetion.inititate_data_ingetion()