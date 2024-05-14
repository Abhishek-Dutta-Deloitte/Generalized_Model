import os # We use os to create path...
import sys
from src.logger import logger
from src.exceptions import CustomException
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder # Ordinal Encoding for categorical variables
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer #Group everything together
from dataclasses import dataclass
from src.utils import *
from src.utils import replace_null_with_mean
import json

# current_directory =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  
# print(current_directory)
# #Define artifacts path
# artifact_path = os.path.join(current_directory, "artifacts")
# # print(artifact_path)
# preprocessor_obj_file_path = os.path.join(artifact_path, "preprocessor_obj.pkl")
# # print(preprocessor_obj_file_path)


@dataclass
class DataTransformationConfig:
    # Get parent Directory path :

    current_directory =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  

    #Define artifacts path
    artifact_path = os.path.join(current_directory, "artifacts")

class DataTransformation:
    
    def __init__(self) :
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self, categories, one_hot_cols, ordinal_cols, num_cols, target):
        
        try:
            # Independent numerical columns
            num_cols_list = [num for num in num_cols if num != target]  ##It can be automated via taking from UI

            # Define pipelines for categorical and numeric data
            categorical_onehot_pipeline = Pipeline([
                
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(sparse=False)),
                ('scaler', StandardScaler())
            ])

            categorical_ordinal_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OrdinalEncoder(categories=categories)),
                ('scaler', StandardScaler())
            ])

            numerical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            # Combine pipelines in a ColumnTransformer
            preprocessor = ColumnTransformer(transformers=[
                ('cat_one_hot', categorical_onehot_pipeline, one_hot_cols),
                ('cat_ordinal', categorical_ordinal_pipeline, ordinal_cols),
                ('num', numerical_pipeline, num_cols_list)
            ])

            logger.info("Pipeline methods creation ends!!!")
            return preprocessor
        
        except Exception as e:
            logger.error("Error in get_data_transformation_object")
            raise CustomException(e)
            
    def inititate_data_transformation(self, train_path, test_path, target, features_to_exclude, user_transformation_name=None):
        try:
            
            logger.info("Initiating data transformation process...")
                
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            logger.info("Data loaded successfully")
            
            df_train.drop(labels=features_to_exclude, axis=1, inplace=True)
            df_test.drop(labels=features_to_exclude, axis=1, inplace=True)
            logger.info("Features excluded successfully")
            
            # Number columns:
            num_cols = [col for col in df_train.columns if df_train[col].dtype!= 'object' and col != target]
            logger.info("Number columns identified successfully")
        
            #Removal of outliers:
            df_train = outlier_removal(df_train, num_cols)
            logger.info("Outliers removed!!!")

            # Outlier detection:
            df_test = outlier_removal(df_test, num_cols)
            logger.info("Outliers removed!!!")
            
            #Replacing NA with mean
            df_train = replace_null_with_mean(df_train, num_cols)
            df_test = replace_null_with_mean(df_test, num_cols)
            logger.info("Replaced Null with mean for input numerical variables")
            
            ##Replacing NA with mean for Target Variable:
            df_train = replace_null_with_mean(df_train, [target])
            df_test = replace_null_with_mean(df_test, [target])            
            logger.info("Replaced Null with mean for Target Variable")
            
            # Calling Feature Classifier for training data:
            feature_classifier_obj = FeatureClassifier(df_train,target)
            one_hot_cols, ordinal_cols, num_cols, ordinal_columns_mapping, one_hot_column_mapping = feature_classifier_obj.ordinal_onehot_numerical_divide()
            logger.info("Categorical columns (one hot, ordinal mapping)  and numerical columns divided successfully")

            #Fill empty feature with mode
            df_train = fill_empty_with_mode(df_train,one_hot_cols)
            df_train = fill_empty_with_mode(df_train,ordinal_cols)
            logger.info("Empty values filled with mode successfully")

            df_test = fill_empty_with_mode(df_test,one_hot_cols)
            df_test = fill_empty_with_mode(df_test,ordinal_cols)
            logger.info("Empty values filled with mode successfully")
            
            logger.info(one_hot_cols, ordinal_cols, num_cols, ordinal_columns_mapping, one_hot_column_mapping)
            # Listing all the categories:
            categories = []
            for key, value in ordinal_columns_mapping.items():
                categories.append(value)
            logger.info("Categories created successfully!!!")
            preprocessor_obj = self.get_data_transformation_object(categories, one_hot_cols, ordinal_cols, num_cols, target)
            
            # Segregation of input and target feature:

            
            X_train = df_train.drop(labels=target, axis=1)
            y_train = df_train[target]

            X_test = df_test.drop(labels=target, axis=1)
            y_test = df_test[target]

            logger.info("Input and target feature segregated successfully!!!")

            # #Transformation using preprocessing object:
            X_train_arr = preprocessor_obj.fit_transform(X_train)
            logger.info(X_train_arr)
            X_test_arr = preprocessor_obj.transform(X_test)
            logger.info("Preprocessing done successfully!!!")
            
            
            ###############################
            """
                Here We need to Implement the AI360 fairness Matrix algorithm
            """

        
            ###############################

            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            logger.info("Data transformation done successfully!!!")

            categorical_column_mapping = ordinal_columns_mapping | one_hot_column_mapping

            #Saving ordinal_columns_mapping into ordinal_columns_mapping.json for app.py use:
            categorical_column_mapping_path = os.path.join(self.data_transformation_config.current_directory, "categorical_column_mapping.json")
            with open(categorical_column_mapping_path, 'w') as f:
                json.dump(categorical_column_mapping, f)

            logger.info("ordinal_columns_mapping.json created successfully!!!")
            
            
            final_transformation_obj_name = f"{user_transformation_name}_preprocessor_obj.pkl"
            preprocessor_obj_file_path = os.path.join(self.data_transformation_config.artifact_path, final_transformation_obj_name)
            logger.info("The file is saved to: {}".format(preprocessor_obj_file_path))
            
            #Saving the Pickle file preprocessing object:
            save_objects(
                file_path = preprocessor_obj_file_path,
                obj = preprocessor_obj
            )        
            logger.info("Saved the Pickle file preprocessing object")    
            return train_arr, test_arr, preprocessor_obj_file_path
        except Exception as e:
            logger.error("Error occured while initiating the data transformation process: {}".format(e))
            raise CustomException(e)

# if __name__ == "__main__":

#     train_data_path = "C:/Users/abhishdutta/Desktop/PRD Projects/IBM WatsonX/Heart Disease Classification/artifacts/train.csv"
#     test_data_path = "C:/Users/abhishdutta/Desktop/PRD Projects/IBM WatsonX/Heart Disease Classification/artifacts/test.csv"

#     data_transformation_obj = DataTransformation()
#     train_arr, test_arr, preprocessor_path = data_transformation_obj.inititate_data_transformation(train_data_path, test_data_path)
    
