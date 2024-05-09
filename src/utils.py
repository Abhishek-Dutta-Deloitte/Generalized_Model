import os
import pickle
import yaml
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.logger import logger

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score


# Get parent Directory path :
current_directory =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
#Getting path of params_classification.yaml
classification_param_path = os.path.join(current_directory,"params_classification.yaml")

regression_param_path = os.path.join(current_directory,"params_regression.yaml")

def replace_null_with_mean(df, num_cols):
    """
    This function replaces null values in the specified numerical columns of a DataFrame with their mean.

    Parameters:
    df (pandas.DataFrame): The DataFrame.
    num_cols (list): The list of numerical columns.

    Returns:
    pandas.DataFrame: The DataFrame with null values replaced.
    """
    for col in num_cols:
        if col in df.columns:
            df[col].fillna(df[col].mean(), inplace=True)
    return df

def save_objects(file_path, obj):
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        logger.info("Object saved successfully")
    except Exception as e:
        logger.info("Error in save_objects: {}".format(e))
        
def outlier_removal(df, num_cols):
    for column in num_cols:
        upper_limit = df[column].mean() + 2 * df[column].std()
        lower_limit = df[column].mean() - 2 * df[column].std()
        df = df[(df[column] < upper_limit) & (df[column] > lower_limit)]
    return df

def fill_empty_with_mode(df, cat_cols):
    for i in cat_cols:
        if (df[i] == '').any():
            mode_value = df[i][df[i]!=""].mode().iloc[0]
            df[i] = df[i].replace('',mode_value )
    return df




def random_search_cv(model, X_train, y_train,params):
    random_cv = RandomizedSearchCV(model, param_distributions=params, scoring="r2", cv = 5, verbose=0 )
    random_cv.fit(X_train, y_train)
    return random_cv, random_cv.best_params_, random_cv.best_score_

#Confusion Matrix

def confusion_matrix_classification_report(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    
    #Classification Report
    report = classification_report(y_test, y_pred)
    logger.info(report)
    

class FeatureClassifier:
    def __init__(self,df, target_column):
        self.df = df
        self.target_column = target_column
    
    def get_ordinal_columns_mapping(self,columns):
        """
        This function is used to get the mapping of ordinal columns.
        Each key is named as 'ColumnName_Map' and contains the unique values for that column.
        """
        columns_mapping = {}
        
        for col in columns:
            sorted_groups = self.df.groupby(col)[self.target_column].mean().sort_values().index.tolist()
            key_name = f"{col}"
            columns_mapping[key_name] = sorted_groups
        
        return columns_mapping
        

        
    def ordinal_onehot_numerical_divide(self):
        """
        This function is used to divide the categorical into ordinal and one-hot columns and numerical columns.
        """
        one_hot_cols = []
        ordinal_cols = []
        num_cols = []
        #Overall mean
        mean = self.df[self.target_column].mean()
        thereshold_percentage = 0.1
        threshold_value = mean * thereshold_percentage
        try:
            for column in self.df.columns:
                if column != self.target_column and self.df[column].dtype == 'object':
                    df_column = self.df[[column, self.target_column]].groupby(column).mean().reset_index()
                    standard_dev = df_column[self.target_column].std()
                    if standard_dev > threshold_value:
                        ordinal_cols.append(column)
                    else:
                        one_hot_cols.append(column)
                else:
                    num_cols.append(column)
            
            logger.info("ordinal_onehot_numerical_divide done!!!")

            #Get Mappingsd for ordinal columns:
            ordinal_columns_mapping = self.get_ordinal_columns_mapping(ordinal_cols)
            one_hot_column_mapping = self.get_ordinal_columns_mapping(one_hot_cols)
            return (one_hot_cols, ordinal_cols, num_cols, ordinal_columns_mapping, one_hot_column_mapping)
                 

        except Exception as e:
            logger.info(e)

def save_objects(file_path, obj):
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        logger.info("Object saved successfully")
    except Exception as e:
        logger.info("Error in save_objects: {}".format(e))
    

def load_obj(file_path):
    try:
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        logger.info("Object loaded successfully")
        return obj
    except Exception as e:
        logger.info("Error in load_obj: {}".format(e))
    
from sklearn.metrics import accuracy_score


def random_search_cv_classification(model, X_train, y_train, params):
    random_cv = RandomizedSearchCV(model, param_distributions=params, scoring="accuracy", cv = 5, verbose=0 )
    random_cv.fit(X_train, y_train)
    return random_cv, random_cv.best_params_, random_cv.best_score_


def random_search_cv_regression(model, X_train, y_train, params):
    random_cv = RandomizedSearchCV(model, param_distributions=params, scoring="r2", cv = 5, verbose=0)
    random_cv.fit(X_train, y_train)
    return random_cv, random_cv.best_params_, random_cv.best_score_


def evaluate_model_classification(X_train, y_train, X_test, y_test, models):
    report = {}
    
    # config_path = "../params.yaml"
    #Load yaml file:x
    try:
        with open(classification_param_path, 'r') as file:
            config = yaml.safe_load(file)
        
        for i in range(len(models)):
            model = list(models.values())[i]
            model_flag = list(models.keys())[i]
            # model.fit(X_train, y_train)
            
            params = config[model_flag]
            model, model.best_params_, model.best_score_ = random_search_cv_classification(model, X_train, y_train, params)
            
            y_pred = model.predict(X_test)

            test_model_score = accuracy_score(y_test, y_pred)
            logger.info('\n====================================================================================\n')
            logger.info(f"The confusion matrix and classification report for the model: {model_flag} is:")
            confusion_matrix_classification_report(y_test, y_pred)
            logger.info('\n====================================================================================\n')


            logger.info('\n====================================================================================\n')
            logger.info(f"The best parameters for the model{model_flag} are {model.best_params_}")
            logger.info('\n====================================================================================\n')


            report[list(models.keys())[i]] =  {"score": test_model_score, "best_params": model.best_params_}
            logger.info(f"Model: {list(models.keys())[i]}, Accuracy score: {test_model_score}")
        logger.info("Model evaluation complete")
        return report

    except Exception as e:
        logger.info("Error in evaluate_model: {}".format(e))
        



def evaluate_model_regression(X_train, y_train, X_test, y_test, models):
    report = {}

    try:
        with open(regression_param_path, 'r') as file:  # Ensure correct path to your regression parameters file
            config = yaml.safe_load(file)

        for i in range(len(models)):
            model = list(models.values())[i]
            model_flag = list(models.keys())[i]

            params = config[model_flag]
            model, model.best_params_, model.best_score_ = random_search_cv_regression(model, X_train, y_train, params)

            y_pred = model.predict(X_test)

            test_model_score = r2_score(y_test, y_pred)  # R-squared used as evaluation metric
            logger.info('\n====================================================================================\n')
            logger.info(f"The R-squared and RMSE for the model: {model_flag} are:")
            logger.info(f"R-squared: {test_model_score}")  # Reporting RMSE along with R-squared
            logger.info('\n====================================================================================\n')

            logger.info('\n====================================================================================\n')
            logger.info(f"The best parameters for the model {model_flag} are {model.best_params_}")
            logger.info('\n====================================================================================\n')

            report[list(models.keys())[i]] = {"score": test_model_score, "best_params": model.best_params_}
            logger.info(f"Model: {list(models.keys())[i]}, R-squared: {test_model_score}")
        logger.info("Model evaluation complete")
        return report

    except Exception as e:
        logger.info("Error in evaluate_model: {}".format(e))
        raise e