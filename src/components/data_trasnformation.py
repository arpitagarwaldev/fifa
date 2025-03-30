from src.logger import logging
from src.exception import CustomException

import sys
import os
import scipy.sparse
from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import save_object


@dataclass
class DataTranformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor_object.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTranformationConfig()
    
    def get_data_transformer_object(self):
        '''
        Function is responsible for data transformation and handling missing values
        '''

        try:
            numerical_columns = ['height', 'weight', 'age', 'ball_control', 'dribbling', 
                                 'slide_tackle', 'stand_tackle', 'aggression', 'reactions', 
                                 'att_position', 'interceptions', 'vision', 'composure', 'crossing', 
                                 'short_pass', 'long_pass', 'acceleration', 'stamina', 'strength', 'balance', 
                                 'sprint_speed', 'agility', 'jumping', 'heading', 'shot_power', 'finishing', 
                                 'long_shots', 'curve', 'fk_acc', 'penalties', 'volleys', 'gk_positioning', 'gk_diving',
                                'gk_handling', 'gk_kicking', 'gk_reflexes'
                                ]
            categorical_columns = ['country', 'club']

            


            num_pipeline = Pipeline(
                steps =[
                    ('imputer', SimpleImputer(strategy = 'median')), #handle missing value
                    ('scaler', StandardScaler())                     #feature scaling                    
                ]
            )

            logging.info(f"NUMERICAL columns Std scaling completed: {numerical_columns}")

            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                    # ('scaler', StandardScaler())                  
                ]
            )

            logging.info(f"Categorical columns encoding completed: {categorical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        

        except Exception as e:
            raise CustomException
        

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f"Data loaded successfully from {train_path} and {test_path}")


            logging.info("Performing value column transformation")

            # Ensure values are strings
            train_df['value'] = train_df['value'].astype(str)
            test_df['value'] = test_df['value'].astype(str)

            # Remove non-digit characters
            train_df['value'] = train_df['value'].str.replace(r'[^\d]', '', regex=True)
            test_df['value'] = test_df['value'].str.replace(r'[^\d]', '', regex=True)

            # Convert to numeric
            train_df['value'] = pd.to_numeric(train_df['value'], errors='coerce').astype('Int64')
            test_df['value'] = pd.to_numeric(test_df['value'], errors='coerce').astype('Int64')


            logging.info(f"Obtaining preprosseing object: {train_df['value']}, {test_df['value']}")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'value'

            input_feature_train_df = train_df.drop(columns=['player', target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            logging.info(f"Shape of input_feature_train_df before preprocessing: {input_feature_train_df.shape}")
            logging.info(f"Shape of target_feature_train_df before preprocessing: {target_feature_train_df.shape}")
            logging.info(f"Columns in input_feature_train_df: {input_feature_train_df.columns.tolist()}")

            input_feature_test_df = test_df.drop(columns=['player', target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Apply preprocessing
            logging.info("Starting preprocessing...")
            logging.info(f"Input feature train df shape before preprocessing: {input_feature_train_df.shape}")
            logging.info(f"Input feature train df columns: {input_feature_train_df.columns.tolist()}")
            
            # Check for any missing values
            missing_cols = input_feature_train_df.columns[input_feature_train_df.isnull().any()].tolist()
            logging.info(f"Columns with missing values: {missing_cols}")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            logging.info(f"Shape after preprocessing - input_feature_train_arr: {input_feature_train_arr.shape}")
            
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info(f"Shape after preprocessing - input_feature_test_arr: {input_feature_test_arr.shape}")



            logging.info(f"Input feature train array shape: {input_feature_train_arr.shape}")
            logging.info(f"Target feature train array shape: {np.array(target_feature_train_df).shape}")
            

            target_train = np.array(target_feature_train_df).reshape(-1, 1)
            logging.info(f"Reshaped target train array shape: {target_train.shape}")
            

            if scipy.sparse.issparse(input_feature_train_arr):
                input_feature_train_arr = input_feature_train_arr.toarray()
            
            train_arr = np.c_[
                input_feature_train_arr, target_train
            ]
            logging.info(f"Input feature test array shape: {input_feature_test_arr.shape}")
            logging.info(f"Target feature test array shape: {np.array(target_feature_test_df).shape}")
            

            target_test = np.array(target_feature_test_df).reshape(-1, 1)
            logging.info(f"Reshaped target test array shape: {target_test.shape}")
            

            if scipy.sparse.issparse(input_feature_test_arr):
                input_feature_test_arr = input_feature_test_arr.toarray()
                
            test_arr = np.c_[
                input_feature_test_arr, target_test
            ]            
            
            logging.info(f"Preprocessing completed")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)