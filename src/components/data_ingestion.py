
import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split 
from dataclasses import dataclass
import sqlite3

from src.components.data_trasnformation import DataTransformation, DataTranformationConfig

from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'raw.csv')


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("data ingestion method or component")
        try:
            conn = sqlite3.connect('notebook/data/player_stats.db')
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM player_stats")
            headers = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            df = pd.DataFrame(rows, columns=headers)
            logging.info('building the dataframe')

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_data_path, index = False, header = True)

            logging.info("train test split")
            
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.data_ingestion_config.train_data_path, index = False, header = True)

            test_set.to_csv(self.data_ingestion_config.test_data_path, index = False, header = True)

            logging.info("Ingestion is done")

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path,
            )


        except Exception as e:
            raise CustomException(e)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_trasformation = DataTransformation()
    train_arr, test_arr, _  = data_trasformation.initiate_data_transformation(train_data, test_data) 

    modeltrainer = ModelTrainer()
    logging.info(modeltrainer.initiate_model_trainer(train_arr, test_arr))
