import os
import sys
import yaml


from dataclasses import dataclass


from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException

from src.utils import save_object, evaluated_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "RandomForest": RandomForestRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "XGBoost": XGBRegressor(),
                "LinearRegression": LinearRegression(),
                "GradientBoosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "KNN": KNeighborsRegressor(),
                "DecisionTree": DecisionTreeRegressor()
            }

            with open("src/config/params_config.yaml", "r") as file:
                config = yaml.safe_load(file)

            params = config["models"]

            print(f"params: {params}")


            model_report:dict = evaluated_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, 
            models = models,  params= params)

            logging.info(f"Model report: {model_report}")

            best_model_name = max(model_report, key=lambda k: model_report[k]["test_score"])
            best_model_score = model_report[best_model_name]["test_score"]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("Model score is less than 0.6")
            
            logging.info(f"Best model found, {best_model_name} with accuracy: {best_model_score}")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            logging.info(f"Model saved to {self.model_trainer_config.trained_model_file_path}")

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            return r2_square
            
            
            




            
            
        except Exception as e:
            raise CustomException(e, sys)

        
