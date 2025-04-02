##read database, save data to cloud

import os
import sys

import pandas as pd
import numpy as np

from src.exception import CustomException


import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_object:
            dill.dump(obj, file_object)


    except Exception as e:
        raise CustomException(e, "Error occurred while saving object")



from sklearn.metrics import r2_score

def evaluated_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]

            para = params[model_name]

            gs = GridSearchCV(model, para, cv=3)

            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)


            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = {
                "train_score": train_model_score,
                "test_score": test_model_score
            }

        return report
    
    except Exception as e:
        raise CustomException(e, sys)
