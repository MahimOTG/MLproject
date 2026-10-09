"""
Store reusable helper functions shared across project components and pipelines.
Provide common operations such as saving objects, loading objects, and evaluating models.
Reduce duplicated code and keep the main workflow files focused on their tasks.
"""


import os
import sys
import numpy as np  # noqa: F401
import pandas as pd
from src.exception import CustomException
import dill  # noqa: F401
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # noqa: F401

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        pd.to_pickle(obj, file_path)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X, y_train)
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
            
        return report
    except Exception as e:
        raise CustomException(e, sys)