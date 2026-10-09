"""
Store reusable helper functions shared across project components and pipelines.
Provide common operations such as saving objects, loading objects, and evaluating models.
Reduce duplicated code and keep the main workflow files focused on their tasks.
"""


import os
import sys

import dill  # noqa: F401
import numpy as np  # noqa: F401
import pandas as pd
from sklearn.metrics import (  # noqa: F401
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold  # noqa: F401

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        pd.to_pickle(obj, file_path)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X, y_train, models, params):
    report = {}
    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"Tuning {name}...", flush=True)

        search = GridSearchCV(
            estimator=model,
            param_grid=params.get(name, {}),
            scoring="r2",
            cv=cv,
            refit=True,
            error_score="raise",
            verbose=2
        )

        search.fit(X, y_train)

        # Keep the fitted winning model for prediction and saving.
        models[name] = search.best_estimator_
        report[name] = search.best_score_

        logging.info(
            "%s | Best parameters: %s | CV R²: %.4f",
            name,
            search.best_params_,
            search.best_score_
        )

    return report


