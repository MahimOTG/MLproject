"""
Train machine learning models using the preprocessed training data.
Compare models and tune hyperparameters using validation data or cross-validation.
Evaluate the selected model on the test set and save it for future predictions.
"""

import os
import sys  # noqa: F401
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (  # noqa: F401
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object


# for every component we create a config class to store the configuration of that component
@dataclass
class  ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    '''
    This class defines the configuration for model training,
    specifically the file path where the trained model will be saved. 
    The trained_model_file_path is set to "artifacts/model.pkl", 
    indicating that the trained model will be stored in the artifacts directory.
    '''
   

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],#take out the last column as target variable
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "KNeighbors Regressor": KNeighborsRegressor()
            }
            model_report = evaluate_models(X=X_train,
                                           y_train=y_train,
                                           X_test=X_test,
                                           y_test=y_test,
                                           models=models)
            #to get the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))
            #to get the best model name from the dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model found",sys)
            logging.info(f"Best found model on both training and testing dataset is {best_model_name} with r2 score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predictted=best_model.predict(X_test)
            r2_square = r2_score(y_test, predictted)  # noqa: F841
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)
