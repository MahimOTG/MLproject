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
            params={
                "Decision Tree":{
                    'criterion':['squared_error',  'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    'n_estimators':[8,16,32,64,128,256],
                    'criterion':['squared_error', 'absolute_error', 'poisson'],
                    'max_features':['sqrt','log2',None],
                },
                "Gradient Boosting":{
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[0.05, 0.1],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    
                    'max_features':['sqrt','log2',None],
                    'n_estimators':[50, 100],
                    'max_depth':[2, 3]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth':[6,8,10],
                    'learning_rate':[.1,.01,.05,.001],
                    'iterations':[30,50,100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators':[8,16,32,64,128,256]
                },
            }
            model_report = evaluate_models(
                                            X=X_train,
                                            y_train=y_train,
                                            models=models,
                                            params=params
                                            )

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                    raise ValueError("No model reached the minimum CV R² score of 0.6")

            logging.info(
                "Best model: %s | CV R²: %.4f",
                 best_model_name,
                 best_model_score
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            logging.info("Saved model. Test R²: %.4f", r2_square)

            return r2_square
        except Exception as e:
            raise CustomException(e,sys)
