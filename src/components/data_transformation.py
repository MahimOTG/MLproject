# Handle missing values, encode categorical columns, and scale numerical features as needed.
# Fit preprocessing on training data, then apply the same transformations to test data.
# Save the fitted preprocessing pipeline for consistent transformations during prediction.
import os
import sys  # noqa: F401
from dataclasses import dataclass

import numpy as np  # noqa: F401
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")
    '''
    This class defines the configuration for data transformation,
    specifically the file path where the fitted preprocessing object will be saved. 
    The preprocessor_obj_file_path is set to "artifacts/preprocessor.pkl", 
    indicating that the preprocessing pipeline will be stored in the artifacts directory.
    '''
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformer_object(self):
        try:
            numerical_columns =["writing_score","reading_score"]
            categorical_columns = ["gender",
                                   "race_ethnicity",
                                   "parental_level_of_education",
                                   "lunch",
                                   "test_preparation_course",
                                   ]
            numerical_pipeline = Pipeline(
                steps=[
                       ("imputer",SimpleImputer(strategy="median")),
                       ("scaler",StandardScaler())
                       ]

                       
            )
            categorical_pipeline = Pipeline(
                steps=[
                       ("imputer",SimpleImputer(strategy="most_frequent")),
                       ("one_hot_encoder",OneHotEncoder()),
                       ("scaler",StandardScaler(with_mean=False))
                       ]
            )

            logging.info("Numerical columns standard scaling Completed")
            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",numerical_pipeline,numerical_columns),
                    ("cat_pipelines",categorical_pipeline,categorical_columns)
                ]
            )

            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            logging.info(f"Train Dataframe Head : \n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head : \n{test_df.head().to_string()}")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns =["writing_score","reading_score"]  # noqa: F841

            input_feature_train_df = train_df.drop(columns=[target_column_name],)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
