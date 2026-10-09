# Load the raw dataset from a CSV file, database, or another source.
# Split the dataset into training and testing sets.
# Save these datasets for data transformation and model training.
#Load,Check,Split,Save -> go for transformation and model training

import os  # noqa: F401
import sys  # noqa: F401
from dataclasses import dataclass  # noqa: F401 generates methods such as __intit__ 

import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401
from sklearn.model_selection import train_test_split  # noqa: F401

from src.exception import CustomException  # noqa: F401
from src.logger import logging  # noqa: F401
from src.components.data_transformation import DataTransformation  # noqa: F401
from src.components.data_transformation import DataTransformationConfig  # noqa: F401
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv") #join() build a path in os 
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    '''
    these are the input and output paths for the data ingestion process. The train_data_path and test_data_path are where the training and testing datasets will be saved, respectively. The raw_data_path is where the original dataset will be stored.
    '''

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() #the variable ingenstion_config consists the three values /instances of Dataingestionconfig

    def initiate_data_ingestion(self):
          '''if my data is stored in databases  this code will read from the database and 
             save it as a csv file in the artifacts folder.
            Then it will split the data into train and test sets and 
            save them as csv files in the artifacts folder.
            also couldve coded the sql aor mongodb client in utils function'''
          logging.info("Entered the data ingestion method or component")
          try:
            df = pd.read_csv(os.path.join("notebook", "data", "stud.csv")) #reading the data from the csv file

            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) #creating the artifacts folder if it does not exist
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) #saving the raw data as csv file in the artifacts folder
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42) #splitting the data into train and test setsq
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
          except Exception as e:
              raise CustomException(e,sys)
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = (
        data_transformation.initiate_data_transformation(
            train_data, test_data
        )
    )

    model_trainer = ModelTrainer()
    score = model_trainer.initiate_model_trainer(train_arr, test_arr)

    print("Model R² score:", score)