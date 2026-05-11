import os 
import sys
from dataclasses import dataclass
from sklearn.ensemble import(
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    f1_score,precision_score,recall_score,roc_auc_score,accuracy_score,confusion_matrix
)
from src.logger import logging
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("spliting the train test start")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
            'Random':RandomForestClassifier(),
            'Gradeint':GradientBoostingClassifier(),
            'Logistic':LogisticRegression(),
            'Decision':DecisionTreeClassifier(),
            'Neighbors':KNeighborsClassifier()
            }
            model_report:dict=evaluate_models(x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test,models=models)
            best_model_score=max(sorted(model_report.value()))
            best_model_name=list(model_report.key())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException('No best model found')
            logging.info(f'Best found model on both training and test dataset')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(x_test)
            r2_square=r2_square(y_test,predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)
