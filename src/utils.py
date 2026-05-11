import os
import dill
import utils
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from src.logger import logging
from src.exception import CustomException

def save_object(file_path,obj):
    try:
        