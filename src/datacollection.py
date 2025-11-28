import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split


data = pd.read_csv(r"/Users/parimal/Desktop/PYTHON/PYTHONPRACTICE/PROJECTS/ml_pipeline/original_data/water_potability.csv")

train_data , test_data = train_test_split(data, test_size=0.2, random_state=42) 

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, 'data', 'raw')
os.makedirs(data_path, exist_ok=True)


os.makedirs(data_path, exist_ok=True)


train_data.to_csv(os.path.join(data_path, 'train_data.csv'), index=False)


test_data.to_csv(os.path.join(data_path, 'test_data.csv'), index=False)


