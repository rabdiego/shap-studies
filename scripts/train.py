import os
from joblib import load, dump
import xgboost as xgb

PATH = os.getcwd()

dataset = load(os.path.join(PATH, '../saved_objects/dataset.joblib'))

model = xgb.XGBClassifier()

model.fit(dataset['X_train'], dataset['y_train'])

dump(model, os.path.join(PATH, '../saved_objects/model.pth'))