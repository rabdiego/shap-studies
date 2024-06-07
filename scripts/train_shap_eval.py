import os
from joblib import load, dump
import xgboost as xgb
import shap

PATH = os.getcwd()

dataset = load(os.path.join(PATH, '../saved_objects/dataset.joblib'))

model = xgb.XGBClassifier()
model.fit(dataset['X_train'], dataset['y_train'])

y_pred = model.predict(dataset['X_test'])

explainer = shap.TreeExplainer(model, dataset['X_train'][:100])

shap_values = explainer(dataset['X_train'])

shap.plots.beeswarm(shap_values)