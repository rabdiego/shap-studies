import os
import xgboost as xgb
from joblib import load
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

PATH = os.getcwd()

model = load(os.path.join(PATH, '../saved_objects/model.pth'))
dataset = load(os.path.join(PATH, '../saved_objects/dataset.joblib'))

y_pred = model.predict(dataset['X_test'])

cm = confusion_matrix(dataset['y_test'], y_pred)
acc = accuracy_score(dataset['y_test'], y_pred)
prec = precision_score(dataset['y_test'], y_pred)
rec = recall_score(dataset['y_test'], y_pred)
f1 = f1_score(dataset['y_test'], y_pred)

fig, axs = plt.subplots(1, 2, figsize=(8, 4))

cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
cm_display.plot(ax=axs[0])

axs[1].bar(
    ['Acc', 'Prec', 'Rec', 'F1'],
    [acc, prec, rec, f1]
)

fig.savefig(os.path.join(PATH, '../results/eval_plot.png'))
