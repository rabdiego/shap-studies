import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import load
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30, 10)
        self.fc2 = nn.Linear(10, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


PATH = os.getcwd()

model = Model()
model.load_state_dict(torch.load(os.path.join(PATH, '../saved_objects/model.pth')))
dataset = load(os.path.join(PATH, '../saved_objects/dataset.joblib'))

y_pred = list()
y_test = list()

attributes = torch.from_numpy(dataset['X_test'].values).float()
labels = torch.from_numpy(dataset['y_test'])

with torch.no_grad():
    outputs = model(attributes)
    _, classes = torch.max(outputs, 1)
    y_pred.extend(classes.tolist())
    y_test.extend(labels.tolist())


y_pred = np.array(y_pred)
y_test = np.array(y_test)

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
