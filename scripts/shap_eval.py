import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import shap
from joblib import load
import matplotlib.pyplot as plt
import io
import numpy as np
from PIL import Image

def plot_to_numpy():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = np.array(Image.open(buf))
    buf.close()

    return img

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

x_train = torch.from_numpy(dataset['X_train'].values).float()
feature_names = list(dataset['X_train'].columns)

with torch.no_grad():
    outputs = model(x_train)
    probas = torch.softmax(outputs, 1)[:, 1]
    base_value = probas.mean().item()

explainer = shap.DeepExplainer(model, x_train[:100])

shap_values = explainer.shap_values(x_train)

exp = shap.Explanation(shap_values[1], data=x_train.numpy(), base_values=base_value, feature_names=feature_names)

shap.plots.beeswarm(exp, show=False)

beeswarm = plot_to_numpy()
Image.fromarray(beeswarm).save('beeswarm.png')

plt.clf()
shap.plots.waterfall(exp[5], show=False)

waterfall = plot_to_numpy()
Image.fromarray(waterfall).save('waterfall.png')
