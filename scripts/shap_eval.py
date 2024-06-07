import os
import xgboost as xgb
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



PATH = os.getcwd()

model = load(os.path.join(PATH, '../saved_objects/model.pth'))
dataset = load(os.path.join(PATH, '../saved_objects/dataset.joblib'))


explainer = shap.TreeExplainer(model, dataset['X_train'][:100])

shap_values = explainer(dataset['X_train'])

shap.plots.beeswarm(shap_values, show=False)

beeswarm = plot_to_numpy()
Image.fromarray(beeswarm).save('beeswarm.png')

plt.clf()
shap.plots.waterfall(shap_values[0], show=False)

waterfall = plot_to_numpy()
Image.fromarray(waterfall).save('waterfall.png')
