import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump

PATH = os.getcwd()

raw_data = pd.read_csv(os.path.join(PATH, '../data/breast-cancer.csv'))

encoder = LabelEncoder()

X = raw_data.drop('diagnosis', axis=1).drop('id', axis=1)
y = encoder.fit_transform(raw_data['diagnosis'])

columns = list(X.columns)

scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

scaler.fit(X_train[columns])

X_train[columns] = scaler.transform(X_train[columns])
X_test[columns] = scaler.transform(X_test[columns])

dataset = {
    'X_train' : X_train,
    'X_test' : X_test,
    'y_train' : y_train,
    'y_test' : y_test
}

dump(dataset, os.path.join(PATH, '../saved_objects/dataset.joblib'))