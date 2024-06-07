import os
from joblib import load, dump
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

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

dataset = load(os.path.join(PATH, '../saved_objects/dataset.joblib'))

model = Model()
num_epochs = 100
lr = 0.001

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = lr)

attributes = torch.from_numpy(dataset['X_train'].values).float()
labels = torch.from_numpy(dataset['y_train'])

for epoch in range(num_epochs):
    outputs = model(attributes)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f'Epoch {epoch} Loss {loss.item()}')

torch.save(model.state_dict(), os.path.join(PATH, '../saved_objects/model.pth'))