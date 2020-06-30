import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn


class ModelStructure(nn.Module):
    def __init__(self, cate_counts, ebd_size, num_counts):
        super().__init__()
        self.hidden_layer = nn.Linear(cate_counts, ebd_size)
        self.out_layer = nn.Linear(ebd_size, num_counts)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.out_layer(x)
        return x


class DeepModel():
    def __init__(self, model_structure):
        self.device = "cpu"
        self.model_structure = model_structure.to(self.device)
        self.loss_function_name = self.loss_funtion = nn.MSELoss().to(self.device)
        self.loss_function_name == "mse"
             
    def __str__(self):
        return ("Model created.\nModel structure: {}\nLoss function: {}".format(
            self.model_structure, self.loss_function_name))
        
    def batch_train(self, batch_X, batch_Y):
        self.model_structure.zero_grad()
        batch_pred = self.model_structure(batch_X)
        batch_loss = self.loss_funtion(batch_pred, batch_Y)
        batch_loss.backward()
        self.optimizer.step()
        return batch_loss
        
    def batch_predict(self, batch_X):
        with torch.no_grad():
            return self.model_structure(batch_X)
        
    def train(self, dataset, batch_size, epoch_round, learning_rate=1e-3):
        self.optimizer = torch.optim.Adam(self.model_structure.parameters(), lr=learning_rate)
        train_dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        tr_y = []
        for _, batch_y in train_dataset:
            tr_y.append(batch_y.numpy())
        tr_y = np.concatenate(tr_y)
        for epoch in range(epoch_round):
            for i, batch in enumerate(train_dataset):
                batch_X, batch_y = batch
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                batch_loss = self.batch_train(batch_X, batch_y)
                print(str(round((i + 1) / len(train_dataset) * 100, 1)) + "% complete;\t" + "loss: " + str(batch_loss) + "\r", end="")
            tr_pred = []
            for batch_X, _ in train_dataset:
                batch_pred = self.batch_predict(batch_X.to(self.device))
                tr_pred.append(batch_pred)
            tr_pred = np.concatenate(tr_pred)
            print("Epoch {} - Train set MSE: {}".format(epoch, mean_squared_error(tr_y, tr_pred)))
        
    def predict(self, X, batch_size):
        X = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=False)
        pred = np.array([])
        for batch_X in X:
            batch_pred = self.batch_predict(batch_X)
            pred = np.hstack([pred, batch_pred])
        return pred


class CateEmbedding:
    def __init__(self, cate_feat, covariant, threshold=10):
        droped = cate_feat.value_counts()[cate_feat.value_counts() < threshold].index
        self.name = cate_feat.name
        self.cate_feat = pd.get_dummies(cate_feat.replace({x: np.nan for x in droped}))
        
        self.covariant = MinMaxScaler().fit_transform(covariant)
        self.tensor = torch.utils.data.TensorDataset(
            torch.from_numpy(self.cate_feat.values).float(), 
            torch.from_numpy(self.covariant).float()
        )
        self.output_size = covariant.shape[1]
        self.embedding = None
        
    def retrain(self, size=50, batch_size=256, epoch_round=100, learning_rate=1e-4):
        deep_model_struc = ModelStructure(self.cate_feat.shape[1], size, self.output_size)
        model = DeepModel(deep_model_struc)
        print(model)
        model.train(self.tensor, batch_size, epoch_round, learning_rate=learning_rate)
        self.embedding = pd.DataFrame(
            model.model_structure.hidden_layer.weight.detach().numpy().T, 
            index=self.cate_feat.columns,
            columns=[self.name + '_{}'.format(i + 1) for i in range(size)]
        )
        print('categerical feature embedding model retrain complete.')
        
    def transform(self, cate_feat):
        return self.embedding.reindex(cate_feat.values).fillna(0)
        