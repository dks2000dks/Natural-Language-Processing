import torch.nn as nn
import torch.nn.functional as F
import torch as tr
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np

class TextClassificationModel(nn.Module):
    def __init__(self, VocabSize, EmbeddingDims, Units, NumClasses, device):
        super(TextClassificationModel, self).__init__()
        # Embedding Layer
        self.embedding = nn.Embedding(VocabSize, EmbeddingDims)
        # RNN Layer
        self.rnn = nn.GRU(EmbeddingDims, Units, bidirectional=True)
        # Flatten Layer
        self.flatten = nn.Flatten()
        # Linear Layer
        self.fc = nn.Linear(Units*2*35, NumClasses)

        # Activation Layers
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # Device
        self.device = device

    def forward(self, In):
        if self.device != "cpu":
            x = tr.cuda.LongTensor(In)
        else:
            x = tr.LongTensor(In)
        x = self.embedding(x)
        x, hn = self.rnn(x)
        x = self.flatten(x)
        x = self.relu(x)
        x = self.fc(x)
        Out = self.softmax(x)
        return Out



class NLPDataset(Dataset):
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]