import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

max_words = 5000
max_len = 200

# Define the NCF Model
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(128, 1)

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        x = torch.cat([user_emb, item_emb], dim=-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.output(x)
        return x
    

class TextCNN(nn.Module):
    def __init__(self, max_words, embedding_dim=128, max_len=max_len):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(max_words, embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=5, stride=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # Change shape for Conv1d
        x = torch.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.output(x)
        return x