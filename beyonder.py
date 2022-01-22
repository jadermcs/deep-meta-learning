#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import numpy as np
import copy
import torch
import math
import wandb
from tqdm import tqdm
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from typing import Optional, Any
from torch.utils.data import DataLoader, Dataset


wandb.init(project='DeepMetaLearning')


# In[3]:


col_number = 200


# In[4]:


class BaseDataDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.files = np.array(os.listdir(path))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        name = self.files[idx]
        data = pd.read_parquet(self.path+name).drop(columns=["class"])
        data = data.values.T
        data = F.pad(torch.tensor(data), (0, 0, 0, col_number-data.shape[0])).float()
        target = [float(x) for x in name.split('_')[-3:-1]]
        target = torch.tensor(target)
        return data, target


# In[5]:


class Encoder(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super(Encoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


# In[6]:


class AttentionMetaExtractor(nn.Module):

    def __init__(self, ninp, noutput, nhead=8, nhid=256, nlayers=12, dropout=.25):
        super(AttentionMetaExtractor, self).__init__()
        self.model_type = 'Transformer'
        encoder_block = Encoder(ninp, nhead, nhid)
        self.embed = nn.Embedding(1, ninp)
        self.encoder = nn.ModuleList([copy.deepcopy(encoder_block) for _ in range(nlayers)])
        self.decoder = nn.Linear(ninp, nhid)
        self.output = nn.Linear(nhid, noutput)
        self.activation = F.relu
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, clf:torch.Tensor) -> torch.Tensor:
        clf = self.embed(clf).unsqueeze(1)
        output = torch.cat((clf, src), dim=1)
        for block in self.encoder:
            output = block(output)
        output = self.decoder(self.dropout1(output[:,0]))
        return self.output(output)


# In[7]:


batch_size = 64

base_data_train = DataLoader(BaseDataDataset("../samples_train/"), batch_size=batch_size,
                             shuffle=True, num_workers=8)
base_data_valid = DataLoader(BaseDataDataset("../samples_valid/"), batch_size=batch_size,
                             num_workers=8)


# In[8]:


ninp = 256 # number of rows in base data
nhead = 8
noutput = 2 # number of algorithms accuracies being regressed
nhid = 512
learning_rate = 1e-4
blocks = 8
dropout = 0.25
device = 'cuda'
epochs = 100
total_steps = len(base_data_train)*epochs


# In[9]:


model = AttentionMetaExtractor(ninp, noutput, nhead, nhid, blocks, dropout=dropout)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, amsgrad=True)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, total_steps=total_steps)


# In[10]:


best_loss = math.inf

progress_bar = tqdm(range(total_steps))
for epoch in range(epochs):
    model.train()
    train_loss = []
    for batch in base_data_train:
        x, y = [tensor.to(device) for tensor in batch]
        clf_tensor = torch.LongTensor([0]*x.shape[0]).to(device)
        output = model(x, clf_tensor)
        loss = F.mse_loss(output, y)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    mloss = np.mean(train_loss)
    wandb.log({"train/loss": mloss, "epoch": epoch})

    model.eval()
    valid_loss = []
    for batch in base_data_valid:
        x, y = [tensor.to(device) for tensor in batch]
        clf_tensor = torch.LongTensor([0]*x.shape[0]).to(device)
        output = model(x, clf_tensor)
        loss = F.mse_loss(output, y)
        valid_loss.append(loss.item())
    mloss = np.mean(valid_loss)
    wandb.log({"valid/loss": mloss, "epoch": epoch})
    if mloss < best_loss:
        best_loss = mloss
        output_dir = f"model/best-{epoch}-{mloss:.5f}/"
        model.save_pretrained(output_dir)

