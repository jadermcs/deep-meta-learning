#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import numpy as np
import copy
import torch
import math
import pytorch_lightning as pl
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from typing import Optional, Any
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning.callbacks as cb
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger


class BaseDataDataset(Dataset):
    def __init__(self, path, mode='regression'):
        self.path = path
        self.files = np.array(os.listdir(path))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        name = self.files[idx]
        data = pd.get_dummies(pd.read_csv(self.path+name), columns=["class"])
        data = data.values.astype(float).T
        data = F.pad(torch.tensor(data), (0, 0, 0, 200-data.shape[0])).float()
        # target = [float(name.split('_')[-3])]
        target = [float(x) for x in name.split('_')[-3:-1]]
        if mode == 'ragression':
            target = torch.tensor(target).float()
        else:
            target = [torch.argmax(target)]
        return data, target


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


# In[4]:


class AttentionMetaExtractor(pl.LightningModule):

    def __init__(self, ninp, noutput, nhead=5, nhid=256, lr=1e-3, nlayers=12, dropout=.25):
        super(AttentionMetaExtractor, self).__init__()
        self.model_type = 'Transformer'
        encoder_block = Encoder(ninp, nhead, nhid)
        self.encoder = nn.ModuleList([copy.deepcopy(encoder_block) for _ in range(nlayers)])
        self.decoder = nn.Linear(ninp, nhid)
        self.output = nn.Linear(nhid, noutput)
        self.activation = F.relu
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.learning_rate = lr

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        output = src
        for block in self.encoder:
            output = block(output)
        output = torch.mean(output, dim=1)
        output = self.dropout1(output)
        output = self.dropout2(self.activation(self.decoder(output)))
        output = self.activation(self.output(output))
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = F.l1_loss(output, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = F.l1_loss(output, y)
        self.log('valid_loss', loss)
        self.log('valid_mse', F.mse_loss(output, y))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, amsgrad=True)
        scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                     'monitor': 'valid_loss',
                     'name': 'ReduceLROnPlateau'}
        return [optimizer], [scheduler]

batch_size = 128

base_data_train = DataLoader(BaseDataDataset("../samples_train/"), batch_size=batch_size,
                             shuffle=True, num_workers=8)
base_data_valid = DataLoader(BaseDataDataset("../samples_valid/"), batch_size=batch_size,
                             num_workers=8)


ninp = 256 # number of rows in base data
nhead = 8
# noutput = 1 # number of algorithms accuracies being regressed
noutput = 2 # number of algorithms accuracies being regressed
nhid = 128
lr = 1e-3
blocks = 8
dropout = 0.1


early_stop_callback = cb.early_stopping.EarlyStopping(
    monitor='valid_loss',
    min_delta=0.00,
    patience=10,
    verbose=False,
    mode='min'
)
lr_monitor = cb.LearningRateMonitor(logging_interval='step')
checkpoint_callback = cb.ModelCheckpoint(monitor='valid_loss')
wandb_logger = WandbLogger(name=f'AdamW-{batch_size}-{lr:.3}-{blocks}-{dropout}-{noutput}',project='deepmtl')


model = AttentionMetaExtractor(ninp, noutput, nhead, nhid, lr, blocks, dropout=dropout)
trainer = pl.Trainer(gpus=-1, logger=wandb_logger, max_epochs=100,
                     callbacks=[early_stop_callback, lr_monitor, checkpoint_callback])
trainer.fit(model, base_data_train, base_data_valid)
