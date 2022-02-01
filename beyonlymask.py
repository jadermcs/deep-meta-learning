#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import copy
import torch
import math
import wandb
import pathlib
import argparse
import datetime
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from typing import Optional, Any
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error


class BaseDataDataset(Dataset):
    def __init__(self, path, row_number=256, col_number=256,
                 scores_path="augment_data.csv"):
        self.path = pathlib.Path(path)
        self.files = np.array(list(self.path.glob('*')))
        self.scores = pd.read_csv(scores_path, index_col="filename")
        self.row_number = row_number
        self.col_number = col_number

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        name = self.files[idx]
        data = pd.read_parquet(name).drop(columns=["class"])
        data = data.values.T
        # pad rows and columns to match dimensions in the batch
        pad_shape = (0, self.row_number-data.shape[1],
                     0, self.col_number-data.shape[0])
        data = F.pad(torch.tensor(data), pad_shape).float()
        target = self.scores.loc[name.name].values
        target = torch.tensor(target).float()
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

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class AttentionMetaExtractor(nn.Module):
    def __init__(self, ninp, noutput, nhead=8, nhid=256, nlayers=12, dropout=.25):
        super(AttentionMetaExtractor, self).__init__()
        self.model_type = 'Transformer'
        encoder_block = Encoder(ninp, nhead, nhid)
        self.encoder = nn.ModuleList([copy.deepcopy(encoder_block) for _ in range(nlayers)])

    def forward(self, src: torch.Tensor, msk: Optional[torch.Tensor] = None) -> torch.Tensor:
        if msk is not None:
            msk_neg = torch.zeros_like(src).masked_fill(msk, -1e6)
            src += msk_neg
        out = src
        for block in self.encoder:
            out = block(out)
        return out


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Finetune a transformers "
                                    "model on a causal language modeling task")
    parser.add_argument("--meta_data", type=str, default="data.csv",
                        help="File containing train and validation files")
    parser.add_argument("--data_path", type=str, default="data/",
                        help="A path to save model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--nrows", type=int, default=256, help="Number of maximum rows in base data.")
    parser.add_argument("--ncols", type=int, default=256, help="Number of maximum cols in base data.")
    parser.add_argument("--nhead", type=int, default=16, help="Number of attention heads.")
    parser.add_argument("--noutput", type=int, default=2, help="Number of outputs being regressed.")
    parser.add_argument("--nhid", type=int, default=256, help="Number of hidden representation vector.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--blocks", type=int, default=12, help="Number of decoder blocks.")
    parser.add_argument("--dropout", type=float, default=.1, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train model.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    return parser.parse_args()

def main():
    """Training routing for the Beyonder Network.
    """
    args = parse_args()
    torch.manual_seed(0)
    time = datetime.datetime.now().isoformat()
    exp_name = f'maskonly-{args.blocks}-{args.nhead}-{args.nhid}-{args.noutput}reg'
    wandb.init(project='DeepMetaLearning', name=exp_name, config=args)

    base_data_train = DataLoader(BaseDataDataset("data/train/", args.nrows,
                                                 args.ncols),
                                 batch_size=args.batch_size,
                                 shuffle=True, num_workers=8)
    base_data_valid = DataLoader(BaseDataDataset("data/valid/", args.nrows,
                                                 args.ncols),
                                 batch_size=args.batch_size,
                                 num_workers=8)

    total_steps = len(base_data_train)*args.epochs

    model = AttentionMetaExtractor(args.nrows, args.noutput, args.nhead,
                                   args.nhid, args.blocks, dropout=args.dropout)
    model.to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  amsgrad=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
                                                    total_steps=total_steps)

    best_loss = math.inf
    progress_bar = tqdm(range(total_steps))
    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        for batch in base_data_train:
            x, y = [tensor.to(args.device) for tensor in batch]
            x_mask = (torch.rand_like(x) < args.dropout).to(args.device)
            output, embs = model(x, x_mask)
            loss = F.mse_loss(x, embs)
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
            x, y = [tensor.to(args.device) for tensor in batch]
            output, embs = model(x)
            loss = F.mse_loss(x, embs)
            valid_loss.append(loss.item())
        mloss = np.mean(valid_loss)
        wandb.log({"valid/loss": mloss, "epoch": epoch})
        if mloss < best_loss:
            best_loss = mloss
            output_dir = pathlib.Path(f"model")
            output_dir.mkdir(exist_ok=True)
            best_name = f"best-{exp_name}-{epoch}-{mloss:.5f}.pth"
            torch.save(model.state_dict(), output_dir/best_name)

if __name__ == "__main__":
    main()
