#!/usr/bin/env python
# coding: utf-8
import wandb
import pathlib
import argparse
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from pymfe.mfe import MFE
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor

def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Finetune a transformers "
                                    "model on a causal language modeling task")
    parser.add_argument("--meta_data", type=str, default="data.csv",
                        help="File containing train and validation files")
    parser.add_argument("--data_path", type=str, default="data/",
                        help="A path to save model.")
    return parser.parse_args()

def main():
    """Extract meta-features with pyMFE and evaluate MSE with LightGBM.
    """
    args = parse_args()
    wandb.init(project='DeepMetaLearning', name='dummy', config=args)
    train_df = []
    train_path = pathlib.Path(args.data_path)/'train'
    train_files = list(train_path.glob('*.parquet'))
    scores_data = pd.read_csv("augment_data.csv", index_col="filename")
    for fname in tqdm(train_files):
        ft = {"x": 1}
        for clf in scores_data.columns:
            ft[f"score_{clf}"] = scores_data.loc[fname.name, clf]
        train_df.append(ft)

    valid_df = []
    valid_path = pathlib.Path(args.data_path)/'valid'
    valid_files = list(valid_path.glob('*.parquet'))
    for fname in tqdm(valid_files):
        ft = {"x": 1}
        for clf in scores_data.columns:
            ft[f"score_{clf}"] = scores_data.loc[fname.name, clf]
        valid_df.append(ft)

    train_df = pd.DataFrame(train_df)
    valid_df = pd.DataFrame(valid_df)

    drop_columns = [f"score_{clf}" for clf in scores_data.columns]
    xtrain = train_df.drop(columns=drop_columns).values
    xtest = valid_df.drop(columns=drop_columns).values
    for score in drop_columns:
        ytrain = train_df[score].values
        ytest = valid_df[score].values
        dr = DummyRegressor()
        dr.fit(xtrain, ytrain)
        mse = mean_squared_error(ytest, dr.predict(xtest))
        wandb.log({f"mse_{score}": mse})

if __name__ == "__main__":
    main()
