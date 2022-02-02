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
from sklearn import metrics
from sklearn.dummy import DummyClassifier

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
        ft["best_clf"] = scores_data.loc[fname.name].argmax()
        train_df.append(ft)

    valid_df = []
    valid_path = pathlib.Path(args.data_path)/'valid'
    valid_files = list(valid_path.glob('*.parquet'))
    for fname in tqdm(valid_files):
        ft = {"x": 1}
        ft["best_clf"] = scores_data.loc[fname.name].argmax()
        valid_df.append(ft)

    train_df = pd.DataFrame(train_df)
    valid_df = pd.DataFrame(valid_df)

    drop_columns = ["best_clf"]
    xtrain = train_df.drop(columns=drop_columns).values
    xtest = valid_df.drop(columns=drop_columns).values
    ytrain = train_df[drop_columns]
    ytrue = valid_df[drop_columns]
    dr = DummyRegressor()
    dr.fit(xtrain, ytrain)
    yhat = dr.predict(xtest)

    recall = metrics.recall_score(ytrue, yhat, average="micro")
    precis = metrics.precision_score(ytrue, yhat, average="micro")
    wandb.log({"recall": recall})
    wandb.log({"precision": precis})

if __name__ == "__main__":
    main()
