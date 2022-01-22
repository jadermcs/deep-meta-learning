#!/usr/bin/env python
# coding: utf-8
import pathlib
import argparse
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from pymfe.mfe import MFE
from fastprogress.fastprogress import master_bar, progress_bar
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

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
    return parser.parse_args()

def main():
    """Extract meta-features with pyMFE and evaluate MSE with LightGBM.
    """
    args = parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    mfe = MFE(random_state=args.seed, groups=["statistical"])
    print("Extracting meta-features for train files")
    train_df = []
    train_path = pathlib.Path(args.data_path)/'train'
    train_files = list(train_path.glob('*.parquet'))
    for fname in tqdm(train_files):
        break
        score_dt = fname.name.split("_")[-3]
        score_knn = fname.name.split("_")[-2]
        df = pd.read_parquet(fname.as_posix())
        X = df.drop(columns=["class"]).values
        # First evaluate only unsupervised features
        #y = df["class"].values
        mfe.fit(X)
        ft = mfe.extract()
        ft = dict(zip(*ft))
        ft["score_dt"] = score_dt
        ft["score_knn"] = score_knn
        train_df.append(ft)

    print("Extracting meta-features for validation files")
    valid_df = []
    valid_path = pathlib.Path(args.data_path)/'valid'
    valid_files = list(valid_path.glob('*.parquet'))
    for fname in tqdm(valid_files):
        print(fname)
        score_dt = fname.name.split("_")[-3]
        score_knn = fname.name.split("_")[-2]
        df = pd.read_csv(fname.as_posix())
        X = df.drop(columns=["class"]).values
        # First evaluate only unsupervised features
        #y = df["class"].values
        mfe.fit(X)
        ft = mfe.extract()
        ft = dict(zip(*ft))
        ft["score_dt"] = score_dt
        ft["score_knn"] = score_knn
        valid_df.append(ft)

    train_df = pd.DataFrame(train_df)
    valid_df = pd.DataFrame(valid_df)
    train_df.to_csv("mfe.train.csv", index=False)
    train_df.to_csv("mfe.test.csv", index=False)

    xtrain = train_df.drop(columns=["score_dt", "score_knn"]).values
    ytrain = train_df["score_dt"].values
    xtest = valid_df.drop(columns=["score_dt", "score_knn"]).values
    ytest = valid_df["score_dt"].values

    lg = LGBMRegressor(random_state=args.seed, objective='mse')
    lg.fit(xtrain, ytrain)
    print(mean_absolute_error(ytest, lg.predict(xtest)))

if __name__ == "__main__":
    main()
