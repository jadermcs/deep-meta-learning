#!/usr/bin/env python
# coding: utf-8
"""Script for generating base data."""
import random
import pathlib
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import robust_scale
from sklearn.metrics import f1_score
from sklearn.utils import resample


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Finetune a transformers "
                                    "model on a causal language modeling task")
    parser.add_argument("--meta_data", type=str, default="data.csv",
                        help="File containing train and validation files")
    parser.add_argument("--data_path", type=str, default="data/",
                        help="A path to save model.")
    parser.add_argument("--aug_size", type=int, default=128,
                        help="Number of times the data is replicated.")
    parser.add_argument("--fold", type=int, default=5,
                        help="K in the k-fold cv.")
    parser.add_argument("--sample_size", type=int, default=256,
                        help="Number of instances in the base dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()

def main():
    """Generate the base datasets by sampling random instances from original
    base data.
    """
    args = parse_args()
    random.seed(args.seed)
    data_path = pathlib.Path(args.data_path)
    train_path = data_path/'train'
    train_path.mkdir(exist_ok=True)
    valid_path = data_path/'valid'
    valid_path.mkdir(exist_ok=True)
    train_valid = pd.read_csv(args.meta_data)
    train_valid = train_valid[~train_valid.train].files.tolist()

    clf1 = DecisionTreeClassifier(random_state=args.seed)
    clf2 = KNeighborsClassifier()

    files = list(data_path.glob("*.csv"))
    progress_bar = tqdm(range(len(files)*args.aug_size), leave=False)

    for fname in files:
        data = pd.read_csv(fname.as_posix()).dropna()
        if data.shape[1] > 255:
            print(f"Skipping {fname.name}, to many columns")
            continue
        majority_class = data["class"].value_counts().sort_values().index[-1]
        data["class"] = (data["class"] == majority_class).astype(int)
        xdata = data.drop(columns=["class"])
        xcolnames = xdata.columns
        xdata = robust_scale(xdata.values)
        ydata = data["class"].values
        for i in range(args.aug_size):
            xsample, ysample = resample(xdata, ydata,
                                        n_samples=random.randint(128, args.sample_size),
                                        random_state=args.seed+i, stratify=ydata)
            kfold = KFold(n_splits=args.fold, shuffle=True, random_state=args.seed)
            scores1 = []
            scores2 = []
            for train_idx, test_idx in kfold.split(xsample, ysample):
                xtrain, y_train = xsample[train_idx], ysample[train_idx]
                xtest, y_test = xsample[test_idx], ysample[test_idx]
                clf1.fit(xtrain, y_train)
                scores1.append(f1_score(y_test, clf1.predict(xtest), average='weighted'))
                clf2.fit(xtrain, y_train)
                scores2.append(f1_score(y_test, clf2.predict(xtest), average='weighted'))
            dataframex = pd.DataFrame(xsample, columns=xcolnames)
            dataframex = dataframex[random.sample(dataframex.columns.to_list(),
                                                  len(dataframex.columns))]
            dataframe = pd.concat([dataframex, pd.DataFrame({'class': ysample})], axis=1)
            dataset_type = train_path if fname.name not in train_valid else valid_path
            save_path = dataset_type.joinpath(f"{fname.with_suffix('').name}_{np.mean(scores1):.5f}_"
                                     f"{np.mean(scores2):.5f}_{i}.parquet")
            dataframe.to_parquet(save_path.as_posix(), index=False)
            progress_bar.update(1)

if __name__ == "__main__":
    main()
