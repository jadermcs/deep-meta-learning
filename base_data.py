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
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
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

    classifiers = {
        "dt": DecisionTreeClassifier(random_state=args.seed),
        "knn": KNeighborsClassifier(),
        "nn": MLPClassifier(max_iter=1000)
    }

    dataset_stats = pd.DataFrame(columns=['number_of_rows', 'number_of_columns'],
                                 dtype=int)

    files = list(data_path.glob("*.csv"))
    progress_bar = tqdm(range(len(files)*args.aug_size))
    score_data = pd.DataFrame(columns=['filename']+list(classifiers.keys()))

    for fname in files:
        data = pd.read_csv(fname).dropna()
        if data.shape[1] > 255:
            print(f"Skipping {fname.name}, to many columns")
            continue
        dataset_stats = dataset_stats.append({
            'number_of_rows': data.shape[0],
            'number_of_columns': data.shape[1]}, ignore_index=True)
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
            kfold = StratifiedKFold(n_splits=args.fold, shuffle=True,
                                    random_state=args.seed+i)
            scores = {}
            for train_idx, test_idx in kfold.split(xsample, ysample):
                xtrain, y_train = xsample[train_idx], ysample[train_idx]
                xtest, y_test = xsample[test_idx], ysample[test_idx]
                for name in classifiers:
                    clf = classifiers[name]
                    clf.fit(xtrain, y_train)
                    if name not in scores: scores[name] = []
                    scores[name].append(f1_score(y_test, clf.predict(xtest), average='weighted'))
            dataframex = pd.DataFrame(xsample, columns=xcolnames)
            dataframex = dataframex[random.sample(dataframex.columns.to_list(),
                                                  len(dataframex.columns))]
            dataframe = pd.concat([dataframex, pd.DataFrame({'class': ysample})], axis=1)
            dataset_type = train_path if fname.name not in train_valid else valid_path
            save_path = dataset_type.joinpath(f"{fname.with_suffix('').name}_{i}.parquet")
            score_data = score_data.append({
                'filename':save_path.name,
                **{name:np.mean(scores[name]) for name in scores}
            }, ignore_index=True)
            dataframe.to_parquet(save_path, index=False)
            progress_bar.update(1)

    score_data.to_csv("augment_data.csv", index=False)
    print(dataset_stats.describe())

if __name__ == "__main__":
    main()
