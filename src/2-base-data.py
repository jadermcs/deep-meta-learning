#!/usr/bin/env python
# coding: utf-8
import os
import random
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from fastprogress.fastprogress import master_bar, progress_bar
from sklearn.utils import resample

seed = 42
#random.seed(seed)
path = '../data/'

os.makedirs('../samples_train', exist_ok=True)
os.makedirs('../samples_valid', exist_ok=True)

train_valid = pd.read_csv("../data.csv")
train_valid = train_valid[train_valid.train==False].files.tolist()

clf1 = DecisionTreeClassifier(random_state=seed)
clf2 = KNeighborsClassifier()

mb = master_bar(os.listdir(path))
count = 0

aug_size = 300
percentage_valid = .1
fold = 5
sample_size = 256

for f in mb:
    mb.main_bar.comment = f'Files'
    data = pd.read_csv(path+f).dropna()
    if data.shape[0] < 150 or data.shape[1] > 200:
        continue

    X = data.drop(columns=["class"]).values
    y = data["class"].values
    if y.dtype == float or any(np.unique(y, return_counts=True)[1] < fold):
        print(f"{f} has a continuous y value or too imbalanced.")
        continue
    for i in progress_bar(range(aug_size), parent=mb):
        Xsample, ysample = resample(X, y, n_samples=sample_size, random_state=seed+i, stratify=y)
        kfold = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
        scores1 = []
        scores2 = []
        for train_idx, test_idx in kfold.split(Xsample, ysample):
            X_train, y_train = Xsample[train_idx], ysample[train_idx]
            X_test, y_test = Xsample[test_idx], ysample[test_idx]
            clf1.fit(X_train, y_train)
            scores1.append(clf1.score(X_test, y_test))
            clf2.fit(X_train, y_train)
            scores2.append(clf2.score(X_test, y_test))
        df_X = pd.DataFrame(Xsample)
        df_X = df_X[random.sample(df_X.columns.to_list(), len(df_X.columns))]
        df = pd.concat([df_X, pd.DataFrame({'class': ysample})], axis=1)
        if f not in train_valid:
            df.to_csv('../samples_train/'+f'{f}_{np.mean(scores1)*100:.3f}_{np.mean(scores2)*100:.3f}_{i}.csv',
                      index=False)
        else:
            df.to_csv('../samples_valid/'+f'{f}_{np.mean(scores1)*100:.3f}_{np.mean(scores2)*100:.3f}_{i}.csv',
                      index=False)
        mb.child.comment = f'Sampler'
        count += 1


print("Number of datasets exported:", count)
