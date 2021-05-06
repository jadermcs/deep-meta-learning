#!/usr/bin/env python
# coding: utf-8
import os
from pymfe.mfe import MFE
import pandas as pd
import numpy as np
from fastprogress.fastprogress import master_bar, progress_bar
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

seed = 42

path = "../data/"
data = pd.read_csv("../data.csv")
train_df = []
valid_df = []

for index, file in progress_bar(data.iterrows(), total=data.shape[0]):
    df = pd.read_csv(path+file["files"])
    X = df.drop(columns=["class"]).values
    y = df["class"].values
    mfe = MFE(random_state=seed)
    mfe.fit(X, y)
    ft = mfe.extract()
    ft = dict(zip(*ft))
    ft["class"] = file["dt"]
    if file["train"]:
        train_df.append(ft)
    else:
        valid_df.append(ft)
    
train_df = pd.DataFrame(train_df)
valid_df = pd.DataFrame(valid_df)
train_df.to_csv("../mfe.train.csv", index=False)
train_df.to_csv("../mfe.test.csv", index=False)

xtrain = train_df.drop(columns=["class"]).values
ytrain = train_df["class"].values
xtest = valid_df.drop(columns=["class"]).values
ytest = valid_df["class"].values
lg = LGBMRegressor(random_state=seed, objective='l1')

lg.fit(xtrain, ytrain)
mean_absolute_error(ytest, lg.predict(xtest))


print(ytest[:3], lg.predict(xtest)[:3])
