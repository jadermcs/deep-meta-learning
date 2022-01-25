Beyonder: Instace spaces through deep learned representations
---



```sh
python base_data.py
```

```sh
python baseline.py
```

```sh
python beyonder.py
```


## TODO:
- apply a supervised deep representation and compare with MFE supervised
- initially test only on classical suite than extend to all classification datasets
- explore other data augmentation (noise; replace; data aug based GAN)
- target encoder
- regress for bounded target like [this](https://stats.stackexchange.com/questions/11985/how-to-model-bounded-target-variable) or [this](https://stackoverflow.com/questions/51693567/best-way-to-bound-outputs-from-neural-networks-on-reinforcement-learning)
- add more algorithms for regression with multitask
- finetune for tree depth, svm kernel, etc
- selfsupervision for data imputation (like in TabNet)
- finetune for best pre-processing pipeline
- inspect attention plots(?)
- evaluate a better padding mode for the transformer, maybe reflect?
- compare baseline also with a dummy regressor (always predict the mean of the training set)
