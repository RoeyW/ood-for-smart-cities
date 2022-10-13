## DATASET ##
We conduct our experiments on two kind of datasets:
1. bike datasets: including N.Y.C. bike dataset and Washington bike dataset.
2. water quality dataset

we first pre-process the three datasets to split each of them into spatial-temporal shift environments.

For N.Y.C. dataset, we split them into different timespans. For example, nyc.npz is original dataset. nycl6s0 is extracted from nyc.npz, whose historial data length is 6. And 's0' means the tempporal gap between this dataset and validation dataset is 0 days. In other word, nycl6s0.npz is test dataset which is closest to the validation dataset.

For water quality dataset, we got data from 6 different regions. So we directly split them in temporal dimension.

---
## CODE ##
We use python 3.6.x and tensorflow 2.0 to implement our model.
The details of folds are as follows:
1. "dataprocess": contains the codes for preprocessing data. seg_data.py is used to split the dataset in temporal and spatial dimensions.

2. "Model"： contians functions for implementing the model as shown in following:
    
    2.1  run 'PIRM.py' or 'PIRM_wph.py', logs and results will generate when the file has finised running. 'PIRM.py' is used to train and evaluate the model using bike datasets. And 'PIRM_wph.py' is used to train and evaluate the model using water quality dataset.

    2.2 'EnvconfounderIRM.py' is used to define the architecture of our model.

    2.3 'losses.py' contains some loss functions, e.g. MSE, contrastive-base loss.

    2.4 'load_data'  is used to load bike or water quality dataset.

