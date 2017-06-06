loan-default-prediction
=======================

Description
-----------
The code was written for [Loan Default Prediction Competition at Kaggle](https://www.kaggle.com/c/loan-default-prediction). The code is modified based on https://github.com/songgc/loan-default-prediction.git. 

Dependencies and requirements
-----------------------------
[pandas](https://github.com/pydata/pandas):  version 0.13.1 or later

[scikit learn](https://github.com/scikit-learn/scikit-learn): dev branch with version commit 884889a4cd36e63d53a067d9380dea7724a93ac5 or later

How to run
----------
1. Download data from [Kaggle](https://www.kaggle.com/c/loan-default-prediction)
2. Unzip the train and test csv files to './data' folder and make sure that their names are 'train_v2.csv' and 'test_v2.csv', respectively
3. Run `bash requirement.sh` to install required dependencies 
4. Run `python train_predict_cv.py --data_dir './data' --train_filename 'train_v2.csv' --test_filaname 'test_v2.csv'`
5. The prediction submission-ready csv (submission.csv) will be found at './data' 

To validate the code
----------
1. Run `bash requirement.sh` to install required dependencies 
2. Run `bash run.sh` to train the model



