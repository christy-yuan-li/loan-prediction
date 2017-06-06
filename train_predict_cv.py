
import pandas as pd
import os
import sys
from functools import partial
import models
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import lightgbm as lgb
import numpy as np
from sklearn import metrics
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_filename', type=str, help='file path to the train dataset')
parser.add_argument('--test_filename', type=str, help='file path to the test dataset')
parser.add_argument('--data_dir', type=str, help='folder path of data')
args = parser.parse_args()

read_csv = partial(pd.read_csv, na_values=['NA', 'na'], low_memory=False)


# ==========
# lightGBM
# ==========

def lightgbm_clf(train, val, test, path):
    # input train, val, test is DataFrame objects
    ### train
    pipeline = get_lightgbm_pipeline()
    print(len(train.values), len(train['loss'].values))
    pipeline.fit(train, train['loss'].values)

    ### test and save
    ypred = pipeline.predict(test)
    columns = ['c' + str(i) for i in range(1, 2)]
    ypred = pd.DataFrame(ypred, columns=columns)
    filepath = os.path.join(path, 'test_ypred.csv')
    ypred.to_csv(filepath, index=False)

    ### val and save
    valpred = pipeline.predict(val)
    valpredbinary = valpred > 0.6
    vallabels = val['loss'].values
    vallabelsbinary = vallabels > 0.6

    ### evaluate
    print('accuracy_score: {:g}'.format(metrics.accuracy_score(vallabelsbinary, valpredbinary)))
    print('auc: {:g}'.format(metrics.auc(vallabelsbinary, valpredbinary, reorder=True)))
    print('precision_score: {:g}'.format(metrics.precision_score(vallabelsbinary, valpredbinary)))
    print('recall_score: {:g}'.format(metrics.recall_score(vallabelsbinary, valpredbinary)))
    print('f1_score: {:g}'.format(metrics.f1_score(vallabelsbinary, valpredbinary)))

    ### save
    valpred = pd.DataFrame(valpred, columns=columns)
    valpred.to_csv(os.path.join(path, 'val_ypred.csv'), index=False)
    vallabels = pd.DataFrame(vallabels, columns=['label'])
    vallabels.to_csv(os.path.join(path, 'val_ylabels.csv'), index=False)


def get_lightgbm_pipeline():
    param = {'num_leaves': 31, 'num_trees': 100, 'objective': 'binary'}
    param['metric'] = ['auc']
    clf = models.LightGBMClassifier(lgb, param, num_round=30)
    steps = [('features', models.FeatureSelector()),
             ('Impute', Imputer(strategy='median')),
             ('scaler', StandardScaler()),
             ('clf', clf)]
    return Pipeline(steps)


# ============================
# GradientBoostingClassifier
# ============================

def default_clf(train, val, test, path):
    ### train
    pipeline = get_clf_pipeline()
    pipeline.fit(train, train['loss'].values)

    ### test
    proba = pipeline.predict(test)

    ### save probability prediction
    _, n = proba.shape
    columns = ['c' + str(i) for i in range(n)]
    proba = pd.DataFrame(proba, columns=columns)
    filepath = os.path.join(path, 'test_proba.csv')
    proba.to_csv(filepath, index=False)

def get_clf_pipeline():
    clf = models.DefaultClassifier(GradientBoostingClassifier(loss='deviance', learning_rate=0.01, n_estimators=3000,
                                                              subsample=0.6, min_samples_split=12, min_samples_leaf=12,
                                                              max_depth=6, random_state=1357, verbose=0))
    steps = [('features', models.FeatureSelector()),  # input is (DataFrame, y), output is (numpy array, y)
             ('Impute', Imputer(strategy='median')),  # input is (numpy array, y), output is (numpy array, y)
             ('scaler', StandardScaler()),  # input is (numpy array, y), output is (numpy array, y)
             ('clf', clf)]  # input is (numpy array, y)
    return Pipeline(steps)


# ============================
# GradientBoostingRegressor
# ============================

def loss_reg(train, val, test, path):
    # input train, val, test is DataFrame objects
    pipeline = get_reg_pipeline()
    pipeline.fit(train, train['loss'].values)
    pred_loss = pipeline.predict(test)
    valpred_loss = pipeline.predict(val)

    prob_default_test = pd.read_csv(os.path.join(path, 'test_ypred.csv'))['c1']  ## get probability of weather it is default or not
    prob_default_val = pd.read_csv(os.path.join(path, 'val_ypred.csv'))['c1']  ## get probability of weather it is default or not
    labels_default_val = pd.read_csv(os.path.join(path, 'val_ylabels.csv'))['label']  ## get probability of weather it is default or not

    print('mean_squared_error: {:g}'.format(metrics.mean_squared_error(labels_default_val, valpred_loss)))

    submission = pd.DataFrame()
    submission['id'] = test['id']
    submission['loss'] = pred_loss * (prob_default_test.values > 0.6)
    submission.to_csv(os.path.join(path, 'submission.csv'), index=False)

    # val_loss = valpred_loss * (prob_default_val.values > 0.6)


def get_reg_pipeline():
    clf = models.LightGBMRegressor(lgb, {}, num_round=30)
    steps = [('features', models.FeatureSelector()),
             ('Impute', Imputer(strategy='median')),
             ('scaler', StandardScaler()),
             ('clf', clf)]
    return Pipeline(steps)


# def get_reg_pipeline():
#     clf = models.PartialRegressor(GradientBoostingRegressor(loss='ls', learning_rate=0.0075, n_estimators=5000,
#                                                             subsample=0.5, min_samples_split=20, min_samples_leaf=20,
#                                                             max_leaf_nodes=30, random_state=9753, verbose=1))
#     steps = [('features', models.FeatureSelector()),
#              ('Impute', Imputer(strategy='median')),
#              ('scaler', StandardScaler()),
#              ('clf', clf)]
#     return Pipeline(steps)


if __name__ == '__main__':
    path = args.data_dir
    filenames = [args.train_filename, args.test_filename]
    train_file, test_file = [os.path.join(path, fname) for fname in filenames]
    train_all = read_csv(train_file)
    msk = np.random.rand(len(train_all)) < 0.8
    train = train_all[msk]
    val = train_all[~msk]
    test = read_csv(test_file)

    print('===>>> default classifier training...')
    lightgbm_clf(train, val, test, path)
    print('===>>> loss regression training...')
    loss_reg(train, val, test, path)
