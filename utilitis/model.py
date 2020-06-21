import copy
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn import metrics


def to_xgb_dm(x, y):
    return xgb.DMatrix(x.values, label=y.values)


class XGBModel(object):
    def __init__(self, num_boost_round=500, early_stopping_rounds=50, seed=0, nthread=None):
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.param = {
            'objective': 'reg:squarederror',
            'gpu_id': 0,
            'tree_method': 'gpu_hist',
            'booster': 'dart',
            'seed': seed
        }
        if nthread:
            self.param['nthread'] = nthread

    def set_hyper_params(self, **kwargs):
        self.param.update(kwargs)

    def train(self, tr, val=None, verbose_eval=False):
        if val:
            evals = [(tr, 'train'), (val, 'eval')]
            self.model = xgb.train(
                self.param, tr, num_boost_round=self.num_boost_round,
                early_stopping_rounds=self.early_stopping_rounds, evals=evals, verbose_eval=verbose_eval
            )
        else:
            self.model = xgb.train(self.param, tr, num_boost_round=self.num_boost_round)

    def predict(self, X):
        X = xgb.DMatrix(X.values)
        return self.model.predict(X)
