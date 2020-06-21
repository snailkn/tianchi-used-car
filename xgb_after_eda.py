import numpy as np
import pandas as pd
import logging
import os

from category_encoders import LeaveOneOutEncoder

from sklearn.model_selection import KFold, ParameterSampler
from sklearn.metrics import mean_absolute_error

from utilitis.utils import date_parser
from utilitis.feature_manager import FeatureManager
from utilitis.model import XGBModel, to_xgb_dm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(os.path.join('data', 'result', 'log_hyper_search.txt'))
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class MineFeatureManager(FeatureManager):
    def __init__(self, num_config=None, categorical_config=None):
        self.num_features = [
            'power', 'kilometer', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_11', 
            'v_12', 'v_13', 'v_14', 'carAge', 'v_10_1', 'v_10_2', 'v_10_3', 'modelEncode', 
            'regionCodeEncode', 'gearbox', 'notRepairedDamage', 'seller', 'offerType'
        ]
        self.categorical_features = ['model', 'brand', 'bodyType', 'fuelType']
        self.encoded_cates = ['model', 'regionCode']
        self.cate_encoder = LeaveOneOutEncoder(cols=self.encoded_cates)
        self.general_model = None
        super().__init__(self.num_features, self.categorical_features, num_config, categorical_config)

    def _feature_engien(self, features):
        zero_na = {0: np.nan}
        features = features.replace({'power': zero_na, 'v_5': zero_na, 'v_6': zero_na})
        
        features['carAge'] = (features['creatDate'] - features['regDate']).apply(lambda x: x.days)
        features['notRepairedDamage'] = features['notRepairedDamage'].replace('-', np.nan).astype(float)
        
        features.loc[features['power'] > 600, 'power'] = np.nan
        features['power'] = np.log(features['power'])
        features.loc[features['v_7'] > 0.5, 'v_7'] = np.nan
        features.loc[features['v_11'] > 10, 'v_11'] = np.nan
        features.loc[features['v_13'] > 7.5, 'v_13'] = np.nan
        features.loc[features['v_14'] > 7.5, 'v_14'] = np.nan
        
        features.loc[features['v_10'] <= 0, 'v_10_1'] = features.loc[features['v_10'] <= 0, 'v_10']
        features.loc[(features['v_10'] >= 0) & (features['v_10'] < 6), 'v_10_2'] = features.loc[(features['v_10'] >= 0) & (features['v_10'] < 6), 'v_10']
        features.loc[features['v_10'] > 8, 'v_10_3'] = features.loc[features['v_10'] > 8, 'v_10']
        features.loc[~features['model'].isin(self.general_model), 'model'] = np.nan
        return features
    
    def get_model_features(self, features):
        features = features.copy()
        self.general_model = df.model.value_counts()[df.model.value_counts() < 2000].index
        encoded_cate = self.cate_encoder.fit_transform(features[self.encoded_cates], features['logPrice'])
        for cate in self.encoded_cates:
            features[cate + 'Encode'] = encoded_cate[cate]
        features = self._feature_engien(features)
        return super().get_model_features(features)
    
    def transform_feature(self, features):
        features = features.copy()
        encoded_cate = self.cate_encoder.transform(features[self.encoded_cates])
        for cate in self.encoded_cates:
            features[cate + 'Encode'] = encoded_cate[cate]
        features = self._feature_engien(features)
        return super().get_model_features(features)


def hyper_params_search_cv(hyper_params_iter, tune_iteration, model, idxes, df, y, random_state=0, verbose=True):
    params, metrics, best_boost_round = [], [], []
    for i, param_set in enumerate(ParameterSampler(hyper_params_iter, tune_iteration, random_state=random_state)):
        model.set_hyper_params(**param_set)
        params.append(param_set)
        val_metrics, best_rounds = [], []
        for tr_idx, val_idx in idxes:
            tr_df, tr_y = df.iloc[tr_idx, :], y.iloc[tr_idx]
            val_df, val_y = df.iloc[val_idx, :], y.iloc[val_idx]
            sub_manager = MineFeatureManager(num_config={'missing_indicator': False}, categorical_config={'missing_indicator': False})
            tr_x = sub_manager.get_model_features(tr_df)
            val_x = sub_manager.transform_feature(val_df)
            
            tr_dm, val_dm = to_xgb_dm(tr_x, tr_y), to_xgb_dm(val_x, val_y)
            model.train(tr_dm, val_dm, verbose_eval=True)
            val_metrics.append(mean_absolute_error(model.predict(val_x), val_y))
            best_rounds.append(model.model.best_iteration)
        best_boost_round.append(np.mean(best_rounds))
        metrics.append(np.mean(val_metrics))
        if verbose:
            print('[', i, '] ', param_set, 'mae: ', metrics[-1], ' best_boost_round: ', best_boost_round[-1])
            logger.info('[{}] {} mae: {} best_boost_round: {}'.format(i, param_set, metrics[-1], best_boost_round[-1]))
    params_idx = np.argmin(metrics)
    cols = ['eta', 'max_depth', 'subsample', 'colsample_bytree', 'gamma', 'lambda']
    df = pd.DataFrame(params).reindex(columns=cols)
    df.insert(0, 'mae', pd.Series(metrics))
    df['best_boost_round'] = np.array(best_boost_round)
    df.at[params_idx, 'tag'] = 'best iter'
    print('best iter', params[params_idx], metrics[params_idx])
    return df


if __name__ == '__main__':
    df = pd.read_csv('data/used_car_train_20200313.csv', sep=' ', parse_dates=['regDate', 'creatDate'], date_parser=date_parser)
    print(df.shape)
    df['logPrice'] = np.log(df['price'])

    test = pd.read_csv('data/used_car_testB_20200421.csv', sep=' ', parse_dates=['regDate', 'creatDate'], date_parser=date_parser)
    print(test.shape)
    
    feature_manager = MineFeatureManager(num_config={'missing_indicator': False}, categorical_config={'missing_indicator': False})
    X_dev = feature_manager.get_model_features(df)
    X_test = feature_manager.transform_feature(test)
    print(X_dev.shape, X_test.shape)
    
    kf = KFold(n_splits=5)
    idxes = list(kf.split(X_dev))
    
    model = XGBModel(seed=1024)
    xgb_hyper_record = hyper_params_search_cv(
        {
            'eta': [0.3],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.5, 0.7, 0.9],
            'colsample_bytree': [0.5, 0.7, 0.9],
            'gamma': [0, 0.01, 0.03, 0.1, 0.3],
            'lambda': [0.1, 0.3, 1]
        }, 100, model, idxes, df, df['logPrice'], random_state=1024
    )
    
    xgb_hyper_record.to_csv('data/result/after-eda-tuning-record.csv')
    xgb_hyper = xgb_hyper_record[xgb_hyper_record['tag'] == 'best iter'].iloc[0, 2:-1].to_dict()
    boost_round = int(xgb_hyper.pop('best_boost_round'))
    model = XGBModel(num_boost_round=boost_round, seed=1024)
    model.set_hyper_params(**xgb_hyper)
    dev_dm = to_xgb_dm(X_dev, df['logPrice'])
    model.train(dev_dm)
    
    pd.DataFrame(np.exp(model.predict(X_test)), index=test['SaleID'], columns=['price']).to_csv('data/result/after-eda-submit.csv')
