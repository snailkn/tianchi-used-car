import numpy as np
import pandas as pd
import logging
import os

from sklearn.model_selection import KFold, ParameterSampler
from sklearn.metrics import mean_absolute_error

from utilitis.feature_manager import FeatureManager
from utilitis.model import XGBModel, to_xgb_dm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(os.path.join('data', 'result', 'log_hyper_search.txt'))
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def date_parser(x):
    try:
        return pd.datetime.strptime(x, "%Y%m%d")
    except:
        return np.nan

    
def modify_feature(df):
    df['carAge'] = (df['creatDate'] - df['regDate']).apply(lambda x: x.days)
    df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', np.nan).astype(float)
    df.loc[df['model'].isin(sparse_model), 'model'] = np.nan

    
def hyper_params_search_cv(hyper_params_iter, tune_iteration, model, idxes, X, y, random_state=0, verbose=True):
    params, metrics, best_boost_round = [], [], []
    for i, param_set in enumerate(ParameterSampler(hyper_params_iter, tune_iteration, random_state=random_state)):
        model.set_hyper_params(**param_set)
        params.append(param_set)
        val_metrics, best_rounds = [], []
        for tr_idx, val_idx in idxes:
            tr_x, tr_y = X.iloc[tr_idx, :], y.iloc[tr_idx]
            val_x, val_y = X.iloc[val_idx, :], y.iloc[val_idx]
            tr_dm, val_dm = to_xgb_dm(tr_x, tr_y), to_xgb_dm(val_x, val_y)
            model.train(tr_dm, val_dm)
            val_metrics.append(mean_absolute_error(model.predict(val_x), val_y))
            best_rounds.append(model.model.best_iteration)
           # print(val_metrics[-1], best_rounds[-1])
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
    test = pd.read_csv('data/used_car_testB_20200421.csv', sep=' ', parse_dates=['regDate', 'creatDate'], date_parser=date_parser)
    
    sparse_model = df.model.value_counts()[df.model.value_counts() < 2000].index
    modify_feature(df)
    modify_feature(test)
    
    df['logPrice'] = df['price'].apply(np.log)
    num_feats = [
        'power', 'kilometer', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12',
        'v_13', 'v_14', 'carAge'
    ]
    bool_feats = ['gearbox', 'notRepairedDamage', 'seller', 'offerType']
    cat_feats = ['model', 'brand', 'bodyType', 'fuelType']
    
    feature_manager = FeatureManager(
        num_features=num_feats + bool_feats, 
        categorical_features=cat_feats,
        num_config={'fillna': 'median'},
        categorical_config={'one_hot': True}
    )
    X_dev = feature_manager.get_model_features(df)
    X_test = feature_manager.transform_feature(test)
    
    kf = KFold(n_splits=10)
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
        }, 100, model, idxes, X_dev, df['logPrice'], random_state=1024
    )
    
    xgb_hyper = xgb_hyper_record[xgb_hyper_record['tag'] == 'best iter'].iloc[0, 2:-1].to_dict()
    boost_round = int(xgb_hyper.pop('best_boost_round'))
    model = XGBModel(num_boost_round=boost_round, seed=1024)
    model.set_hyper_params(**xgb_hyper)
    dev_dm = to_xgb_dm(X_dev, df['logPrice'])
    model.train(dev_dm)
    
    pd.DataFrame(np.exp(model.predict(X_test)), index=test['SaleID'], columns=['price']).to_csv('data/result/baseline-submit.csv')
