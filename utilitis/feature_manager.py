"""
created by snailkn@2019-12-12
"""
import numbers
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class FeatureFactory(object):
    @staticmethod
    def get_missing_indicator(feature):
        return feature.isnull().astype(int).rename('_'.join([feature.name, 'missing']))

    def _build_features(self, feature):
        raise NotImplementedError

    def transform_feature(self, feature):
        return self._build_features(feature)


class NumFeatureFactory(FeatureFactory):
    def __init__(self):
        self.filling_num = None
        self.missing_indicator = None
        self.scaler = None

    def _set_filling_num(self, feature, fillna):
        if fillna is not None:
            if isinstance(fillna, numbers.Number):
                self.filling_num = fillna
            else:
                self.filling_num = feature.agg(fillna)

    def _build_features(self, feature):
        features = []
        # 处理空值
        if self.filling_num is None:
            out_feature = feature
        else:
            out_feature = feature.fillna(self.filling_num)
        # 数值归一化
        if self.scaler:
            features.append(pd.DataFrame(
                self.scaler.transform(pd.DataFrame(out_feature)), index=feature.index, columns=[feature.name]))
        else:
            features.append(out_feature)
        # 添加缺失标记
        if self.missing_indicator:
            features.append(self.get_missing_indicator(feature))
        return pd.concat(features, axis=1)

    def get_model_features(self, feature, missing_indicator=True, fillna=None, scaling=False):
        self.missing_indicator = missing_indicator and feature.isna().sum()
        self._set_filling_num(feature, fillna)
        if scaling:
            self.scaler = MinMaxScaler()
            self.scaler.fit(pd.DataFrame(feature))
        return self._build_features(feature)


class CategoricalFeatureFactory(FeatureFactory):
    def __init__(self):
        self.missing_indicator = None
        self.one_hot_features = None

    def _build_features(self, feature):
        features = []
        if self.one_hot_features is not None:
            missing_features = set(self.one_hot_features) - set(feature.dropna().drop_duplicates())
            missing_features_df = pd.DataFrame(
                pd.np.zeros((len(feature), len(missing_features))),
                index=feature.index,
                columns=['{}_{}'.format(feature.name, x) for x in missing_features]
            )
            feature_names = ['{}_{}'.format(feature.name, x) for x in self.one_hot_features]
            features.append((pd.get_dummies(feature, prefix=feature.name).join(missing_features_df))[feature_names])
        else:
            features.append(feature)
        if self.missing_indicator:
            features.append(self.get_missing_indicator(feature))
        return pd.concat(features, axis=1)

    def get_model_features(self, feature, missing_indicator=True, one_hot=False):
        self.missing_indicator = missing_indicator and feature.isna().sum()
        if one_hot:
            self.one_hot_features = feature.dropna().drop_duplicates()
        return self._build_features(feature)


class FeatureManager(object):
    def __init__(self, num_features=None, categorical_features=None, num_config=None, categorical_config=None):
        if num_features is None and categorical_features is None:
            raise ValueError('未声明任何特征项')
        self.features = {}
        for num_feature in num_features:
            self.features[num_feature] = (NumFeatureFactory(), num_config or {})
        for categorical_feature in categorical_features:
            self.features[categorical_feature] = (CategoricalFeatureFactory(), categorical_config or {})

    def get_model_features(self, features):
        out_features = []
        for feature, (feature_factory, config) in self.features.items():
            out_features.append(feature_factory.get_model_features(features[feature], **config))
        return pd.concat(out_features, axis=1)

    def transform_feature(self, features):
        out_features = []
        for feature, (feature_factory, config) in self.features.items():
            out_features.append(feature_factory.transform_feature(features[feature]))
        return pd.concat(out_features, axis=1)
