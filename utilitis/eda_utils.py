import numpy as np
from scipy.stats import anderson, ttest_ind, ranksums, f_oneway, kruskal, chi2_contingency, normaltest
import pandas as pd

from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import _rebuild
from statsmodels.stats.multitest import multipletests

_rebuild()
plt.rcParams['font.family']=['Microsoft YaHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用于正常显示负号


def check_norm_distribute(*args) -> bool:
    anderson_result = [anderson(x, 'norm') for x in args]  # Anderson-Darling test for normality returns A2 (the test
    # statistic), the critical values and the significance levels for the corresponding critical values in percents (
    # 15%, 10%, 5%, 2.5%, 1%).
    sigs = [sum(result[1] < result[0]) < 3 for result in anderson_result]
    result = False
    if all(sigs):
        result = True
    return result


def get_num_p_value(obs_list) -> str:
    if len(obs_list) == 2:
        if check_norm_distribute(*obs_list):
            method = 'T-test'
            _, p_value = ttest_ind(obs_list[0], obs_list[1])
        else:
            method = 'Wilcoxon rank-sum'
            _, p_value = ranksums(obs_list[0], obs_list[1])
    else:
        if check_norm_distribute(*obs_list):
            method = 'One-way ANOVA'
            _, p_value = f_oneway(*obs_list)
        else:
            method = 'Kruskal-Wallis H-test'
            _, p_value = kruskal(*obs_list)
    sig_rank = ''
    if 0.01 <= p_value < 0.05:
        sig_rank = '*'
    elif p_value < 0.01:
        sig_rank = '**'
    if p_value < 0.0001:
        result = '<0.0001{} ({})'.format(sig_rank, method)
    else:
        result = '{:.4f}{} ({})'.format(float(p_value), sig_rank, method)
    return result, p_value


def get_cat_p_value(obs_list, tot_list) -> str:
    obs_list = np.array(obs_list)
    tot_list = np.array(tot_list)
    non_obs_list = tot_list - obs_list
    if any(obs_list == 0) or any(non_obs_list == 0):
        result = '/'
    else:
        obs = np.array([obs_list, tot_list - obs_list])
        _, p_value, _, _ = chi2_contingency(obs, correction=False)
        sig_rank = ''
        if 0.01 <= p_value < 0.05:
            sig_rank = '*'
        elif p_value < 0.01:
            sig_rank = '**'
        if p_value < 0.0001:
            result = '<0.0001{}'.format(sig_rank)
        else:
            result = '{:.4f}{}'.format(float(p_value), sig_rank)
    return result


def get_cat_feat_p_value(group_cat_count):
    df = pd.DataFrame(group_cat_count).fillna(0)
    _, p_value, _, _ = chi2_contingency(df.values)
    return '{:.4f}{}{}'.format(float(p_value), '*' * (p_value < 0.05), '**' * (p_value < 0.01)), p_value


def series_to_mean_std(s):
    _,normal_p = normaltest(s)
    return '{:.1f}±{:.1f} ({:.1f}, {:.1f}-{:.1f}) [{}]'.format(
        s.mean(), s.std(), s.median(), s.quantile(0.25), s.quantile(0.75), normal_p
    )


def series_to_missing_rate(s):
    return '({}) {:.1%}'.format(s.isna().sum(), s.isna().mean())


def get_cat_rate(counter, cat_name):
    return '({}) {:.1%}'.format(counter.get(cat_name, 0), counter.get(cat_name, 0) / sum(counter.values()))


def create_characteristic_table_without_p(data_df: pd.DataFrame, num_cols: list, num_name_dict: dict, cat_cols: list,
                                          cat_name_dict: dict, cat_mi: str, label_col: str, label_name_dict: dict):
    n = len(data_df)
    label_group_dict = {label_name_dict[code]: data_df[label_col] == code for code in label_name_dict}
    first_col_names = ['Characteristic', 'whole(n={})'.format(n)] + \
                      ['{} (n={})'.format(group_name, sum(label_group_dict[group_name])) for group_name in
                       label_group_dict]
    # 'Characteristic', 'Non-SSI (n=100)', 'SSI (n=100)'
    characteristic_table = pd.DataFrame(columns=first_col_names)

    for num_col in num_cols:
        characteristic_table = characteristic_table.append(
            dict(zip(first_col_names, [num_name_dict[num_col]] + [''] * (len(label_group_dict) + 2))),
            ignore_index=True)  # 'feature name', '', '', '', ''
        characteristic_table = characteristic_table.append(
            dict(zip(first_col_names, ['-Mean±SD (Median, IQR 25%-75%)', series_to_mean_std(data_df[num_col])] +
                     [series_to_mean_std(data_df[num_col][label_group_dict[group_name]]) for group_name in
                      label_group_dict])),
            ignore_index=True)
        # '-Mean±SD (Median, IQR 25%-75%)', '53.987±15.242 (55.0, 44.0-64.0)','63.296±16.522 (65.0, 56.0-73.0)'
        characteristic_table = characteristic_table.append(
            dict(zip(first_col_names, ['-Missing rate % (n)', series_to_missing_rate(data_df[num_col])] +
                     [series_to_missing_rate(data_df[num_col][label_group_dict[group_name]]) for group_name in
                      label_group_dict])),
            ignore_index=True)  # '-Missing rate % (n)', '0.293% (65)', '0.0% (0)'

    for cat_col in cat_cols:
        cat_series = data_df[cat_col].fillna(cat_mi)
        total = Counter(cat_series)
        group_cat_count = {group_name: Counter(cat_series[label_group_dict[group_name]]) for group_name in
                           label_group_dict}
        characteristic_table = characteristic_table.append(
            dict(zip(first_col_names, [cat_name_dict[cat_col]] + [''] * (len(label_group_dict) + 1))),
            ignore_index=True)  # 'feature name', '', '', ''
        for cat_name in total:
            if cat_name != cat_mi:
                characteristic_table = characteristic_table.append(dict(zip(
                    first_col_names,
                    ['n (-{} % )'.format(cat_name), get_cat_rate(total, cat_name)] +
                    [get_cat_rate(group_cat_count[group_name], cat_name) for group_name in label_group_dict]
                )), ignore_index=True)
                # '-GA % (n)', '88.361% (19618)', '94.836% (202)'
        characteristic_table = characteristic_table.append(dict(zip(
            first_col_names,
            ['-Missing rate % (n)', get_cat_rate(total, cat_mi)] +
            [get_cat_rate(group_cat_count[group_name], cat_mi) for group_name in label_group_dict]
        )), ignore_index=True)

    return characteristic_table


def create_characteristic_table(data_df: pd.DataFrame, num_cols: list, num_name_dict: dict, cat_cols: list,
                                cat_name_dict: dict, cat_mi: str, label_col: str,
                                label_name_dict: dict):
    n = len(data_df)
    label_group_dict = {label_name_dict[code]: data_df[label_col] == code for code in label_name_dict}
    first_col_names = ['Characteristic', 'whole(n={})'.format(n)] + \
                      ['{} (n={})'.format(group_name, sum(label_group_dict[group_name])) for group_name in label_group_dict] + \
                      ['P value']
    # 'Characteristic', 'Non-SSI (n=100)', 'SSI (n=100)', 'P value'
    characteristic_table = pd.DataFrame(columns=first_col_names)
    p_values = []

    for num_col in num_cols:
        characteristic_table = characteristic_table.append(
            dict(zip(first_col_names, [num_name_dict[num_col]] + [''] * (len(label_group_dict) + 2))),
            ignore_index=True)  # 'feature name', '', '', '', ''
        p_value_str, p_value = get_num_p_value([data_df[num_col][label_group_dict[group_name]].dropna() for group_name in label_group_dict])

        characteristic_table = characteristic_table.append(
            dict(zip(first_col_names, ['-Mean±SD (Median, IQR 25%-75%)', series_to_mean_std(data_df[num_col])] +
                     [series_to_mean_std(data_df[num_col][label_group_dict[group_name]]) for group_name in label_group_dict] +
                     [p_value_str])),
            ignore_index=True)
        p_values.append((num_col, p_value))
        # '-Mean±SD (Median, IQR 25%-75%)', '53.987±15.242 (55.0, 44.0-64.0)','63.296±16.522
        # (65.0, 56.0-73.0)', '0.034* (T-test)'
        characteristic_table = characteristic_table.append(
            dict(zip(first_col_names, ['-Missing rate % (n)', series_to_missing_rate(data_df[num_col])] +
                     [series_to_missing_rate(data_df[num_col][label_group_dict[group_name]]) for group_name in label_group_dict] +
                     [get_cat_p_value(
                [data_df[num_col][label_group_dict[group_name]].isna().sum() for group_name in label_group_dict],
                [sum(label_group_dict[group_name]) for group_name in label_group_dict])])),
            ignore_index=True)  # '-Missing rate % (n)', '0.293% (65)', '0.0% (0)', '0.034*'

    for cat_col in cat_cols:
        cat_series = data_df[cat_col].fillna(cat_mi)
        total = Counter(cat_series)
        group_cat_count = {group_name: Counter(cat_series[label_group_dict[group_name]]) for group_name in
                           label_group_dict}
        p_value_str, p_value = get_cat_feat_p_value(group_cat_count)
        characteristic_table = characteristic_table.append(
            dict(zip(first_col_names, [cat_name_dict[cat_col]] +
                     [''] * (len(label_group_dict) + 1) +
                     [p_value_str])),
            ignore_index=True)  # 'feature name', '', '', ''
        p_values.append((cat_col, p_value))
        for cat_name in total:
            if cat_name != cat_mi:
                characteristic_table = characteristic_table.append(dict(zip(
                    first_col_names,
                    ['-{} % (n)'.format(cat_name), get_cat_rate(total, cat_name)] +
                    [get_cat_rate(group_cat_count[group_name], cat_name) for group_name in label_group_dict] +
                    [get_cat_p_value(
                        [group_cat_count[group_name].get(cat_name, 0) for group_name in label_group_dict],
                        [sum(label_group_dict[group_name]) - group_cat_count[group_name].get(cat_mi, 0) for group_name in label_group_dict])]
                )), ignore_index=True)
                # '-GA % (n)', '88.361% (19618)', '94.836% (202)', '0.034*'
        characteristic_table = characteristic_table.append(dict(zip(
            first_col_names,
            ['-Missing rate % (n)', get_cat_rate(total, cat_mi)] +
            [get_cat_rate(group_cat_count[group_name], cat_mi) for group_name in label_group_dict] +
            [get_cat_p_value(
                [group_cat_count[group_name].get(cat_mi, 0) for group_name in label_group_dict],
                [sum(label_group_dict[group_name]) for group_name in label_group_dict])]
        )), ignore_index=True)

    dfp = pd.DataFrame(p_values, columns=['feature', 'p_value'])
    _, pval_corrected, _, _ = multipletests(dfp['p_value'], method='fdr_bh', returnsorted=False)
    dfp['corrected_p_value'] = pval_corrected

    return characteristic_table, dfp


def generate_img(raw_df, num_feature, cat_feature, label_col, path, file_name):
    assert raw_df[label_col].isna().sum() == 0
    label_counts = raw_df[label_col].value_counts()
    labels = list(label_counts.index)
    n = len(num_feature)
    f, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
    for i, name in enumerate(num_feature):
        sns.boxplot(raw_df[name], ax=axes[i, 0])
        sns.distplot(raw_df[name].dropna(), hist=True, ax=axes[i, 1])
        sns.boxplot(x=name, y=label_col, data=raw_df, ax=axes[i, 2])
        for label in labels:
            sns.distplot(raw_df[raw_df[label_col] == label][name].dropna(), hist=True, label=label, ax=axes[i, 3])
    f.savefig(path + '/' + file_name + '-num-fig.png')

    n = len(cat_feature)
    f, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    for i, name in enumerate(cat_feature):
        sns.countplot(x=raw_df[name].fillna('missing'), ax=axes[i, 0])
        sns.countplot(x=raw_df[name].fillna('missing'), hue=raw_df[label_col], ax=axes[i, 1])
        group = raw_df[[name, label_col]].fillna('missing').groupby([name, label_col])[name].count() \
                    .unstack(name) \
                    .reindex(labels) \
                    .apply(lambda x: x / pd.Series(label_counts, index=labels)) \
                    .stack() \
                    .reset_index()
        sns.barplot(x=name, y=0, hue=label_col, data=group, ci=None, ax=axes[i, 2])
    f.savefig(path + '/' + file_name + '-cat-fig.png')
