# -*-encoding: utf-8 -*-
"""
@time: 2020/6/8
@author: bingli
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from woe_encoder.cat_utils import (calculate_bad_rate_diff_for_bin_df,
                                   calculate_chi2_for_bin_df,
                                   initialize_bins,
                                   locate_index,
                                   process_special_values,
                                   update_bin_df)
from woe_encoder.utils import if_monotonic, calculate_woe_iv


class CategoryWOEEncoder(BaseEstimator, TransformerMixin):
    """A sklearn-compatible woe encoder for categorical features.

    这里提供了 2 种分箱方法：基于卡方的分箱和基于坏样本率差异最大化的分箱.

    Parameters
    ----------
    col_name: str
        需要转换的特征名列表
    target_col_name: str
        目标特征名
    max_bins: int
        最大箱数
    bin_pct_threshold: float, default to 0.05
        分箱后每一箱含有的样本量比列（相对于全体样本）
    woe_method: str, "chi2" (default) or "bad_rate"
        bin 的合并原则。"chi2" 表示按卡方值，"bad_rate" 表示坏样本率
    special_value_list: list, default to None
        特征中需要特殊对待的值。like: [s_v_1, s_v_2]
    missing_value:
        fill 缺失值
    need_monotonic: boolean, default to False
        特征转换后是否需要满足单调性
    u: boolen, default to False
        特征转换后是否需要满足 U 型
    value_order_dict: dict, default to None
        离散有序特征值的顺序
    regularization: float, default to 1.0
        计算 woe 时加入正则化

    Attributes
    ----------
    bin_df_: pd.DataFrame
        分箱后结果，含有每个 bin 的统计值
    iv_: float
        分箱后该特征的 IV 值
    bin_woe_mapping_: dict
        bin 与 woe 值的对应关系

    Examples
    --------
    from category_encoders import WOEEncoder
    from sklearn.datasets import load_boston

    bunch = load_boston()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    y = bunch.target > 22.5
    df['y'] = y

    col = 'RAD'
    my_encoder = CategoryWOEEncoder(col, 'y', max_bins=100, bin_pct_threshold=0)
    df_my = my_encoder.fit_transform(df)
    print(df_my[col+'_woe'])

    # 对比 sklearn_contrib，两者的结果应该相同
    sklearn_encoder = WOEEncoder(cols=[col]).fit(df, y)
    df_sklearn = sklearn_encoder.transform(df)
    print(df_sklearn[col])
    """
    def __init__(self,
                 col_name: str,
                 target_col_name: str,
                 max_bins=10,
                 bin_pct_threshold=0.05,
                 woe_method='chi2',  # "bad_rate"
                 special_value_list=None,
                 missing_value=None,
                 need_monotonic=None,
                 u=None,
                 value_order_dict=None,
                 regularization=1.0):
        self.col_name = col_name
        self.target_col_name = target_col_name
        self.max_bins = max_bins
        self.bin_pct_threshold = bin_pct_threshold
        self.woe_method = woe_method
        self.special_value_list = special_value_list
        self.missing_value = missing_value
        self.need_monotonic = need_monotonic
        self.u = u
        self.value_order_dict = value_order_dict
        self.regularization = regularization

    def inspect(self):
        if not isinstance(self.col_name, str):
            raise ValueError("")
        if not isinstance(self.target_col_name, str):
            raise ValueError("")
        if not isinstance(self.max_bins, int):
            raise ValueError("")
        if not isinstance(self.bin_pct_threshold):
            raise ValueError("")
        if not 0. <= self.bin_pct_threshold < 1.:
            raise ValueError("")
        if self.woe_method not in ('chi2', 'bad_rate'):
            raise ValueError("")
        if not isinstance(self.special_value_list, list):
            raise ValueError("")
        if not isinstance(self.need_monotonic, bool):
            raise ValueError("")
        if not isinstance(self.u, bool):
            raise ValueError("")
        if not isinstance(self.value_order_dict, dict):
            raise ValueError("")
        if not isinstance(self.regularization, (float, int)):
            raise ValueError("")

    def _train(self, df, bin_num_threshold) -> pd.DataFrame:
        # 初始化分箱
        bin_df = initialize_bins(
            df, self.col_name, self.target_col_name, self.value_order_dict)

        # 如果合并原则按照卡方值，则需要计算每 2 个相邻 bin 的卡方值
        if self.woe_method == 'chi2':
            calculator_between_bins = calculate_chi2_for_bin_df
        # 如果合并原则按照坏样本率，则需要计算每 2 个相邻 bin 的坏样本率差值
        else:  # 'bad_rate'
            calculator_between_bins = calculate_bad_rate_diff_for_bin_df
        values_calculated = calculator_between_bins(bin_df)

        # condition 1: bin 的个数不大于 max_bins
        while len(bin_df) > self.max_bins:
            index = np.argmin(values_calculated)
            bin_df = update_bin_df(bin_df, index)
            values_calculated = calculator_between_bins(bin_df)

        # condition 2: 每个 bin 的 bad_rate 不为 0 / 1
        i = 0
        while i < len(bin_df):
            if bin_df.iloc[i, 4] in (0, 1):
                index = locate_index(bin_df, values_calculated, i)
                bin_df = update_bin_df(bin_df, index)
                values_calculated = calculator_between_bins(bin_df)
                i -= 1
            i += 1

        # condition 3: 每个 bin 中的样本数不能少于阈值
        while np.min(bin_df['bin_num']) < bin_num_threshold:
            index = np.argmin(bin_df['bin_num'])
            index = locate_index(bin_df, values_calculated, index)
            bin_df = update_bin_df(bin_df, index)
            values_calculated = calculator_between_bins(bin_df)

        # condition 4: (离散有序型特征) bin 是否满足单调性
        if self.value_order_dict:
            if self.need_monotonic:
                bad_rates = bin_df.iloc[:, 4]
                while not if_monotonic(bad_rates, u=self.u):
                    index = np.argmin(values_calculated)
                    bin_df = update_bin_df(bin_df, index)
                    values_calculated = calculator_between_bins(bin_df)

        return bin_df

    def fit(self, x):
        df = x[[self.col_name, self.target_col_name]].copy()

        raw_length = len(df)
        bin_num_threshold = raw_length * self.bin_pct_threshold

        # missing_value 也相当于特殊值
        special_value_flag = False
        if self.missing_value is not None:
            # Fill missing values with specified value.
            df[self.col_name] = df[self.col_name].fillna(self.missing_value)
            if self.special_value_list is not None:
                self.special_value_list.append(self.missing_value)
            else:
                self.special_value_list = [self.missing_value]
            special_value_flag = True
        else:
            if self.special_value_list is not None:
                special_value_flag = True

        if special_value_flag:  # 处理特殊值
            df, stats = process_special_values(
                df, self.col_name, self.target_col_name,
                self.special_value_list)

        bin_df = self._train(df, bin_num_threshold)
        if special_value_flag:
            bin_df.append(stats)

        # Calculate WOE and IV
        bin_df[['woe', 'iv']] = calculate_woe_iv(
            bin_df, bad_col='bad_num', good_col='good_num',
            regularization=self.regularization)

        # 值与对应 WOE 的映射, 方便 transform
        bin_woe_mapping = {}
        for bins, woe in zip(bin_df[self.col_name], bin_df['woe']):
            for bin_v in bins:
                bin_woe_mapping[bin_v] = woe

        self.bin_df_ = bin_df
        self.iv_ = bin_df['iv'].sum()
        self.bin_woe_mapping_ = bin_woe_mapping
        return self

    def transform(self, X):
        new_x = X.copy()
        new_x[self.col_name + '_woe'] = new_x[self.col_name].map(
            self.bin_woe_mapping_)
        return new_x
