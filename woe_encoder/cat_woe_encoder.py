# -*-encoding: utf-8 -*-
"""
@time: 2020/6/8
@author: bingli
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from woe_encoder.utils_for_cat import (calculate_bad_rate_diff_for_bin_df,
                                       calculate_chi2_for_bin_df,
                                       initialize_bins,
                                       locate_index,
                                       process_special_values,
                                       update_bin_df)
from woe_encoder.utils import if_monotonic, calculate_woe_iv


class CategoryWOEEncoder(BaseEstimator, TransformerMixin):
    """A sklearn-compatible woe encoder for categorical features.

    这里提供了 2 种分箱方法：基于卡方值最小的分箱和基于坏样本率差异最大化的分箱.

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
    confidence: float, default to 3.841
        卡方分箱停止的阈值
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
    >>> from category_encoders import WOEEncoder
    >>> from sklearn.datasets import load_boston
    >>>
    >>> bunch = load_boston()
    >>> df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    >>> y = bunch.target > 22.5
    >>> df['y'] = y
    >>>
    >>> col = 'RAD'
    >>> my_encoder = CategoryWOEEncoder(col, 'y', max_bins=100, bin_pct_threshold=0)
    >>> df_my = my_encoder.fit_transform(df)
    >>> print(df_my[col+'_woe'])
    >>>
    >>> # 对比 sklearn_contrib，两者的结果应该相同
    >>> sklearn_encoder = WOEEncoder(cols=[col]).fit(df, y)
    >>> df_sklearn = sklearn_encoder.transform(df)
    >>> print(df_sklearn[col])
    """
    def __init__(self,
                 col_name: str,
                 target_col_name: str,
                 max_bins=10,
                 bin_pct_threshold=0.05,
                 woe_method='chi2',  # or "bad_rate"
                 confidence=3.841,
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
        self.confidence = confidence
        self.special_value_list = special_value_list
        self.missing_value = missing_value
        self.need_monotonic = need_monotonic
        self.u = u
        self.value_order_dict = value_order_dict
        self.regularization = regularization

    def _inspect(self):
        """A thorough inspection of the parameters."""
        if not isinstance(self.col_name, str):
            raise ValueError("The feature name must be a string.")
        if not isinstance(self.target_col_name, str):
            raise ValueError("The target feature name must be a string.")
        if not isinstance(self.max_bins, int):
            raise ValueError(
                "The maximal bins after binning must be an integer.")
        if not (isinstance(self.bin_pct_threshold, float)
                and 0. <= self.bin_pct_threshold < 1.):
            raise ValueError(
                "`bin_pct_threshold must be a float in between 0 and 1.`")
        if self.woe_method not in ('chi2', 'bad_rate'):
            raise ValueError("`woe_method` must be `'chi2'` or `'bad_rate'`.")
        if not isinstance(self.special_value_list, list):
            raise ValueError("`special_value_list` must be a Python list.")
        if not isinstance(self.need_monotonic, bool):
            raise ValueError("`need_monotonic` must be a boolean value.")
        if not isinstance(self.u, bool):
            raise ValueError("`u` must be a boolean value.")
        if not isinstance(self.value_order_dict, dict):
            raise ValueError("`value_order_dict` must be a Python dictionary.")
        if not isinstance(self.regularization, (float, int)):
            raise ValueError("`regularization` must be a float or int.")

    def _train(self, df, bin_num_threshold) -> pd.DataFrame:
        # 初始化分箱（不包含特殊值&缺失值）
        bin_df = initialize_bins(
            df, self.col_name, self.target_col_name, self.value_order_dict)

        # 如果合并原则按照卡方值（默认），则需要计算每 2 个相邻 bin 的卡方值
        calculator_between_bins = calculate_chi2_for_bin_df
        # 如果合并原则按照坏样本率，则需要计算每 2 个相邻 bin 的坏样本率差值
        if self.woe_method == 'bad_rate':
            calculator_between_bins = calculate_bad_rate_diff_for_bin_df
        values_calculated = calculator_between_bins(bin_df)

        # condition 1 卡方分箱时，停止条件：大于阈值 & 小于最大箱数
        if self.woe_method == 'chi2':
            while (len(df) > self.max_bins) and (min(values_calculated) < self.confidence):
                index = np.argmin(values_calculated)
                bin_df = update_bin_df(bin_df, index)
                values_calculated = calculator_between_bins(bin_df)
        # condition 1 坏样本率差异最大化: bin 的个数不大于 max_bins
        else:
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

    def fit(self, x: pd.DataFrame):
        self._inspect()  # a thorough inspection of the parameters
        df = x[[self.col_name, self.target_col_name]].copy()

        raw_length = len(df)
        bin_num_threshold = raw_length * self.bin_pct_threshold

        # missing_value 也相当于特殊值，放到 special_value_list 里一起处理
        special_value_flag = False
        if self.missing_value is not None:
            # Fill missing values with the specified value.
            df[self.col_name] = df[self.col_name].fillna(self.missing_value)
            if self.special_value_list is not None:
                self.special_value_list.append(self.missing_value)
            else:
                self.special_value_list = [self.missing_value]
            special_value_flag = True
        else:
            if self.special_value_list is not None:
                special_value_flag = True

        if special_value_flag:  # 处理特殊值——每个特殊值（缺失值）单独作为一个 bin
            df, stats = process_special_values(
                df, self.col_name, self.target_col_name,
                self.special_value_list)
            self.max_bins -= len(self.special_value_list)

        bin_df = self._train(df, bin_num_threshold)
        if special_value_flag:
            bin_df = bin_df.append(stats, ignore_index=True)

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

    def transform(self, x: pd.DataFrame):
        new_x = x.copy()

        if self.missing_value is not None:
            new_col = self.col_name + '_filled'
            new_x[new_col] = new_x[self.col_name].fillna(self.missing_value)
        else:
            new_col = self.col_name + '_copied'
            new_x[new_col] = new_x[self.col_name].copy()

        new_x[self.col_name+'_woe'] = new_x[new_col].map(self.bin_woe_mapping_)
        del new_x[new_col]

        return new_x


if __name__ == '__main__':
    from category_encoders import WOEEncoder
    from sklearn.datasets import load_boston

    bunch = load_boston()
    data = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    y = bunch.target > 22.5
    data['y'] = y

    col = 'RAD'

    # Test: vs sklearn_contrib
    # my_encoder = CategoryWOEEncoder(col, 'y', max_bins=100,
    #                                 bin_pct_threshold=0)
    # sklearn_encoder = WOEEncoder(cols=[col]).fit(data, y)
    # df_sklearn = sklearn_encoder.transform(data)
    # print(df_sklearn[col])
    # df_my = my_encoder.fit_transform(data)
    # print(df_my[[col + '_woe', col]])
    # print(my_encoder.bin_df_)

    # Test: 缺失值 & 特殊值
    # data[col] = data[col].where(data[col] != 1., np.nan)
    # my_encoder = CategoryWOEEncoder(col, 'y', special_value_list=[2., 3.],
    #                                 missing_value=100.)
    # df_my = my_encoder.fit_transform(data)
    # print(df_my[[col+'_woe', col]])
    # print(my_encoder.bin_df_)

    # Test: woe_method='bad_rate'
    # data = pd.read_excel(
    #     '/Users/bingli/codes/risk_control_with_ai/data/data_for_tree.xlsx')
    # col = 'class_new'
    # target_col = 'bad_ind'
    # my_encoder = CategoryWOEEncoder(col, target_col, max_bins=10,
    #                                 bin_pct_threshold=0.05,
    #                                 # special_value_list=['A'],
    #                                 woe_method='bad_rate', regularization=0,
    #                                 )
    # df_my = my_encoder.fit_transform(data)
    # print(df_my[[col+'_woe', col]])
    # print(my_encoder.bin_df_)
