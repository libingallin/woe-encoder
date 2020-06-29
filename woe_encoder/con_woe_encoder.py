# -*-encoding: utf-8 -*-
"""
@time: 2020/6/9
@author: bingli
"""
import bisect

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from woe_encoder.utils import (calculate_woe_iv,
                               gen_special_value_list,
                               if_monotonic)
from woe_encoder.utils_for_con import (calculate_bad_rate_diff_for_bin_arr,
                                       calculate_chi2_for_bin_arr,
                                       initialize_bins_for_con,
                                       locate_index,
                                       process_special_values,
                                       update_bin_arr)


class ContinuousWOEEncoder(BaseEstimator, TransformerMixin):
    """A sklearn-compatible woe encoder for Continuous features.

    这里提供了 2 种分箱方法：基于卡方值的分箱和基于坏样本率差异最大化的分箱.
    1. 基于卡方的分箱
       - 按照卡方阈值停止
       - 按照最大箱数停止
    2. 基于坏样本率差异最大化的分箱
       - 按照最大箱数停止

    Parameters
    ----------
    col_name: str
        需要转换的特征名列表
    target_col_name: str
        目标特征名
    max_bins: int
        最大箱数
    bin_pct_threshold: float, default=0.05
        分箱后每一箱含有的样本量比列（相对于全体样本）
    woe_method: str, "chi2" (default) or "bad_rate"
        bin 的合并原则。"chi2" 表示按卡方值，"bad_rate" 表示坏样本率
    min_chi2_flag: boolean, default=True
        如果是基于卡方值阈值的分箱，停止条件可以是最小卡方值大于阈值（True），也可以是
        箱数大于最大箱数（False）
    confidence: float, default=.841
        卡方分箱停止的阈值
    special_value_list: list, default=None
        特征中需要特殊对待的值。like: [s_v_1, s_v_2]
    imputation_value: default=None
        缺失值的填充值
    need_monotonic: boolean, default=False
        特征转换后是否需要满足单调性
    u: boolen, default=False
        特征转换后是否需要满足 U 型
    min_margin: float
        第一个 bin 的左边界
    max_margin: float
        最后一个 bin 的右边界
    regularization: float, default=1.0
        计算 woe 时加入正则化

    Attributes
    ----------
    bin_result_: pd.DataFrame
        分箱后结果，含有每个 bin 的统计值
    iv_: float
        分箱后该特征的 IV 值
    """
    def __init__(self,
                 col_name: str,
                 target_col_name: str,
                 max_bins=10,
                 bin_pct_threshold=0.05,
                 woe_method='chi2',  # "bad_rate"
                 min_chi2_flag=True,
                 confidence=3.841,
                 special_value_list=None,
                 imputation_value=None,
                 need_monotonic=True,
                 u=False,
                 min_margin=float('-inf'),
                 max_margin=float('inf'),
                 regularization=1.0):
        self.col_name = col_name
        self.target_col_name = target_col_name
        self.max_bins = max_bins
        self.bin_pct_threshold = bin_pct_threshold
        self.woe_method = woe_method
        self.min_chi2_flag = min_chi2_flag
        self.confidence = confidence
        self.special_value_list = special_value_list
        self.imputation_value = imputation_value
        self.need_monotonic = need_monotonic
        self.u = u
        self.min_margin = min_margin
        self.max_margin = max_margin
        self.regularization = regularization

    def _inspect(self):
        """A thorough inspection of the parameters."""
        assert isinstance(self.col_name, str), "Need a string."
        assert isinstance(self.target_col_name, str), "Need a string."
        assert isinstance(self.max_bins, int), "Need an integer."

        if not (isinstance(self.bin_pct_threshold, (float, int))
                and 0. <= self.bin_pct_threshold < 1.):
            raise ValueError(
                "`bin_pct_threshold must be a float in between 0 and 1.`")
        assert self.woe_method in ('chi2', 'bad_rate'), "'chi2' or 'bad_rate'."

        if self.woe_method == 'chi2':
            assert isinstance(self.min_chi2_flag, bool), "True or False."
        else:  # woe_method='bad_rate'
            self.min_chi2_flag = False

        if self.special_value_list is not None:
            assert isinstance(self.special_value_list, list), "Need a list."

        assert isinstance(self.need_monotonic, bool), "Need a boolean value."
        assert isinstance(self.u, bool), "Need a boolean value"

        assert isinstance(self.regularization, (float, int)), "float or integer."

    def _train(self, df, bin_num_threshold):
        bin_arr = initialize_bins_for_con(
            df, self.col_name, self.target_col_name)

        # 如果合并原则按照卡方值（默认），则需要计算每 2 个相邻 bin 的卡方值
        calculator_between_bins = calculate_chi2_for_bin_arr
        # 如果合并原则按照坏样本率，则需要计算每 2 个相邻 bin 的坏样本率差值
        if self.woe_method == 'bad_rate':
            calculator_between_bins = calculate_bad_rate_diff_for_bin_arr
        values_calculated = calculator_between_bins(bin_arr)

        # condition 1:
        # 如果基于卡方分箱，则可以是按照阈值合并（min_chi2_flag=True）也可以是按照最大箱数合并
        # 如果基于坏样本率差异极大化，只能是按照最大箱数合并
        if self.min_chi2_flag:   # 停止条件：最小卡方值大于阈值
            while (len(values_calculated) > 0
                   and min(values_calculated) < self.confidence
                   and len(bin_arr) > 1):
                index = np.argmin(values_calculated)
                bin_arr, values_calculated = update_bin_arr(
                    bin_arr, values_calculated, index, self.woe_method)
        else:  # 停止条件：小于最大箱数
            while len(bin_arr) > self.max_bins and len(bin_arr) > 1:
                index = np.argmin(values_calculated)
                bin_arr, values_calculated = update_bin_arr(
                    bin_arr, values_calculated, index, self.woe_method)

        # condition 2: 每个 bin 中不会出现 bad_rate 为 0/1
        i = 0
        while i < len(bin_arr) and len(bin_arr) > 1:
            if 0 in bin_arr[i, 1:]:
                # 需要确定该 bin 是和上一个还是下一个 bin 合并
                # 不管哪种，统一转换成合并掉下一个 bin
                index = locate_index(bin_arr, values_calculated, i)
                bin_arr, values_calculated = update_bin_arr(
                    bin_arr, values_calculated, index, self.woe_method)
                i -= 1  # 需要继续从 i 位置开始
            i += 1

        # condition 3: 每个 bin 中的样本数不少于阈值
        i = 0
        while i < len(bin_arr) and len(bin_arr) > 1:
            bin_num = bin_arr[i, 1:].sum()
            if bin_num < bin_num_threshold:
                index = locate_index(bin_arr, values_calculated, i)
                bin_arr, values_calculated = update_bin_arr(
                    bin_arr, values_calculated, index, self.woe_method)
                i -= 1
            i += 1

        # condition 4: 满足单调性 (U 型)
        if self.need_monotonic and len(bin_arr) > 2:
            bad_rates = bin_arr[:, 1] / bin_arr[:, 1:].sum(axis=1)
            while not if_monotonic(bad_rates, self.u):
                index = np.argmin(values_calculated)
                bin_arr, values_calculated = update_bin_arr(
                    bin_arr, values_calculated, index, self.woe_method)
                bad_rates = bin_arr[:, 1] / bin_arr[:, 1:].sum(axis=1)

        return bin_arr

    def fit(self, x: pd.DataFrame):
        self._inspect()
        df = x[[self.col_name, self.target_col_name]].copy()

        raw_length = len(df)
        bin_num_threshold = raw_length * self.bin_pct_threshold

        # imputation_value 也相当于特殊值，放到 special_value_list 里一起处理
        special_value_flag, special_values = gen_special_value_list(
            df, self.col_name, self.imputation_value, self.special_value_list)
        if special_value_flag:  # 处理特殊值——每个特殊值（缺失值）单独作为一个 bin
            df, stats = process_special_values(
                df, self.col_name, self.target_col_name, special_values)
            self.max_bins -= len(special_values)

        # 分箱
        if len(df) == 0:
            bin_df = pd.DataFrame()
        else:
            bin_arr = self._train(df, bin_num_threshold)

            # 分箱结果
            bin_df = pd.DataFrame(bin_arr[:, 1:], columns=['bad_num', 'good_num'])
            bin_df['bad_rate'] = bin_arr[:, 1] / bin_arr[:, 1:].sum(axis=1)
            cutoffs = bin_arr[:-1, 0].tolist()
            left_values = cutoffs.copy()
            left_values.insert(0, self.min_margin)
            bin_df['left_exclusive'] = left_values
            right_values = cutoffs.copy()
            right_values.append(self.max_margin)
            bin_df['right_inclusive'] = right_values

        if special_value_flag:
            bin_df = bin_df.append(stats, ignore_index=True)

        # Calculate WOE and IV
        bin_df[['woe', 'iv']] = calculate_woe_iv(
            bin_df, bad_col='bad_num', good_col='good_num',
            regularization=self.regularization)

        bin_df = bin_df.reindex(
            columns=['left_exclusive', 'right_inclusive', 'good_num',
                     'bad_num', 'bad_rate', 'woe', 'iv'])

        self.cutoffs_ = cutoffs
        self.bin_result_ = bin_df
        self.iv_ = bin_df['iv'].sum()
        return self

    def transform(self, X):
        new_x = X.copy()

        if self.imputation_value is not None:
            new_col = self.col_name + '_filled'
            new_x[new_col] = new_x[self.col_name].fillna(self.imputation_value)
        else:
            new_col = self.col_name + '_copied'
            new_x[new_col] = new_x[self.col_name].copy()

        # 特殊值和缺失的填充值一起处理
        if self.special_value_list is not None or self.imputation_value is not None:
            new_x[self.col_name + '_woe'] = new_x[new_col].apply(
                self._woe_replace_with_special_value)
        else:
            woes = self.bin_result_['woe'].tolist()
            new_x[self.col_name + '_woe'] = new_x[new_col].apply(
                lambda x: woes[bisect.bisect_left(self.cutoffs_, x)])
        return new_x.drop(columns=new_col)

    def _woe_replace_with_special_value(self, x):
        """返回与单个值（可能是特殊值/缺失值）对应的 woe 值."""
        woes = self.bin_result_['woe'].tolist()

        special_values = []
        if self.special_value_list is not None:
            special_values.extend(self.special_value_list)
        if self.imputation_value is not None:
            special_values.append(self.imputation_value)

        if x in special_values:
            return self.bin_result_.loc[
                self.bin_result_['left_exclusive'] == x, 'woe'
            ].values[0]
        else:
            return woes[bisect.bisect_left(self.cutoffs_, x)]
