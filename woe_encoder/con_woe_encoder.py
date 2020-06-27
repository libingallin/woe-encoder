# -*-encoding: utf-8 -*-
"""
@time: 2020/6/9
@author: bingli
"""
import bisect

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, TransformerMixin

from woe_encoder.utils import calculate_woe_iv, gen_special_value_list, if_monotonic
from woe_encoder.utils_for_con import initialize_bins_for_con, \
    process_special_values, calculate_chi2_for_bin_arr, \
    calculate_bad_rate_diff_for_bin_arr


class ContinuousWOEEncoder(BaseEstimator, TransformerMixin):
    """A sklearn-compatible woe encoder for Continuous features.

    这里提供了 2 种分箱方法：基于卡方值最小的分箱和基于坏样本率差异最大化的分箱.

    连续型变量分箱后需要满足的条件:
    1. pass
    2. pass
    3. pass
    4. pass

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
    imputation_value: default to None
        缺失值的填充值
    need_monotonic: boolean, default to False
        特征转换后是否需要满足单调性
    u: boolen, default to False
        特征转换后是否需要满足 U 型
    min_margin: float
        第一个 bin 的左边界
    max_margin: float
        最后一个 bin 的右边界
    regularization: float, default to 1.0
        计算 woe 时加入正则化
    """
    def __init__(self,
                 col_name: str,
                 target_col_name: str,
                 max_bins=10,
                 bin_pct_threshold=0.05,
                 woe_method='chi2',  # "bad_rate"
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
        pass

    def _train(self, df, bin_num_threshold):
        bin_arr = initialize_bins_for_con(
            df, self.col_name, self.target_col_name)

        # 如果合并原则按照卡方值（默认），则需要计算每 2 个相邻 bin 的卡方值
        calculator_between_bins = calculate_chi2_for_bin_arr
        # 如果合并原则按照坏样本率，则需要计算每 2 个相邻 bin 的坏样本率差值
        if self.woe_method == 'bad_rate':
            calculator_between_bins = calculate_bad_rate_diff_for_bin_arr
        values_calculated = calculator_between_bins(bin_arr)

        # condition 1: 卡方分箱时，停止条件：大于阈值 & 小于最大箱数 （？？？）
        if self.woe_method == 'chi2':
            while ((len(bin_arr) > self.max_bins)
                    and (min(values_calculated) < self.confidence)
                    and (len(values_calculated) > 0)):
                index = np.argmin(values_calculated)
                bin_df = update_bin_df(bin_df, index)
                values_calculated = calculator_between_bins(bin_df)
        else:  # condition 1: 坏样本率差异最大化时，停止条件：小于最大箱数
            while len(bin_arr) > self.max_bins:
                index = np.argmin(values_calculated)
                bin_df = update_bin_df(bin_df, index)
                values_calculated = calculator_between_bins(bin_df)

        # # condition 1: 停止条件：小于最大箱数
        # while len(bin_df) > self.max_bins:
        #     index = np.argmin(values_calculated)
        #     bin_df = update_bin_df(bin_df, index)
        #     values_calculated = calculator_between_bins(bin_df)

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

        self._train(df, bin_num_threshold)







    def transform(self, x: pd.DataFrame):
        pass

