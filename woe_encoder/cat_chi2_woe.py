# -*-encoding: utf-8 -*-
"""
@time: 2020-05-26
@author: libingallin@outlook.com
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, TransformerMixin

from woe_encoder.utils import calculate_woe_iv, if_monotonic
from woe_encoder.utils_for_cat import initialize_bins, locate_index, update_bin_df


class CategoryWOEEncoder(BaseEstimator, TransformerMixin):
    """WOE transformer for categorical feature."""

    def __init__(self,
                 col_name: str,
                 target_col_name: str,
                 max_bins=10,
                 bin_pct_threshold=0.05,
                 # woe_method='normal',  # max_bad_rated_diff
                 value_order_dict=None,
                 special_value=None,
                 need_monotonic=False,
                 u=False,
                 regularization=1.0):
        self.col_name = col_name
        self.target_col_name = target_col_name
        self.max_bins = max_bins
        self.bin_pct_threshold = bin_pct_threshold
        # self.woe_method = woe_method
        self.value_order_dict = value_order_dict
        self.special_value = special_value
        self.need_monotonic = need_monotonic
        self.u = u,
        self.regularization = regularization

    @staticmethod
    def calculate_chi2_for_bin_df(bin_df: pd.DataFrame) -> list:
        """计算所有相邻 bin 的卡方值."""
        chi2_list = []

        bin_df_len = len(bin_df)
        for i in range(bin_df_len - 1):
            chi2_v = chi2_contingency(bin_df.iloc[i:i + 2, 2:4])[0]
            chi2_list.append(chi2_v)

        if bin_df_len - 1 != len(chi2_list):
            raise ValueError(
                "The number of chi2 values is smaller than the length "
                "of `bin_df` by 1.")
        return chi2_list

    def _train(self, bin_df, bin_num_threshold):
        chi2_list = self.calculate_chi2_for_bin_df(bin_df)

        # condition 1: bin 的个数不大于 max_bins
        while len(bin_df) > self.max_bins:
            min_chi2_index = np.argmin(chi2_list)
            bin_df = update_bin_df(bin_df, min_chi2_index)
            chi2_list = self.calculate_chi2_for_bin_df(bin_df)

        # condition 2: 每个 bin 的 bad_rate 不为 0 / 1
        i = 0
        while i < len(bin_df):
            if bin_df.iloc[i, 4] in (0, 1):
                # index = self.locate_index(bin_df, chi2_list, i)
                index = locate_index(bin_df, chi2_list, i)
                bin_df = update_bin_df(bin_df, index)
                chi2_list = self.calculate_chi2_for_bin_df(bin_df)
                i -= 1
            i += 1

        # # condition 3: 每个 bin 的样本数比例大于 5%
        # i = 0
        # while i < len(bin_df):
        #     if bin_df.iloc[i, 1] < bin_num_threshold:
        #         # index = self.locate_index(bin_df, chi2_list, i)
        #         index = locate_index(bin_df, chi2_list, i)
        #         bin_df = update_bin_df(bin_df, index)
        #         chi2_list = self.calculate_chi2_for_bin_df(bin_df)
        #         i -= 1
        #     i += 1

        # condition 3: 每个 bin 中的样本数不能少于阈值
        while np.min(bin_df['bin_num']) < bin_num_threshold:
            index = np.argmin(bin_df['bin_num'])
            index = locate_index(bin_df, chi2_list, index)
            bin_df = update_bin_df(bin_df, index)
            chi2_list = self.calculate_chi2_for_bin_df(bin_df)

        # condition 4: bin 是否满足单调性
        # 只有有序的离散变量才需要
        if self.need_monotonic:
            bad_rates = bin_df.iloc[:, 4]
            while not if_monotonic(bad_rates, u=self.u):
                index = np.argmin(chi2_list)
                # bin_df, chi2_list = self._update_bin_df(bin_df, index)
                bin_df = update_bin_df(bin_df, index)
                chi2_list = self.calculate_chi2_for_bin_df(bin_df)
                bad_rates = bin_df.iloc[:, 4]

        return bin_df, chi2_list

    def fit(self, X):
        raw_length = len(X)
        bin_num_threshold = raw_length * self.bin_pct_threshold

        if self.special_value is not None:
            X_special = X[X[self.col_name] == self.special_value]
            X = X[X[self.col_name] != self.special_value]

            special_bin_num = len(X_special)
            special_bad_num = X_special[self.target_col_name].sum()
            special_good_num = special_bin_num - special_bad_num
            statistic_values = {
                self.col_name: [self.special_value],
                'bin_num': special_bin_num,
                'bad_num': special_bad_num,
                'good_num': special_good_num,
                'bad_rate': special_bad_num / special_bin_num,
                'bin_pct': special_bin_num / raw_length,
            }
            if self.value_order_dict:
                statistic_values.update(
                    {'order': self.value_order_dict[self.special_value]})

        bin_df = initialize_bins(X, self.col_name, self.target_col_name,
                                 value_order_dict=self.value_order_dict)
        bin_df, chi2_list = self._train(bin_df, bin_num_threshold)

        if self.special_value is not None:
            bin_df = bin_df.append(statistic_values, ignore_index=True)

        # Calculate WOE and IV
        bin_df[['woe', 'iv']] = calculate_woe_iv(
            bin_df, bad_col='bad_num', good_col='good_num',
            regularization=self.regularization)

        # 值与对应 WOE 的映射, 方便 transform
        bin_woe_mapping = {}
        for bin_list, woe_v in zip(bin_df[self.col_name], bin_df['woe']):
            for bin_v in bin_list:
                bin_woe_mapping[bin_v] = woe_v

        self.woe_ = bin_df['woe'].sum()
        self.iv_ = bin_df['iv'].sum()
        self.bin_result_ = bin_df
        self.bin_woe_mapping_ = bin_woe_mapping
        return self

    def transform(self, X):
        new_X = X.copy()
        new_X[self.col_name + '_woe'] = new_X[self.col_name].map(
            self.bin_woe_mapping_)
        return new_X


if __name__ == '__main__':
    import time

    from sklearn.datasets import load_boston
    from category_encoders import WOEEncoder
    bunch = load_boston()
    X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    y = bunch.target > 22.5

    # sklearn
    tmp = X.copy()
    tmp['y'] = y
    t_0 = time.time()
    sklearn_woe = WOEEncoder(cols=['RAD']).fit(tmp, y)
    res_1 = sklearn_woe.transform(tmp)
    t_1 = time.time()
    print(res_1['RAD'])

    # my
    t_2 = time.time()
    my_woe = CategoryWOEEncoder('RAD', 'y',
                                max_bins=1000, bin_pct_threshold=0.0)
    res_2 = my_woe.fit_transform(tmp)
    t_3 = time.time()
    print(res_2['RAD_woe'])
    print(t_1-t_0, t_3-t_2)


