# -*-encoding: utf-8 -*-
"""
@time: 2020/6/1
@author: libingallin@outlook.com
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from woe_encoder.utils import calculate_woe_iv
from woe_encoder.cat_utils import initialize_bins, locate_index, update_bin_df


class CatMaxBinBadRateDiffWOEEncoder(BaseEstimator, TransformerMixin):
    """基于 bad_rate 差值最大的离散型 WOE 编码.

    Parameters
    ----------
    col_name: str
        特征列名
    target_col_name: str
        目标列名
    max_bins: int, default to 10
       分箱后最大的箱数.
    bin_pct_threshold: float, default to 0.05
       每个 bin 中样本数最小阈值.
    special_value: string or int
       需要特殊对待的特征值.
    value_order_dict

    Attributes
    ----------
    woe_:
    """
    def __init__(self,
                 col_name: str,
                 target_col_name: str,
                 max_bins=10,
                 bin_pct_threshold=0.05,
                 special_value=None,
                 value_order_dict=None):
        self.col_name = col_name
        self.target_col_name = target_col_name
        self.max_bins = max_bins
        self.bin_pct_threshold = bin_pct_threshold
        self.special_value = special_value
        self.value_order_dict = value_order_dict

    def _inspect(self):
        """确保入参、输入数据格式的正确性."""
        if not isinstance(self.col_name, str):
            raise ValueError("Please enter a string for `col_name`.")

    def _train(self, bin_df, bin_num_threshold):
        """"按照一定的条件合并 bin."""
        bad_rates, bad_rate_diff = self.calculate_bin_bad_rate_diff(bin_df)

        # 1. 按照每个 bin 负样本占比差异最大化原则合并至 10 个 bin 以内.
        while len(bin_df) > self.max_bins:
            min_diff_index = np.argmin(bad_rate_diff)
            bin_df = update_bin_df(bin_df, min_diff_index)
            bad_rates, bad_rate_diff = self.calculate_bin_bad_rate_diff(bin_df)

        # 2: 每个 bin 的 bad_rate 不为 0 / 1
        i = 0
        while i < len(bin_df):
            if bin_df.iloc[i, 4] in (0, 1):
                index = locate_index(bin_df, bad_rate_diff, i)
                bin_df = update_bin_df(bin_df, index)
                bad_rates, bad_rate_diff = self.calculate_bin_bad_rate_diff(
                    bin_df)
                i -= 1
            i += 1

        # 3. 每个 bin 中的样本数不能少于阈值
        while np.min(bin_df['bin_num']) < bin_num_threshold:
            index = np.argmin(bin_df['bin_num'])
            index = locate_index(bin_df, bad_rate_diff, index)
            bin_df = update_bin_df(bin_df, index)
            bad_rates, bad_rate_diff = self.calculate_bin_bad_rate_diff(bin_df)

        return bin_df

    @staticmethod
    def calculate_bin_bad_rate_diff(bin_df):
        bad_rates = bin_df['bad_rate'].values
        bad_rate_diff = [j - i for i, j in zip(bad_rates, bad_rates[1:])]
        return bad_rates, bad_rate_diff

    def fit(self, X):
        self._inspect()

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
            # 如果特征含有特殊值，且每个特征有顺序，特征值也需要顺序？
            if self.value_order_dict:
                statistic_values.update(
                    {'order': self.value_order_dict[self.special_value]})

        bin_df = initialize_bins(X, self.col_name, self.target_col_name,
                                 value_order_dict=self.value_order_dict)
        bin_df = self._train(bin_df, bin_num_threshold)

        if self.special_value is not None:
            bin_df = bin_df.append(statistic_values, ignore_index=True)

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
    data = pd.read_excel(
        '/Users/bingli/codes/risk_control_with_ai/data/data_for_tree.xlsx')
    encoder = CatMaxBinBadRateDiffWOEEncoder(
        col_name='class_new',
        target_col_name='bad_ind',
        # special_value='D'
    )
    data_new = encoder.fit_transform(data)
    print(data_new[['class_new_woe', 'class_new']])
    print(encoder.woe_, encoder.iv_)
    print(encoder.bin_result_)
