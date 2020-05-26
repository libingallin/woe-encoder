# -*-encoding: utf-8 -*-
"""
@time: 2020-05-25 16:40
@author: libingallin@outlook.com
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, TransformerMixin

from .utils import calculate_woe_iv, if_monotonic


class CategoryWOEEncoder(BaseEstimator, TransformerMixin):
    """WOE transformer for categorical feature."""
    def __init__(self,
                 col_name: str,
                 target_col_name: str,
                 max_bins=6,
                 bin_pct_threshold=0.05,
                 value_order_dict=None,
                 special_value=None,
                 need_monotonic=False,
                 u=False):
        self.col_name = col_name
        self.target_col_name = target_col_name
        self.max_bins = max_bins
        self.bin_pct_threshold = bin_pct_threshold
        self.value_order_dict = value_order_dict
        self.special_value = special_value
        self.need_monotonic = need_monotonic
        self.u = u

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

    def _update_bin_df(self, bin_df, index):
        """合并 bin_df 中第 index 个 bin 和其下一个 bin."""
        total_n = bin_df.iloc[:, 1].sum()
        bin_df.iloc[index, 0].extend(bin_df.iloc[index + 1, 0])
        bin_df.iloc[index, 1:4] += bin_df.iloc[index + 1, 1:4]
        bin_df.iloc[index, 4] = bin_df.iloc[index, 2] / bin_df.iloc[index, 1]
        bin_df.iloc[index, 5] = bin_df.iloc[index, 1] / total_n
        if self.value_order_dict:
            bin_df.iloc[index, 6] = (bin_df.iloc[index, 6] + bin_df.iloc[
                index + 1, 6]) / 2

        bin_df.drop(index=index + 1, inplace=True)
        # index 永远是从 0 开始的序列
        bin_df.reset_index(drop=True, inplace=True)

        chi2_list = self.calculate_chi2_for_bin_df(bin_df)
        return bin_df, chi2_list

    @staticmethod
    def locate_index(bin_df: pd.DataFrame, chi2_list: list, i: int) -> int:
        """确定要被更新的 index (return). 该 index 表明第 index 个 bin 与其下一个 bin 合并.

        1. 如果 i 等于 0, 只能向下合并, index 为 0.
        2. 如果 i 是最后一个 bin, 即 bin_df 的最后一行, 那么只能向上合并, 但可以看成倒数第二个
        bin 向下合并, 此时 index=i-1.
        3. 如果 i 是以上 2 种情况的其他值. 需要该 bin (第 i 个) 与相邻 bin (上和下) 的较小卡方值
        来确定.
           - 如果与上一个 bin 的卡方值较小, 那么向上合并, 看作上一个 bin 向下合并, 此时
        index=i-1;
           - 如果与上一个 bin 的卡方值和与下一个 bin 的卡方值相等, 并且上一个 bin 的数据
        量较少, 也向上合并, 此时 index=i-1.
           - 其他情况, 该 bin (第 i 个) 向下合并, index=i.

        Parameter
        ---------
        bin_df: pd.DataFrame
            含有 bin 及其相关简单统计. 每 1 行代表 1 个 bin.
        chi2_list: list
            bin_df 中连续相邻 bin 的卡方值
        i: int
            i 表示 bin_df 中的索引/ bin 的索引.

        Returns
        -------
        int. 根据输入来确定 index 的位置.
        """
        if i == 0:
            # index = i
            return i
        elif i == len(bin_df) - 1:
            # index = i - 1
            return i - 1
        else:
            chi2_1 = chi2_list[i - 1]  # 与上一个 bin 的卡方值
            chi2_2 = chi2_list[i]  # 与下一个 bin 的卡方值

            cond1 = chi2_1 < chi2_2
            cond2 = (chi2_1 == chi2_2) and (bin_df[i - 1, 5] <= bin_df[i, 5])
            if cond1 or cond2:
                # index = i - 1
                return i - 1
            else:
                # index = i
                return i
        # return index

    def _init_bins(self, df: pd.DataFrame) -> (pd.DataFrame, list):
        """初始化生成每个 bin. 每个 unique 值当作一个 bin.

        如果有 value_order，则在生成的 pd.DataFrame 末尾增加一列 (order) 表示顺序.
        对结果进行排序: 如果没有 value_order, 则根据 bad_rate 升序 sort, 否则根据
        order 值升序 sort.

        Parameters
        ----------
        df: pd.DataFrame
            原始数据集. 含有 col_name 和 target_col_name 列.

        Returns
        -------
        pd.DataFrame. 每一行都是一个 bin. 第一列是每个 bin 的取值 (list 形式), 其余列
        是每个 bin 的简单统计.
        0                  1        2        3         4         5        6
        col_name           bin_num  bad_num  good_num  bad_rate  bin_pct  order
        -----------------------------------------------------------------------
        [value_1]
        [value_2]
        [value_3, value_4]
            最后 1 列 order 是否存在取决于该特征是否要求有序.
        """
        # print("Initializing bins for categorical feature...")
        grouped = df.groupby(self.col_name, sort=True)[self.target_col_name]
        bin_num = grouped.count()  # 每个 bin 的数量
        bin_num = pd.DataFrame({'bin_num': bin_num})
        bad_num = grouped.sum()  # 每个 bin 中 1 的数量
        bad_num = pd.DataFrame({'bad_num': bad_num})
        bin_df = bin_num.merge(bad_num, how='inner',
                               left_index=True, right_index=True)
        bin_df['good_num'] = bin_df['bin_num'] - bin_df['bad_num']
        bin_df['bad_rate'] = bin_df['bad_num'] / bin_df['bin_num']
        total_n = len(df)
        bin_df['bin_pct'] = bin_df['bin_num'] / total_n
        bin_df.reset_index(inplace=True)
        # index 永远是从 0 开始的序列
        bin_df = bin_df.sort_values('bad_rate', ignore_index=True)

        if self.value_order_dict:
            bin_df['order'] = bin_df[self.col_name].map(self.value_order_dict)
            bin_df = bin_df.sort_values('order', ignore_index=True)

        bin_df[self.col_name] = bin_df[self.col_name].map(lambda x: [x, ])

        chi2_list = self.calculate_chi2_for_bin_df(bin_df)

        return bin_df, chi2_list

    def _train(self, bin_df, chi2_list):
        # condition 1: bin 的个数不大于 max_bins
        while len(bin_df) > self.max_bins:
            min_chi2_index = np.argmin(chi2_list)
            bin_df, chi2_list = self._update_bin_df(bin_df, min_chi2_index)

        # condition 2: 每个 bin 的 bad_rate 不为 0 / 1
        i = 0
        while i < len(bin_df):
            if bin_df.iloc[i, 4] in (0, 1):
                index = self.locate_index(bin_df, chi2_list, i)
                bin_df, chi2_list = self._update_bin_df(bin_df, index)
                i -= 1
            i += 1

        # condition 3: 每个 bin 的样本数比例大于 5%
        i = 0
        while i < len(bin_df):
            if bin_df.iloc[i, 1] < self.bin_num_threshold:
                index = self.locate_index(bin_df, chi2_list, i)
                bin_df, chi2_list = self._update_bin_df(bin_df, index)
                i -= 1
            i += 1

        # condition 4: bin 是否满足单调性
        # 只有有序的离散变量才需要
        if self.need_monotonic:
            bad_rates = bin_df.iloc[:, 4]
            while not if_monotonic(bad_rates, u=self.u):
                index = np.argmin(chi2_list)
                bin_df, chi2_list = self._update_bin_df(bin_df, index)
                bad_rates = bin_df.iloc[:, 4]

        return bin_df, chi2_list

    def fit(self, X):
        raw_length = len(X)
        self.bin_num_threshold = raw_length * self.bin_pct_threshold

        if self.special_value:
            x_special = X[X[self.col_name == self.special_value]]
            X = X[X[self.col_name != self.special_value]]

            special_bin_num = len(x_special)
            special_bad_num = x_special[self.target_col_name].sum()
            special_good_num = special_bin_num - special_bad_num
            statistic_values = {
                self.col_name: self.special_value,
                'bin_num': special_bin_num,
                'bad_num': special_bad_num,
                'good_num': special_good_num,
                'bad_rate': special_bad_num / special_bin_num,
                'bin_pct': special_bin_num / raw_length,
            }
            if self.value_order_dict:
                statistic_values.update(
                    {'order': self.value_order_dict[self.special_value]})

        bin_df, chi2_list = self._init_bins(X)
        bin_df, chi2_list = self._train(bin_df, chi2_list)

        if self.special_value:
            bin_df = bin_df.append(statistic_values, ignore_index=True)

        # Calculate WOE and IV
        bin_df[['woe', 'iv']] = calculate_woe_iv(
            bin_df, bad_col='bad_num', good_col='good_num')

        self.woe_ = bin_df['woe'].sum()
        self.iv_ = bin_df['iv'].sum()
        self.bin_result_ = bin_df
        self.chi2_list_ = chi2_list
        bin_woe_mapping = {}
        for bin_list, woe_v in zip(bin_df[self.col_name], bin_df['woe']):
            for bin_v in bin_list:
                bin_woe_mapping[bin_v] = woe_v
        self.bin_woe_mapping_ = bin_woe_mapping
        return self

    def transform(self, X):
        new_X = X.copy()
        new_X[self.col_name] = new_X[self.col_name].map(self.bin_woe_mapping_)
        return new_X


if __name__ == '__main__':
    import time
    from category_encoders import WOEEncoder

    train_data = pd.read_csv('../data/train_data.csv', index_col=0)
    tmp_1 = train_data[['capital_name', 'repaid_flag']].copy()

    time_0 = time.time()
    cat_encoder = CatWOEEncoder('capital_name', 'repaid_flag',
                                max_bins=1000, bin_pct_threshold=0.0, )
    tmp_1_enc = cat_encoder.fit_transform(train_data)
    time_1 = time.time()

    tmp_2 = train_data[['capital_name', 'repaid_flag']].copy()
    time_2 = time.time()
    cat_encoder_2 = WOEEncoder(cols=['capital_name'])
    cat_encoder_2.fit(tmp_2, tmp_2['repaid_flag'])
    tmp_2_enc = cat_encoder_2.transform(tmp_2)
    time_3 = time.time()

    print(time_1 - time_0, time_3 - time_2)
    print(tmp_1_enc['capital_name'].equals(tmp_2_enc['capital_name']))
    print(tmp_1_enc['capital_name'], tmp_2_enc['capital_name'])


