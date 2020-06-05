# -*-encoding: utf-8 -*-
"""
@time: 2020-05-26
@author: libingallin@outlook.com
"""
import bisect

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, TransformerMixin

from woe_encoder.utils import calculate_woe_iv, if_monotonic


class NumericalWOEEncoder(BaseEstimator, TransformerMixin):
    """WOE Encoder based on a specific Chi-square value chi2 for numerical feature.

    对连续型特征做卡方阈值法分箱 + WOE 转换.

    对连续型变量分箱后需要满足以下要求:
    1. bins 的数量不超过最大数. 如果以 bin 之间的 chi2 值是否小于阈值来决定是否合并 bin,
       那么这个条件可以不需要.
       如果不满足, 将具有最小 chi2 值的相邻 bin 合并.
    2. 每个 bin 中的 bad_rate 不为 0 或 1.
       如果不满足, 将该 bin 与其上 (或者下, chi2 值较小) 一个 bin 合并.
    3. 每个 bin 中的总样本数占总样本数的比例不低于阈值 (默认为 5%).
       如果不满足, 将该 bin 与其上 (或者下, chi2 值较小) 一个 bin 合并.
    4. 连续 bin 的 bad_rate 满足单调性.
       如果不满足, 将具有最小 chi2 值的相邻 bin 合并.
    Notice: 这里, 以 2 个 bin 之间的 chi2 值小于阈值, 则合并这 2 个 bin. 所以并没有满足
    要求 1. 如果非要满足要求 1, 则可以指定 max_bins (默认为 None), 要求 1 将在要求 2, 3
    和 4 执行后再执行.

    Parameters
    ----------
    col_name: str
        需要转换的 WOE 转换的列
    target_col_name: str
        目标列 (y)
    initialize_method: str, default to 'all'
        初始化分箱方法. 默认每个 unique 值当作一个 bin
    bin_pct_threshold: float, default to 0.05
        每个 bin 最少数量
    confidence: float, default to 3.841
        合并 bin 的最大阈值
    special_value: None or float, default to None
        需要特殊对待的值. 单独作为一个 bin
    max_bins: None or int, default to None
        bin 的最大数量.
    need_monotonic: boolean, default to True
        是否需要满足单调性. 连续型特征分箱后默认需要满足单调性.
    u: boolean, default to False
        不满足单调性的时候是否允许 U 型 (正/倒). 默认不允许.
    min_cutoff: float or "min", default to float('-inf')
        最左边的值. 可以选择 min (TODO)
    max_cutoff: float, or "max", default to float('inf')
        最右边的值. 可以选择 max (TODO)

    Attributes
    ----------
    cutoffs_: list
        分割点组成的 list. 不含第一个 bin 的左边，不含最后一个 bin 的右边
    woe_: list
        每个 bin 的 WOE 值
    iv_: float
        分箱之后该特征的 IV 值.
    bin_result_: pd.DataFrame
        分箱之后的统计值. 如果有 special_value，则最后一行是 special_value 的想关信息.
    chi2_list_：list of float
        分箱的之后的卡方值.

    Examples
    --------
    import numpy as np
    import pandas as pd
    from scipy.stats import chi2_contingency
    from woe_encoder.woe_encoder import NumWOEEncoder

    X = ...   # pd.DataFrame
    woe_encoder = NumWOEEncoder('client_age', 'repaid_flag')
    woe_encoder.fit(X)
    # 替换掉相应的列
    X_transformed = woe_encoder.transform(X)   # pd.DataFrame
    """

    def __init__(self,
                 col_name: str,
                 target_col_name: str,
                 initialize_method='all',
                 bin_pct_threshold=0.05,
                 confidence=3.841,
                 max_bins=None,
                 special_value=None,
                 need_monotonic=True,
                 u=False,
                 min_cutoff=float('-inf'),
                 max_cutoff=float('inf')):
        self.col_name = col_name
        self.target_col_name = target_col_name
        self.initialize_method = initialize_method
        self.bin_pct_threshold = bin_pct_threshold
        self.confidence = confidence
        self.special_value = special_value
        self.max_bins = max_bins
        self.need_monotonic = need_monotonic
        self.u = u
        self.min_cutoff = min_cutoff
        self.max_cutoff = max_cutoff

    def _inspect(self, df):
        """确保入参、输入数据格式的正确性."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input data must be DataFrame.")

        if isinstance(self.col_name, str):
            raise ValueError("Please enter a string for `col_name`.")
        if self.col_name not in df.columns:
            raise ValueError("{} is not in the input data.".format(
                self.col_name))
        if len(df) != sum(df[self.col_name].notnull()):
            raise ValueError(
                "There are missing values in `{}`".format(self.col_name))

        if isinstance(self.target_col_name, str):
            raise ValueError("Please enter a string for `target_col_name`.")
        if self.target_col_name not in df.columns:
            raise ValueError("{} is not in the input data.".format(
                self.col_name))
        if len(df) != sum(df[self.target_col_name].notnull()):
            raise ValueError(
                "There are missing values in `{}`".format(
                    self.target_col_name))

    @staticmethod
    def calculate_bad_rate(np_arr: np.ndarray, only_bad_rate=True):
        """"计算 NumPy 数组的 bad_rate.

        Parameters
        ----------
        np_arr: np.ndarray
        only_bad_rate: bool, default to True
            是否只返回 bad_rate。默认只返回 bad_rate。如果为 False，
            那么在 `np_arr` 末尾追加 bad_rate 列。

        Returns
        -------
        arrray-like. 每个 bin 的 bad_rate.

        Examples
        --------
        calculate_bad_rate(a) -> [0.333, 0.4]
            bin   bad_count   good_count
            ----------------------------
            xx    10           20
            xxx   20           30
        calculate_bad_rate(a, False) ->
            bin   bad_count   good_count  bad_rate
            --------------------------------------
            xx    10           20         0.3333
            xxx   20           30         0.4
        """
        # 增加 bad_rate
        row_sum = np_arr[:, 1:].sum(axis=1)
        # np_arr = np.concatenate((np_arr, row_sum.reshape(-1, 1)), axis=1)
        # 为每一个 bin 增加 bad_rate (==bad/(bad+good))
        bad_rate = np.round(np_arr[:, 1] / row_sum, decimals=4)
        # np_arr = np.concatenate((np_arr, bad_rate.reshape(-1, 1)), axis=1)
        if only_bad_rate:
            return bad_rate
        else:
            return np.concatenate((np_arr, bad_rate.reshape(-1, 1)), axis=1)

    @staticmethod
    def calculate_chi2_for_array(combined_arr: np.ndarray) -> list:
        """从第一行开始, 计算相邻 2 行的卡方值.

        每次重新计算很慢.
        """
        chi2_list = []
        arr_len = len(combined_arr)
        for i in range(arr_len - 1):
            chi2_v = chi2_contingency(combined_arr[i:i + 2, 1:])[0]
            chi2_list.append(chi2_v)
        if len(chi2_list) != arr_len - 1:
            raise ValueError("卡方值的数量应该等于数组长度减 1.")
        return chi2_list

    def _init_bins(self, df):
        """初始化分箱操作, 并计算相邻 2 个 bin 的卡方值.

        默认每个 unique value 当作一个 bin.

        Parameters
        ----------
        df: pd.DataFrame
            原始输入数据. 内含 col_name 和 target_col_name.

        Returns
        -------
        np.ndarray. 分箱的结果 with 一些统计信息.
            col_name  bad_num  good_num
            ---------------------------
            10        2        10
            20        4        16
        list. 相邻 2 个 bin 的卡方值组成的列表.
        """
        # TODO(libing@souche.com): 等频、等宽初始化分箱
        # print("Initializing bins for numerical feature...")
        if self.initialize_method == 'all':
            # 统计每个 bin 包含的数据量
            # default sort=True
            grouped = df.groupby(self.col_name)[self.target_col_name]
            bin_num = grouped.count()
            # index: bin 值, col: 数量
            bin_num = pd.DataFrame({'bin_num': bin_num})
            # 统计每个 bin 的正样本数 (target_col_value == 1)
            bad_num = grouped.sum()
            # index: bin 值, col: 正样本数
            bad_num = pd.DataFrame({'bad_num': bad_num})

            count_df = bin_num.merge(bad_num, how='inner',
                                     left_index=True, right_index=True)
            # 每个 bin 的负样本数 (target_col_value == 0)
            count_df['good_num'] = count_df['bin_num'] - count_df['bad_num']
            del count_df['bin_num']
            count_df.reset_index(inplace=True)
            if not len(bin_num) == len(bad_num) == len(count_df):
                raise ValueError("The length of data must be SAME.")

            combined_arr = count_df.values  # For high speed

        # 处理连续没有正/负样本的区间，则进行区间的向下合并 (防止计算 chi2 出错)
        # print("Merging unreasonable bins...")
        i = 0
        while i <= (len(combined_arr) - 2):
            # 第 i 和 i+1 个位置的 bad_num 连续为 0 或者 good_num 连续为 0
            if (combined_arr[i:i + 2, 1] == 0).all() or (
                    combined_arr[i: i + 2, 2] == 0).all():
                combined_arr[i, 0] = combined_arr[i + 1, 0]
                combined_arr[i, 1:] += combined_arr[i + 1, 1:]
                combined_arr = np.delete(combined_arr, i + 1, axis=0)
                i -= 1  # 需要继续从 i 位置开始
            i += 1

        # 计算相邻 2 个 bin 的 chi2
        # print("Calculating chi2 value for every two sequent bins...")
        chi2_list = self.calculate_chi2_for_array(combined_arr)
        return combined_arr, chi2_list

    def _merge_bins(self, np_arr: np.ndarray, chi2_list: list, index: int):
        """将 index 位置的 bin 与其下一个 bin 合并.

        Parameters
        ----------
        np_arr: np.ndarray
            每一行代表一个 bin.
            length L:   [0, 1, 2, 3, ..., L-2, L-1]
        chi2_list: list
            每一个元素表示相邻 2 个 bin 的卡方值.
            length L-1: [0, 1, 2, 3, ..., L-2]
        index: int
            待合并 bin 的位置 (与其下一个 bin 合并）.

        Examples
        --------
        (50, 60] 向下合并, 与 (60, 70] 合并成 (50, 70].
            col_name  bad_num  good_num
            50        10       20
            60        30       40
            70        15       50
        --->
            col_name  bad_num  good_num
            50        10       20
            70        45       90
        """
        raw_arr_len = len(np_arr)
        if raw_arr_len != len(chi2_list) + 1:
            raise ValueError(
                "The length of RAW `np_arr` is greater than the length of "
                "RAW `chi_values` by 1.\n Please check the inputs.")

        # 向下合并时 (index == index+1)
        np_arr[index, 0] = np_arr[index + 1, 0]
        np_arr[index, 1:] += np_arr[index + 1, 1:]
        np_arr = np.delete(np_arr, index + 1, axis=0)

        # 可以每次重新从头计算 chi2_list 来避免这种烧脑地手动更新 chi2_list.
        # 但是这种方法很慢, 而且手动更新有助于理解全过程.
        if index == raw_arr_len - 1:  # 最后 1 个 bin 没法向下合并
            raise ValueError("The last bin must be merged with the above.")
        elif index == raw_arr_len - 2:  # 倒数第 2 个 bin，则合并最后 2 个 bin
            chi2_list[index - 1] = \
            chi2_contingency(np_arr[index - 1:index + 1, 1:])[0]
            _ = chi2_list.pop(index)
        else:
            if index != 0:  # 如果是第 1 个 bin，只更新第 1 chi2，然后删除第 2 个 chi2
                chi2_list[index - 1] = \
                chi2_contingency(np_arr[index - 1:index + 1, 1:])[0]
            chi2_list[index] = chi2_contingency(np_arr[index:index + 2, 1:])[0]
            chi2_list.pop(index + 1)

        # 处理后验证长度关系
        if len(np_arr) - 1 != len(chi2_list):
            raise ValueError(
                "The length of `np_arr` is greater than the length of "
                "`chi2_list` by 1 after process.")
        return np_arr, chi2_list

    @staticmethod
    def locate_index(combined_arr: np.ndarray, chi2_list: list, i: int) -> int:
        """定位到在 coding 中处理的 bin 的位置.

        第 i 个 bin 需要合并 (与其上一个或者下一个合并). 与其上一个 bin 合并, 可以看成其上
        一个 bin 与该 bin 合并.
        1. 如果是第一个 bin, 则只能向下合并, 返回 i.
        2. 如果是最后一个 bin, 则只能和其上一个 bin 合并, 返回 i-1.
        3. 如果是中间某个 bin, 其与上下哪一个 bin 合并取决于该卡方值. 与上一个 bin 的卡方
           值较小, 则与上一个 bin 合并, 返回 i-1; 与上下 bin 的卡方值相等, 但上一个 bin
           的数量较少, 则与上一个 bin 合并, 返回 i-1. 剩余的情况返回 i.
        """
        total_n = combined_arr[:, 1:].sum()
        if i == 0:
            return i
        elif i == len(combined_arr) - 1:
            return i - 1
        else:
            chi2_1 = chi2_list[i - 1]  # 与上一个 bin 的卡方值
            chi2_2 = chi2_list[i]  # 与下一个 bin 的卡方值
            above_bin_pct = combined_arr[i - 1, 1:].sum() / total_n
            below_bin_pct = combined_arr[i, 1:].sum() / total_n

            cond1 = chi2_1 < chi2_2  # 与上一个 bin 的卡方值较小
            # 与上下 bin 的卡方值相等, 但上一个 bin 中的样本数较少
            cond2 = chi2_1 == chi2_2 and above_bin_pct <= below_bin_pct
            if cond1 or cond2:
                return i - 1
            else:
                return i

    def _train(self, combined_arr, chi2_list):
        # condition 1: 相邻 bin 的 chi2 要大于 confidence
        # 否则合并 chi2 最小的 2 个 bin
        while len(chi2_list) > 0 and min(chi2_list) < self.confidence:
            index = np.argmin(chi2_list)
            combined_arr, chi2_list = self._merge_bins(
                combined_arr, chi2_list, index)

        # condition 2: 每个 bin 中不会出现 bad_rate 为 0/1
        # print("Processing the bin containing 0...")
        i = 0
        while i < len(combined_arr):
            row_values = combined_arr[i, 1:]
            if 0 in row_values:
                index = self.locate_index(combined_arr, chi2_list, i)
                combined_arr, chi2_list = self._merge_bins(
                    combined_arr, chi2_list, index)
                i -= 1  # 需要继续从 i 位置开始
            i += 1

        # condition 3: 每个 bin 中的样本数不少于阈值.
        # print("Processing the bin less than {}%...".format(
        #     self.bin_pct_threshold * 100))
        i = 0
        while i < len(combined_arr):
            bin_num = combined_arr[i, 1:].sum()
            if bin_num < self.bin_num_threshold:
                index = self.locate_index(combined_arr, chi2_list, i)
                combined_arr, chi2_list = self._merge_bins(
                    combined_arr, chi2_list, index)
                i -= 1
            i += 1

        bad_rates = self.calculate_bad_rate(combined_arr)

        # condition 4: 满足单调性 (U 型)
        if self.need_monotonic:
            # print("Merging bins to meet monotonicity...")
            while not if_monotonic(bad_rates, self.u):
                index = np.argmin(chi2_list)
                combined_arr, chi2_list = self._merge_bins(
                    combined_arr, chi2_list, index)
                bad_rates = self.calculate_bad_rate(combined_arr)

        # bin 的数量不超过 max_bins
        if self.max_bins:
            if not isinstance(self.max_bins, int):
                raise ValueError("Please enter an integer for `max_bins`.")
            while len(combined_arr) > self.max_bins:
                index = np.argmin(chi2_list)
                combined_arr, chi2_list = self._merge_bins(
                    combined_arr, chi2_list, index)
                bad_rates = self.calculate_bad_rate(combined_arr)

        return combined_arr, chi2_list, bad_rates

    def fit(self, X):
        df = X.copy()
        # 每个 bin 的最少样本数
        self.bin_num_threshold = len(df) * self.bin_pct_threshold

        if self.special_value:
            # print("Process special value {}...".format(self.special_value))
            df_special = df[df[self.col_name] == self.special_value]
            df = df[df[self.col_name] != self.special_value]

            # 计算 special_value 的统计
            left, right = self.special_value, self.special_value
            bad_num_special = df_special[self.target_col_name].sum()
            all_num_special = len(df_special)
            good_num_special = all_num_special - bad_num_special
            bad_rate_special = bad_num_special / all_num_special
            if all_num_special < self.bin_num_threshold:
                print("The number for `{}` is less than {}".format(
                    self.special_value, self.bin_pct_threshold))
            statistic_values = {
                'left': left,
                'right': right,
                'good_num': good_num_special,
                'bad_num': bad_num_special,
                'bad_rate': bad_rate_special
            }

        combined_arr, chi2_list = self._init_bins(df)
        combined_arr, chi2_list, bad_rates = self._train(combined_arr,
                                                         chi2_list)

        # 转换并保存结果
        arr_len = len(combined_arr)
        result = pd.DataFrame({'raw_col_name': [self.col_name] * arr_len})
        cutoffs = combined_arr[:-1, 0].tolist()
        left_values = cutoffs.copy()
        left_values.insert(0, self.min_cutoff)
        result['left_exclusive'] = left_values
        right_values = cutoffs.copy()
        right_values.append(self.max_cutoff)
        result['right_inclusive'] = right_values
        result['good_num'] = combined_arr[:, 2]
        result['bad_num'] = combined_arr[:, 1]
        result['bad_rate'] = bad_rates
        if self.special_value:
            # 最后 1 行是 special_value 的相关信息
            result = result.append(statistic_values, ignore_index=True)
        # Calculate WOE and IV
        result[['woe', 'iv']] = calculate_woe_iv(
            result, bad_col='bad_num', good_col='good_num')

        self.cutoffs_ = cutoffs
        self.woe_ = result['woe'].tolist()
        self.iv_ = result['iv'].sum()
        self.bin_result_ = result.iloc[:, 1:]
        self.chi2_list_ = chi2_list
        return self

    def transform(self, X):
        new_X = X.copy()

        if self.special_value:
            new_X[self.col_name + '_woe'] = new_X[self.col_name].apply(
                lambda x: self._woe_replace_with_special_value)
        else:
            new_X[self.col_name + '_woe'] = new_X[self.col_name].apply(
                lambda x: self.woe_[bisect.bisect_left(self.cutoffs_, x)])
        return new_X

    def _woe_replace_with_special_value(self, x):
        if x == self.special_value:
            return self.bin_result_.iloc[-1, -2]
        else:
            return self.woe_[bisect.bisect_left(self.cutoffs_, x)]
