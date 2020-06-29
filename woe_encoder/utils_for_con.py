# -*-encoding: utf-8 -*-
"""
@time: 2020/6/9
@author: bingli
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def initialize_bins_for_con(df, col_name: str, target_col_name: str):
    """初始化分箱操作.

    默认每个 unique value 当作一个 bin.

    Parameters
    ----------
    df: pd.DataFrame
        原始输入数据. 内含 col_name 和 target_col_name.
    col_name: str
    target_col_name: str

    Returns
    -------
    np.ndarray. 分箱的结果 with 一些统计信息.
        col_name  bad_num  good_num
        ---------------------------
        10        2        10
        20        4        16
    list. 相邻 2 个 bin 的卡方值组成的列表.
    """
    bin_df = df.groupby(col_name, sort=True)[target_col_name].agg(
        bad_num='sum', bin_num='count')
    bin_df['good_num'] = bin_df['bin_num'] - bin_df['bad_num']
    del bin_df['bin_num']
    bin_df.reset_index(inplace=True)

    combined_arr = bin_df.values  # For high speed

    # 处理连续没有正/负样本的区间，则进行区间的向下合并 (防止计算 chi2 出错)
    i = 0
    while i <= (len(combined_arr) - 2):
        cond1 = (combined_arr[i:i + 2, 1] == 0).all()   # bad_num 连续为 0
        cond2 = (combined_arr[i: i + 2, 2] == 0).all()  # good_num 连续为 0
        if cond1 or cond2:
            combined_arr[i, 0] = combined_arr[i + 1, 0]
            combined_arr[i, 1:] += combined_arr[i + 1, 1:]
            combined_arr = np.delete(combined_arr, i + 1, axis=0)
            i -= 1  # 需要继续从 i 位置开始
        i += 1

    return combined_arr


def process_special_values(df, col: str, target_col_name: str,
                           special_value_list: list):
    df_special = df[df[col].isin(special_value_list)]
    df = df[~df[col].isin(special_value_list)]

    stats = df_special.groupby(col)[target_col_name].agg(
        bad_num='sum', bin_num='count')
    stats['left_exclusive'] = special_value_list
    stats['right_inclusive'] = special_value_list
    stats['good_num'] = stats['bin_num'] - stats['bad_num']
    stats['bad_rate'] = stats['bad_num'] / stats['bin_num']
    stats.reset_index(inplace=True, drop=True)
    del stats['bin_num']

    return df, stats


def calculate_chi2_for_bin_arr(combined_arr: np.ndarray) -> list:
    """计算数组相邻 2 行（bin）的卡方值."""
    chi2_list = []
    arr_len = len(combined_arr)
    for i in range(arr_len - 1):
        chi2_v = chi2_contingency(combined_arr[i:i + 2, 1:])[0]
        chi2_list.append(chi2_v)
    if len(chi2_list) != arr_len - 1:
        raise ValueError("卡方值的数量应该等于数组长度减 1.")
    return chi2_list


def calculate_bad_rate_diff_for_bin_arr(bin_arr: np.ndarray) -> list:
    """计算所有相邻 bin 的 bad_rate 之差."""
    row_sum = bin_arr[:, 1:].sum(axis=1)
    bad_rates = bin_arr[:, 1] / row_sum
    # bad_rates = np.round(bin_arr[:, 1] / row_sum, decimals=4)
    bad_rate_diffs = [j - i for i, j in zip(bad_rates, bad_rates[1:])]
    return bad_rate_diffs


def bad_rate_diff(np_arr) -> float:
    """类似于 chi2_contingency 的小函数.

    用于更新 bin_arr 时，计算输入数组 (两行) 的 bin_rate 之差.

    Parameters
    ----------
    np_arr: np.ndarray
        需要计算 bad_rate 的两个 bin 组成的 ndarray
        bad_num    good_num
        1          2
        3          4

    Returns
    -------
    bad_rate_diff: float
        两个 bin 的 bad_rate 之差

    Examples
    --------
    >>> arr = np.array([[1, 2], [3, 4]])
    >>> bad_rate_diff(arr)   # 0.09523809
    """
    bin_bad_rates = np_arr[:, 0] / np_arr.sum(axis=1)
    # return bin_bad_rates[-1] - bin_bad_rates[0]
    return np.diff(bin_bad_rates)[0]


def update_bin_arr(bin_arr: np.ndarray, metric_between_bin: list, index: int,
                   woe_method='chi2'):
    """将 index 位置的 bin 与其下一个 bin 合并, 并重新计算相邻 bin 之间的度量值（chi2_value
    or bad_rate）.

    Parameters
    ----------
    bin_arr: np.ndarray
        每一行代表一个 bin.
        length L:    [0, 1, 2, 3, ..., L-2, L-1]
    metric_between_bin: list
        相邻 2 个 bin 之间的度量值组成的 list
        length L-1:  [0, 1, 2, 3, ..., L-2]
    index: int
        待合并 bin 的位置 (与其下一个 bin 合并）.
    woe_method: str, default = 'chi2'
        分箱方法不同, 相邻 bin 之间的度量值不同.
        `chi2` 对应卡方值, 'bad_rate' 对应坏样本率.

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
    raw_arr_len = len(bin_arr)
    assert raw_arr_len == len(metric_between_bin) + 1
    assert woe_method in ('chi2', 'bad_rate')

    if woe_method == 'chi2':
        metric = lambda arr: chi2_contingency(arr)[0]
    else:
        metric = bad_rate_diff
    # metric = g if woe_method == 'chi2' else bad_rate_diff

    if index == raw_arr_len - 1:  # 最后 1 个 bin 没法向下合并
        raise ValueError("The last bin must be merged with the above.")

    # 向下合并时 (index == index+1)
    bin_arr[index, 0] = bin_arr[index + 1, 0]
    bin_arr[index, 1:] += bin_arr[index + 1, 1:]
    np_arr = np.delete(bin_arr, index + 1, axis=0)

    if len(np_arr) == 1:  # 如果和完只有一个 bin 了
        return np_arr, None

    # 可以每次重新从头计算 metric_between_bin 来避免这种烧脑地手动更新 metric_between_bin.
    # 但是这种方法很慢, 而且手动更新有助于理解全过程.

    # 倒数第 2 个 bin，则合并最后 2 个 bin
    if index == raw_arr_len - 2:
        metric_between_bin[index - 1] = metric(np_arr[index-1:index+1, 1:])
        _ = metric_between_bin.pop(index)
    else:
        # 如果是第 1 个 bin，只更新第 1 个 metric，然后删除第 2 个 metric
        # 否则，需要更新第 index-1 个 metric 和第 index 个 metric
        if index != 0:
            metric_between_bin[index - 1] = metric(np_arr[index-1:index+1, 1:])
        metric_between_bin[index] = metric(np_arr[index:index + 2, 1:])
        metric_between_bin.pop(index + 1)

    # 更新完成后再次验证长度关系
    assert len(np_arr) - 1 == len(metric_between_bin)
    return np_arr, metric_between_bin


def locate_index(np_arr: np.ndarray, metric: list, i: int) -> int:
    """定位到在 coding 中处理的 bin 的位置.

    第 i 个 bin 需要合并 (与其上一个或者下一个合并). 与其上一个 bin 合并, 可以看成其
    第 i-1 bin 与该 bin 合并.
    1. 如果是第一个 bin, 则只能向下合并, 返回 i.
    2. 如果是最后一个 bin, 则只能和其上一个 bin 合并, 返回 i-1.
    3. 如果是中间某个 bin, 其与上下哪一个 bin 合并取决于度量值（卡方值或坏样本率差值）:
       - 与上一个 bin 的度量值较小, 则与上一个 bin 合并, 返回 i-1;
       - 与上下 bin 的度量值相等, 但上一个 bin 的数量较少, 则与上一个 bin 合并, 返回 i-1;
       - 剩余的情况返回 i.
    """
    total_n = np_arr[:, 1:].sum()
    if i == 0:
        return i
    elif i == len(np_arr) - 1:
        return i - 1
    else:
        metric_v_1 = metric[i - 1]  # 与上一个 bin 的度量值
        metric_v_2 = metric[i]  # 与下一个 bin 的度量值
        above_bin_pct = np_arr[i - 1, 1:].sum() / total_n
        below_bin_pct = np_arr[i, 1:].sum() / total_n

        cond1 = metric_v_1 < metric_v_2  # 与上一个 bin 的 metric 较小
        # 与上下 bin 的卡方值相等, 但上一个 bin 中的样本数较少
        cond2 = metric_v_1 == metric_v_2 and above_bin_pct <= below_bin_pct
        if cond1 or cond2:
            return i - 1
        else:
            return i
