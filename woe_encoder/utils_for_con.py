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
    """计算相邻 bin 的 bad_rate 之差."""
    row_sum = bin_arr[:, 1:].sum(axis=1)
    bad_rates = bin_arr[:, 1] / row_sum
    # bad_rates = np.round(bin_arr[:, 1] / row_sum, decimals=4)
    bad_rate_diff = [j - i for i, j in zip(bad_rates, bad_rates[1:])]
    return bad_rate_diff
