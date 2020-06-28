# -*-encoding: utf-8 -*-
"""
@time: 2020-05-26
@author: bingli
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gen_special_value_list(df: pd.DataFrame, col_name: str,
                           imputation_value=None, special_value_list=None):
    """将缺失值的填充值 imputation_value 与特殊值组成的列表组成一个新的列表。

    缺失值（填充）也相当于特殊值，放到 special_value_list 里一起处理。

    Parameters
    ----------
    df: pd.DataFrame
    col_name: str
        特征名
    imputation_value:
        缺失值的填充值
    special_value_list: list
        需要特殊对待的值

    Returns
    -------
    special_value_flag: bool or None (default)
        表示有需要的特殊对待的值
    special_value_list: list or None (default)
        需要特殊对待值组成的列表
    """
    special_value_flag = False
    if imputation_value is not None:
        # Fill missing values with the specified value.
        df[col_name] = df[col_name].fillna(imputation_value)
        if special_value_list is not None:
            special_value_list.append(imputation_value)
        else:
            special_value_list = [imputation_value]
        special_value_flag = True
    else:
        if special_value_list is not None:
            special_value_flag = True
    return special_value_flag, special_value_list


def calculate_woe_iv(bin_df: pd.DataFrame, bad_col: str, good_col: str,
                     regularization=1.):
    """计算 pd.DataFrame 的正则化 WOE 值和 IV 值.

    Parameters
    ----------
    bin_df: pd.DataFrame
        应该是分箱结束后的统计.
    bad_col: str
        表示 1 的字符串名称.
    good_col: str
        表示 0 的字符串名称.
    regularization: float, default to 1.0
        防止计算 woe 时分母为 0

    Examples
    --------
    bad_cnt  good_cnt  woe                             iv
    a    b    e = np.log((a/bad_sum) / (b/good_sum))   (a/bad_sum-b/good_sum)*e
    c    d    f = np.log((c/bad_sum) / (d/good_sum))   (c/bad_sum-d/good_sum)*f
    bad_sum = a + c
    good_sum = b + d
    """
    bad_sum = bin_df[bad_col].sum()
    good_sum = bin_df[good_col].sum()

    bin_df['woe'] = bin_df.apply(
        lambda row: np.log(
            ((row[bad_col] + regularization) / (bad_sum + 2 * regularization)) /
            ((row[good_col] + regularization) / (good_sum + 2 * regularization))),
        axis=1)
    bin_df['iv'] = bin_df.apply(
        lambda row: (row[bad_col]/bad_sum - row[good_col]/good_sum) * row['woe'],
        axis=1)
    return bin_df[['woe', 'iv']]


def _compare(arr, start=0, stop=None, increase=True) -> bool:
    """判断一个序列的单调性.

    pd.Series 的 attribute is_monotonic 也可以用来判断单调性.
    """
    if stop is None:
        stop = len(arr)
    if increase:
        flag = all([i <= j for i, j in
                    zip(arr[start:stop], arr[start + 1:stop])])
        return flag
    else:
        flag = all([i >= j for i, j in
                    zip(arr[start:stop], arr[start + 1:stop])])
        return flag


def if_monotonic(bad_rates, u: bool) -> bool:
    """"判断一个序列是否满足单调性/U 型.

    - 只有 2 个值的时候肯定单调, 返回 True.
    - 单调递增/减都算单调, 返回 True.
    - 如果接受 (正/倒) U 型, 返回 True.
    """
    bad_rates_len = len(bad_rates)
    if bad_rates_len == 2:  # 只有 2 个值的时候肯定单调
        return True

    up_all = _compare(bad_rates, increase=True)
    down_all = _compare(bad_rates, increase=False)
    if up_all or down_all:
        return True

    # 如果非单调，但是接受 U 型
    if u:
        # 如果存在相邻的 2 个值相等，那么即便呈 U 型，也是不严格的
        any_equal = any([i == j for i, j in zip(bad_rates, bad_rates[1:])])
        if any_equal:
            return False

        min_idx = np.argmin(bad_rates)
        max_idx = np.argmax(bad_rates)

        # 倒 U 型，极大值不在首尾
        if max_idx not in [0, bad_rates_len - 1]:
            left_up = _compare(bad_rates, stop=max_idx, increase=True)
            right_down = _compare(bad_rates, start=max_idx, increase=False)
            if left_up and right_down:
                return True
        # 正 U 型，极小值不在首尾
        if min_idx not in [0, bad_rates_len - 1]:
            left_down = _compare(bad_rates, stop=min_idx, increase=False)
            right_up = _compare(bad_rates, start=min_idx, increase=True)
            if left_down and right_up:
                return True
    return False


def set_color_for_iv(ivs) -> (list, dict):
    """Different color for different iv range.

    用不同的颜色代表不同的 IV 值范围. 返回值中 legends 用来绘制 legend.
    """
    colors = []
    for iv in ivs:
        if iv < 0.02:
            colors.append('gray')
        elif iv < 0.1:
            colors.append('lightgreen')
        elif iv < 0.3:
            colors.append('lightskyblue')
        elif iv < 0.5:
            colors.append('gold')
        else:
            colors.append('lightcoral')

    legends = {
        '[0.00, 0.02) Useless': 'gray',
        '[0.02, 0.10) Weak': 'lightgreen',
        '[0.10, 0.30) Medium': 'lightskyblue',
        '[0.30, 0.50) Strong': 'gold',
        '[0.50, +inf) Suspicious': 'lightcoral'}

    return colors, legends


def plot_ivs(col_iv_dict: dict, figsize=(15, 6)):
    """绘制每个特征的 IV 值.

    Parameters
    ----------
    col_iv_dict: dict
        (column_name, iv_value)
    figsize: tuple of float
        图的尺寸
    """
    colors, legends = set_color_for_iv(col_iv_dict.values())

    x = np.arange(len(col_iv_dict))
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, col_iv_dict.values(), color=colors, edgecolor='k')
    ax.set_xticks(x)
    ax.set_xticklabels(col_iv_dict.keys(), rotation=90)
    ax.tick_params(axis='both', which='major', labelsize=12, direction='in')
    ax.set_ylabel('IV', fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    legend_labels = list(legends.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=legends[label])
               for label in legend_labels]
    ax.legend(handles, legend_labels, shadow=True, fancybox=True, fontsize=14)
    plt.show()
