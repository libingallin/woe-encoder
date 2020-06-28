# -*-encoding: utf-8 -*-
"""
@time: 2020/6/9
@author: bingli
"""
import pandas as pd
from scipy.stats import chi2_contingency


def initialize_bins(df, col_name: str, target_col_name: str,
                    value_order_dict=None) -> pd.DataFrame:
    """初始化生成每个 bin. 每个 unique 值当作一个 bin.

    如果有 value_order，则在生成的 pd.DataFrame 末尾增加一列 (order) 表示顺序。
    对结果进行排序：如果没有 value_order, 则根据 bad_rate 升序 sort, 否则根据
    order 值升序 sort。

    Parameters
    ----------
    df: pd.DataFrame
        原始数据集. 含有 col_name 和 target_col_name 列，且不含有缺失值和特殊值
    col_name: str
        要转换的特征名
    target_col_name: str
        目标列名
    value_order_dict: dict or None (default)
        该特征值是否有序。默认值无序, 否则传入一个字典, 表示每个值的顺序等级 (递增)。

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
    df_len = len(df)
    bin_df = df.groupby(col_name)[target_col_name].agg(
        bin_num='count', bad_num='sum')
    bin_df.reset_index(inplace=True)
    bin_df['good_num'] = bin_df['bin_num'] - bin_df['bad_num']
    bin_df['bad_rate'] = bin_df['bad_num'] / bin_df['bin_num']
    bin_df['bin_pct'] = bin_df['bin_num'] / df_len

    # index 永远是从 0 开始
    if value_order_dict:
        # value_order_dict = {[key]: v for key, v in value_order_dict.items()}
        bin_df['order'] = bin_df[col_name].map(value_order_dict)
        bin_df = bin_df.sort_values('order', ignore_index=True)
    else:
        bin_df = bin_df.sort_values('bad_rate', ignore_index=True)

    bin_df[col_name] = bin_df[col_name].map(lambda x: [x])
    return bin_df


def locate_index(bin_df: pd.DataFrame, comparison: list, i: int) -> int:
    """确定要被更新的 index (return). 该 index 表明第 index 个 bin 与其下一个 bin 合并.

    1. 如果 i 等于 0, 只能向下合并, index 为 0.
    2. 如果 i 是最后一个 bin, 即 bin_df 的最后一行, 那么只能向上合并, 但可以看成倒数
    第二个 bin 向下合并, 此时 index=i-1.
    3. 如果 i 是以上 2 种情况的其他值. 需要该 bin (第 i 个) 与相邻 bin (上和下) 的
    比较来确定.
        - 如果与上一个 bin 的值 (卡方值/bad_rate 差值) 较小, 那么向上合并, 看作上一个
        bin 向下合并, 此时 index=i-1;
        - 如果与上一个 bin 的卡方值和与下一个 bin 的值 (卡方值/bad_rate 差值) 相等,
        并且上一个 bin 的数据量较少, 也向上合并, 此时 index=i-1.
        - 其他情况, 该 bin (第 i 个) 向下合并, index=i.

    Parameter
    ---------
    bin_df: pd.DataFrame
        含有 bin 及其相关简单统计. 每 1 行代表 1 个 bin.
    comparison: list
        bin_df 中连续相邻 bin 的卡方值或 bad_rate 之差.
    i: int
        i 表示 bin_df 中的索引/ bin 的索引.

    Returns
    -------
    int. 根据输入来确定 index 的位置.
    """
    if i == 0:
        return i
    elif i == len(bin_df) - 1:
        return i - 1
    else:
        chi2_1 = comparison[i - 1]  # 与上一个 bin 的卡方值
        chi2_2 = comparison[i]  # 与下一个 bin 的卡方值

        cond1 = chi2_1 < chi2_2
        cond2 = (chi2_1 == chi2_2) and (bin_df[i - 1, 5] <= bin_df[i, 5])
        if cond1 or cond2:
            return i - 1
        else:
            return i


def update_bin_df(bin_df, index, value_order_dict=None):
    """合并 bin_df 中第 index 个 bin 和其下一个 bin."""
    total_n = bin_df.iloc[:, 1].sum()
    bin_df.iloc[index, 0].extend(bin_df.iloc[index + 1, 0])
    bin_df.iloc[index, 1:4] += bin_df.iloc[index + 1, 1:4]
    bin_df.iloc[index, 4] = bin_df.iloc[index, 2] / bin_df.iloc[index, 1]
    bin_df.iloc[index, 5] = bin_df.iloc[index, 1] / total_n

    # 如果特征值要求有序，那么合并后 bin_df 中字段 order 也要保持有序
    if value_order_dict:
        bin_df.iloc[index, 6] = (bin_df.iloc[index, 6]
                                 + bin_df.iloc[index + 1, 6]) / 2

    bin_df.drop(index=index+1, inplace=True)
    # index 永远是从 0 开始
    bin_df.reset_index(drop=True, inplace=True)

    return bin_df


def process_special_values(df, col: str, target_col_name: str,
                           special_value_list: list):
    """特殊值（含缺失值）不参与分箱，单独处理。"""
    df_len = len(df)
    df_special = df[df[col].isin(special_value_list)]
    # df = df[~df.isin(df_special)].dropna(how='all')
    df = df[~df[col].isin(special_value_list)]

    stats = df_special.groupby(col)[target_col_name].agg(
        bad_num='sum', bin_num='count')
    stats.reset_index(inplace=True)
    stats[col] = stats[col].apply(lambda x: [x])
    stats['good_num'] = stats['bin_num'] - stats['bad_num']
    stats['bad_rate'] = stats['bad_num'] / stats['bin_num']
    stats['bin_pct'] = stats['bin_num'] / df_len

    return df, stats


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


def calculate_bad_rate_diff_for_bin_df(bin_df: pd.DataFrame):
    """计算相邻 bin 的 bad_rate 之差."""
    bad_rates = bin_df['bad_rate'].values
    bad_rate_diffs = [j - i for i, j in zip(bad_rates, bad_rates[1:])]

    assert len(bin_df) - 1 == len(bad_rate_diffs)
    return bad_rate_diffs
