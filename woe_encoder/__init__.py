# -*-encoding: utf-8 -*-
"""
@time: 2020/6/1 17:43
@author: libingallin@outlook.com
"""
from woe_encoder.cat_max_bin_bad_rate_diff_woe import CatMaxBinBadRateDiffWOEEncoder
from woe_encoder.cat_normal_woe import CategoryWOEEncoder
from woe_encoder.num_normal_woe import NumericalWOEEncoder

__version__ = '1.0'

__author__ = 'libingallin@outlook.com'

__all__ = [
    'CatMaxBinBadRateDiffWOEEncoder',
    'CategoryWOEEncoder',
    'NumericalWOEEncoder',
]
