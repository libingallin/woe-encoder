# -*-encoding: utf-8 -*-
"""
@time: 2020/6/1
@author: libingallin@outlook.com
"""
from woe_encoder.cat_bad_rate_woe import CatMaxBinBadRateDiffWOEEncoder
from woe_encoder.cat_chi2_woe import CategoryWOEEncoder
from woe_encoder.num_chi2_woe import NumericalWOEEncoder

__version__ = '1.0'

__author__ = 'libingallin@outlook.com'

__all__ = [
    'CatMaxBinBadRateDiffWOEEncoder',
    'CategoryWOEEncoder',
    'NumericalWOEEncoder',
]
