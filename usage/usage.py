# -*-encoding: utf-8 -*-
"""
@time: 2020-05-26
@author: libingallin@outlook.com
"""
import time

import pandas as pd
from sklearn.datasets import load_boston
from woe_encoder import CategoryWOEEncoder, NumericalWOEEncoder
from category_encoders import WOEEncoder

# Prepare data
bunch = load_boston()
x = pd.DataFrame(bunch.data, columns=bunch.feature_names)
y = bunch.target > 22.5
x['y'] = y


if __name__ == '__main__':
    # my category woe encoder
    x_1 = x.copy()
    time_0 = time.time()
    my_cat = CategoryWOEEncoder('RAD', 'y', max_bins=1000, bin_pct_threshold=0.0)
    x_1 = my_cat.fit_transform(x_1)
    time_1 = time.time()

    # sklearn category woe encoder
    x_2 = x.copy()
    time_2 = time.time()
    # sklearn_cat = WOEEncoder(cols=['CHAS', 'RAD']).fit(x_2, y)
    sklearn_cat = WOEEncoder(cols=['CHAS', 'RAD'], regularization=0.5).fit(x_2, y)
    x_2 = numeric_dataset = sklearn_cat.transform(x_2)
    time_3 = time.time()

    # print(time_1-time_0, time_3-time_2)
    # print(x_1['RAD'])
    # print()
    # print(x_2['RAD'])

    time_4 = time.time()
    my_num = NumericalWOEEncoder('CRIM', 'y', bin_pct_threshold=0.05,
                                 need_monotonic=True, u=False)
    x_1 = my_num.fit_transform(x_1)
    time_5 = time.time()
    print(time_5 - time_4)
    print(my_num.woe_, my_num.iv_)
    print(my_num.bin_result_)
    print(x['CRIM'], '\n', x_1['CRIM'])

    from woe_encoder import CatMaxBadRateDiffWOEEncoder