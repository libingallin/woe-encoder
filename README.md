# A sklearn-compatible WOE encoder :blush:

## 1. 功能

-   [x] [基于卡方阈值的分箱](#21-基于卡方阈值的分箱)

-   [x] [基于最大分箱数的分箱](#22-基于最大分箱数的分箱)

-   [x] [基于负样本率极大的分箱](#23-基于负样本率极大的分箱)


## 2. WOE 转换方法

### 2.1 基于卡方阈值的分箱

-   **bin 的合并原则：** 将具有最小卡方值的相邻 2 个 bin 合并。
-   **分箱要求（分箱后需要满足的条件）：**
    1.  所有相邻 bin 的卡方值大于阈值 `confidence`（通常为 3.841）；
    2.  每个 bin 的坏样本率 `bad_rate` 不为 0 或 1；
    3.  每个 bin 内的样本比例大于阈值 `bin_pct_threshold`（通常为 5%）；
    4.  （可选）bin 的坏样本率 `bad_rate` 满足单调性或者 U 型（根据业务情况而定）。

### 2.2 基于最大分箱数的分箱

-   **bin 的合并原则：** 将具有最小卡方值的相邻 2 个 bin 合并。
-   **分箱要求（分箱后需要满足的条件）：**
    1.  bin 个数小于等于 `max_bins`（通常为 5～10）；
    2.  每个 bin 的坏样本率 `bad_rate` 不为 0 或 1；
    3.  每个 bin 内的样本比例大于阈值 `bin_pct_threshold`（通常为 5%）；
    4.  （可选）bin 的坏样本率 `bad_rate` 满足单调性或者 U 型（根据业务情况而定）。

### 2.3 基于负样本率极大的分箱

-   **bin 的合并原则：** 将坏样本率差值最小的相邻 2 个 bin 合并。
-   **分箱要求（分箱后需要满足的条件）：**
    1.  bin 个数小于等于 `max_bins`（通常为 5～10）；
    2.  每个 bin 的坏样本率 `bad_rate` 不为 0 或 1；
    3.  每个 bin 内的样本比例大于阈值 `bin_pct_threshold`（通常为 5%）；
    4.  ~~（可选）bin 的坏样本率 `bad_rate` 满足单调性或者 U 型（根据业务情况而定）~~。



以上 3 种分箱方法，对于不同类型的特征有不同的处理方法。**特征取值分组：**

-   连续型特征。每个 unique 值作为一个 bin 来初始化，然后将坏样本数（率）连续为 0 或 1 的 bin 合并，再对照分箱要求做后续处理。
-   离散无序型特征。先以每个 unique 值作为一个 bin 来初始化，再对照分箱要求做后续处理。或者按照以下情况分开处理：
    -   初始化的 bin 数小于等于 `max_bins`，再对照分箱要求做后续处理；
    -   初始化的 bin 数大于 `max_bins`，以坏样本率 `bad_rate` 对每个 bin  编码后当作连续型特征处理。
-   离散有序型特征。每个特征取值当作一个 bin，然后按照特征取值的期望顺序 `value_order_dict` 对每个 bin 进行排序，再根据要求处理。在处理过程中，始终维持有序性。



可以指定了特征中不参与分箱的特殊值 `special_values: list`（比如缺失值标识），需要将数据分成 2 部分：

-   其余值使用正常的分箱方法，得到分箱结果；
-   特征值单独作为一个 bin。如果该 bin 不符合分箱要求时：
    -   bin 的坏样本率为 0 或 1
        -   处理方法：坏样本数 +1 或 -1 再计算坏样本率。若 bad_rate 为 0， 则将 bad_cnt 调整为 1，再计算 bad_rate；若 bad_rate 为 1，则将 bad_cnt 调整为 bad_cnt-1，再计算 bad_rate。
        -   理由：`woe=ln(bad_rate/good_rate)`，分子分母都不能为 0。bad_rate 为 0 或 1，表示该 bin 内的好坏样本数量差异达到了极致。为了能使 woe 正常计算，需要做调整以维持 bin 内好坏样本数量差异极大这一事实。
    -   bin 内样本比例小于阈值。仍做单独一个 bin，不做任何处理。
    -   不满足单调性。特征值组成的 bin 与正常分箱形成的 bin 之间不考虑单调性的问题。



为什么分箱要求是这样的顺序？

TODO(libing)


## 3. Examples

```python
from woe_encoder import CategoryWOEEncoder
from woe_encoder import ContinuousWOEEncoder
```

## 4. References

1. https://github.com/scikit-learn-contrib/category_encoders/blob/master/category_encoders/woe.py
2. https://blog.csdn.net/qq_40913605/article/details/88133449


