"""Portable project paths and shared modelling constants.

Set ``YONON_DATA_PATH`` when the local data directory is outside the
repository.  No source or data files are bundled with the public project.
"""

import os


YONON_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.abspath(
    os.environ.get("YONON_DATA_PATH", os.path.join(YONON_PATH, "..", "data"))
)

matrix_path = os.path.join(DATA_PATH, "matrix")
py_path = os.path.join(matrix_path, "披露预测数据")
label_path = os.path.join(matrix_path, "label")

data_fields = [
    "发电总出力(MW)",
    "D日(MW)",
    "D+1日(MW)",
    "D+2日(MW)",
    "光伏出力预测(MW)",
    "风电出力预测(MW)",
    "水电（含抽蓄）总出力(MW)",
    "预测出力(MW)",
    "正备用(MW)",
    "负备用(MW)",
    "一次调频备用(MW)",
]

dataset_name = "qlib158"
factor_folder_path = os.path.join(matrix_path, "factor_data")
dataset_path = os.path.join(factor_folder_path, dataset_name)

SPREAD_THRESHOLD = 50.0
SPREAD_THRESHOLD_BIG = 100.0
SPREAD_CLASSES = ["big_neg", "neg", "neu", "pos", "big_pos"]
SPREAD_CLASS_MAP = {name: i for i, name in enumerate(SPREAD_CLASSES)}

da_price_file = os.path.join(
    matrix_path, "实际运行结果-用电侧", "日前统一结算价.feather"
)
rt_price_file = os.path.join(
    matrix_path, "实际运行结果-用电侧", "实时统一结算价.feather"
)
spread_label_file = os.path.join(label_path, "spread_label.feather")
da_price_latest = os.path.join(YONON_PATH, "data", "日前统一结算价.feather")
rt_price_latest = os.path.join(YONON_PATH, "data", "实时统一结算价.feather")
