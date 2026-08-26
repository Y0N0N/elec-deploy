# ============================================================
# deploy/0_config.py — 每日自动化部署·配置加载器
#
# 【重要】这是所有脚本读取配置的唯一入口。
#   用户可改的设置存放在 config.json（推荐用 UI 客户端 ui_client.py 修改，
#   也可直接手改 config.json）。本文件负责加载 JSON 并计算派生路径。
#   不要直接改本文件里的数值——改 config.json 或打开 UI 客户端。
#
# 日常你要改的设置（都在 config.json / UI 里）：
#   - active_model               切换推理/重建的模型
#   - predict_window             预测窗口（D-4~D+1）
#   - spread 阈值                 价差业务阈值
#   - factor_rebuild             重建模式（full/tail）+ 尾部天数
#   - training                   模型训练参数
# ============================================================

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONFIG_FILE = os.path.join(_HERE, "config.json")


def _path_from_config(value, default):
    """Resolve configured paths relative to this deploy directory."""
    path = value if value else default
    if not os.path.isabs(path):
        path = os.path.join(_HERE, path)
    return os.path.abspath(path)

def _load_json():
    if not os.path.exists(_CONFIG_FILE):
        raise FileNotFoundError(
            f"找不到配置文件 {_CONFIG_FILE}\n"
            f"请把 config.json 放在 {_HERE} 下（或用 UI 客户端重建）")
    with open(_CONFIG_FILE, encoding='utf-8') as f:
        return json.load(f)


_C = _load_json()

# ════════════════════════════════════════════════════════════
# 一、基础环境（来自 config.json）
# ════════════════════════════════════════════════════════════
PROJECT_DIR = _path_from_config(_C.get("project_dir"), _HERE)
DATA_DIR    = _path_from_config(_C.get("data_dir"), os.path.join(_HERE, "..", "data"))
# 模型/上传目录默认放在 deploy 下（可用 UI 客户端在「模型与路径」页调整）
MODEL_DIR   = _path_from_config(_C.get("model_dir"), os.path.join(_HERE, "models"))
UPLOAD_DIR  = _path_from_config(_C.get("upload_dir"), os.path.join(_HERE, "upload"))
# 本部署只使用 DATA_DIR 下的数据；请将该目录指向经过授权的本地数据副本。

# ════════════════════════════════════════════════════════════
# 二、输入输出路径（由上面的 data_dir / project_dir 派生，一般不用改）
# ════════════════════════════════════════════════════════════
DISCLOSURE_RAW  = os.path.join(DATA_DIR, "披露预测数据.feather")          # 披露合并宽表
ACTUAL_RAW      = os.path.join(DATA_DIR, "实际运行结果-用电侧.feather")    # 实际结果合并
DISCLOSURE_MATRIX = os.path.join(DATA_DIR, "matrix", "披露预测数据")       # 逐通道矩阵 date×96
ACTUAL_MATRIX   = os.path.join(DATA_DIR, "matrix", "实际运行结果-用电侧")    # 4 个矩阵 date×24
LABEL_DIR       = os.path.join(DATA_DIR, "matrix", "label")                # spread_label 等
FACTOR_DIR      = os.path.join(DATA_DIR, "matrix", "factor_data", "qlib158")  # 因子库 .fea
OUTPUT_DIR      = os.path.join(_HERE, "output")                            # 推理结果 csv
LATEST_MODEL_FILE = os.path.join(_HERE, "latest_model.json")              # 最新模型指针
MANIFEST_FILE   = os.path.join(_HERE, "manifest.json")                    # 已处理文件账本
OUTPUT_FILE_PREFIX = "预测"

# ════════════════════════════════════════════════════════════
# 三、模型注册表（来自 config.json）
# ════════════════════════════════════════════════════════════
MODEL_REGISTRY = _C.get("model_registry", {})
ACTIVE_MODEL   = _C.get("active_model", "v7")

def model_trigger_rule(model_key):
    """模型触发规则：'value_driven'（v8/v8.1 数值驱动）| 'class'（v7 分类驱动）。
    默认按 predict_mode 推断（spread 用 value_driven，price 用 class）。"""
    reg = MODEL_REGISTRY.get(model_key) or {}
    if reg.get('trigger_rule'):
        return reg['trigger_rule']
    return 'value_driven' if reg.get('predict_mode') == 'spread' else 'class'

# ════════════════════════════════════════════════════════════
# 四、价差任务参数（来自 config.json）
# ════════════════════════════════════════════════════════════
SPREAD_THRESHOLD     = float(_C["spread"]["threshold"])       # τ_minor 元/MWh
SPREAD_THRESHOLD_BIG = float(_C["spread"]["threshold_big"])   # τ_big 元/MWh
SPREAD_CLASSES = ["big_neg", "neg", "neu", "pos", "big_pos"]
SPREAD_CLASS_MAP = {"big_neg": 0, "neg": 1, "neu": 2, "pos": 3, "big_pos": 4}

# ════════════════════════════════════════════════════════════
# 五、预测窗口（来自 config.json）
# ════════════════════════════════════════════════════════════
# 参考日 D = 最新披露日 - 1 天；窗口 = [D - back_days, D + fwd_days]
PREDICT_BACK_DAYS = int(_C["predict_window"]["back_days"])
PREDICT_FWD_DAYS  = int(_C["predict_window"]["fwd_days"])

# ════════════════════════════════════════════════════════════
# 六、因子重建设置（来自 config.json）
# ════════════════════════════════════════════════════════════
FACTOR_REBUILD_MODE = _C["factor_rebuild"].get("mode", "full")   # full | tail
TAIL_DAYS           = int(_C["factor_rebuild"].get("tail_days", 60))
REBUILD_SP_WOW      = bool(_C["factor_rebuild"].get("rebuild_sp_wow", True))

# ════════════════════════════════════════════════════════════
# 七、模型训练参数（来自 config.json）
# ════════════════════════════════════════════════════════════
_t = _C.get("training", {})
TRAIN_VALID_DAYS     = int(_t.get("valid_days", 30))       # 验证集天数（红线：不混入训练）
MODEL_N_ESTIMATORS   = int(_t.get("n_estimators", 200))
MODEL_EARLY_STOPPING = int(_t.get("early_stopping", 30))
MODEL_RANDOM_STATE   = int(_t.get("random_state", 42))
MODEL_FIXED_PARAMS   = dict(_t.get("fixed_params", {
    "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8,
    "colsample_bytree": 1.0, "tree_method": "hist", "n_jobs": 8,
}))

# ════════════════════════════════════════════════════════════
# 八、披露 xlsx 解析规则（结构性常量，不用改）
# ════════════════════════════════════════════════════════════
# 披露 xlsx 的 23 个 sheet 中，只有下面两类会进入矩阵（已用 07-27 实测 182/182 验证）：
#   1. 供给侧 sheet：列名 = 非时间列 " | " 连接（如 "预测 | 统调负荷(MW)"）
#   2. 必开必停机组（群）约束 sheet：列名 = 机组群名|台数|电厂ID|电厂名称|机组ID|机组名称|数据类型
SUPPLY_SHEET_KEYWORDS = [
    "负荷预测信息", "地方电预测信息", "发电总出力预测信息", "现货新能源总出力",
    "统调新能源出力信息", "水电", "抽蓄电站出力计划", "备用预测信息",
]
CONSTRAINT_COL_FIELDS = ["机组群名", "机组群名称", "机组台数", "电厂ID", "电厂名称",
                         "机组ID", "机组名称", "数据类型"]
ACTUAL_ITEM_MAP = {
    "日前统一结算价": "日前统一结算价",
    "实时统一结算价": "实时统一结算价",
    "日前成交电量":   "日前成交电量",
    "实际用电量":     "实际用电量",
}


def _key_pattern(model_key):
    """model_key 对应的匹配前缀/后缀。

    pattern 形如 'xgb_v8.1_{ts}.joblib' → (prefix='xgb_v8.1_', suffix='.joblib')。
    无 pattern 或 pattern 无 {ts}（固定名，如 'xgb_v8.joblib'）时，固定名本身也作为候选。"""
    reg = MODEL_REGISTRY.get(model_key) or {}
    pattern = reg.get('pattern')
    if pattern and '{ts}' in pattern:
        prefix, _, suffix = pattern.partition('{ts}')
        return prefix, suffix
    if pattern:
        return pattern, ''   # 固定名
    return f'{model_key}.joblib', ''


def scan_models_by_key(model_key):
    """返回 model_key 对应的全部模型 joblib 路径（按修改时间升序，旧的在前）。

    供归档/清理使用：与归档逻辑约定“同前缀，非最新者进 archive/models”。
    匹配用“去掉 {ts} 后的精确前缀”，避免误收同前缀的其它模型
    （如 'xgb_v8' 前缀不会命中 'xgb_v8.1_*'）。固定名（xgb_v7.joblib）也会纳入。"""
    prefix, suffix = _key_pattern(model_key)
    cands = [f for f in os.listdir(MODEL_DIR)
             if (suffix and f.startswith(prefix) and f.endswith(suffix))]
    fixed = prefix.rstrip('_') + '.joblib'
    if os.path.isfile(os.path.join(MODEL_DIR, fixed)):
        cands.append(fixed)
    paths = [os.path.join(MODEL_DIR, f) for f in cands
             if os.path.isfile(os.path.join(MODEL_DIR, f))]
    return sorted(paths, key=os.path.getmtime)   # 旧 → 新


def resolve_latest_model(model_key):
    """按“同 key 前缀的最新模型”解析路径（2A/2B/UI 共用的唯一来源）。

    与旧实现（读 latest_model.json 指针）不同：不再信任可能悬空的指针文件，
    而是直接扫 MODEL_DIR 取该 key 下修改时间最新的 joblib。返回 None 表示未找到。"""
    paths = scan_models_by_key(model_key)
    return paths[-1] if paths else None


# ════════════════════════════════════════════════════════════
# 运行检查（脚本启动时自检）
# ════════════════════════════════════════════════════════════
def validate():
    """校验关键配置/目录，缺失则打印红色警告。返回缺失目录列表。"""
    missing = [p for p in [UPLOAD_DIR, DISCLOSURE_MATRIX, ACTUAL_MATRIX,
                           LABEL_DIR, FACTOR_DIR, MODEL_DIR] if not os.path.isdir(p)]
    if missing:
        print("⚠️  以下目录不存在（可能首次运行或 data_dir 配置有误）:")
        for p in missing:
            print(f"    {p}")
        print("   请在 config.json（或用 UI 客户端）核对 data_dir。")
    if ACTIVE_MODEL not in MODEL_REGISTRY:
        print(f"⚠️  active_model='{ACTIVE_MODEL}' 不在 model_registry 中，请检查 config.json")
    return missing


if __name__ == "__main__":
    print(f"配置加载成功。数据目录: {DATA_DIR} | 活跃模型: {ACTIVE_MODEL}")
    validate()
