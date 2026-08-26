# ============================================================
# deploy/_cfg.py — 加载 0_config.py 的共享加载器（内部模块）
#
# 0_config.py 以数字开头，无法用 import 直接导入，统一通过本模块加载。
# 所有脚本用法:  from _cfg import cfg
# 脚本内一切路径/设置均从 cfg.xxx 读取。
# ============================================================
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

_spec = importlib.util.spec_from_file_location(
    "deploy_config", os.path.join(_HERE, "0_config.py"))
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)


# ── code/ 子目录 sys.path 引导 ─────────────────────────────
# 目录整理后，代码实现按模型分组进了 code/common、code/v9、code/v8.1。
# 任何脚本只要 import _cfg（所有脚本都从这里读配置），就让这三个目录可被裸名
# import：code/common 供 _verify / _factors_supply 等共享模块；code/v9、code/v8.1
# 供 v9_wrappers / v8_wrappers（joblib pickle 反序列化需要按裸名解析类路径）。
# 这样移动脚本无需各自维护相对 import，根目录脚本也无需改动。
for _sub in ('code/common', 'code/v9', 'code/v8.1'):
    _p = os.path.join(_HERE, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def latest_disclosure_date():
    """最新披露日（各脚本共用）。

    权威来源是 披露预测数据.feather 宽表（1_ingest_xlsx 每次导入时同步维护，
    Date 索引排序后取最后一天）。不用扫描逐通道矩阵——矩阵有 6000+ 个文件，
    逐个读耗时 ~11s，且旧的实现只扫 sorted()[:30] 会漏掉真正的供给通道，
    导致"UI 披露日不更新 / 预测窗口停留在 22~27"。
    """
    raw = cfg.DISCLOSURE_RAW
    if os.path.exists(raw):
        import pyarrow.feather as pyf
        import pandas as pd
        try:
            df = pyf.read_table(raw).to_pandas()
            dates = df.index.get_level_values(0).astype(str)
            return str(sorted(dates.unique())[-1])
        except Exception:
            pass
    # 兜底：扫描矩阵（只挑 date×96 的供给通道，跳过 0.xx* 线路组合）
    mx = cfg.DISCLOSURE_MATRIX
    if os.path.isdir(mx):
        latest = None
        for f in os.listdir(mx):
            if not f.endswith('.feather'):
                continue
            if f[0].isdigit():     # 0.05*回蝶甲乙... 线路组合文件跳过
                continue
            try:
                df = pd.read_feather(os.path.join(mx, f))
                mx_d = str(df.index.max())
                if latest is None or mx_d > latest:
                    latest = mx_d
            except Exception:
                continue
        if latest:
            return latest
    return None


def latest_actual_date():
    """最新实际结果日（实际矩阵的日前统一结算价文件，Date 索引取最大）。"""
    import pandas as pd
    p = os.path.join(cfg.ACTUAL_MATRIX, '日前统一结算价.feather')
    if os.path.exists(p):
        try:
            df = pd.read_feather(p)
            return str(df.index.max())
        except Exception:
            pass
    return None


def active_model_path():
    """返回 ACTIVE_MODEL 对应的最新模型 joblib 路径（读指针文件；无则 None）。
    UI 指标卡片 / 2B 控制台共用，避免各自重复读取指针。"""
    if os.path.exists(cfg.LATEST_MODEL_FILE):
        import json
        try:
            ptr = json.load(open(cfg.LATEST_MODEL_FILE))
            p = ptr.get(cfg.ACTIVE_MODEL)
            if p and os.path.exists(p):
                return p
        except Exception:
            pass
    return None
