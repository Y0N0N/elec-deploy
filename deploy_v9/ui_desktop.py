#!/usr/bin/env python
# ============================================================
# deploy/ui_desktop.py — 配置管理 UI（桌面版，tkinter）
#
# 作用：用桌面窗口查看/修改 config.json，比网页版更紧凑、
#       支持弹窗选目录，并可一键运行每日流程（1→2A→2B）。
#
# 用法：
#   Windows: 双击 run_ui.bat
#   macOS:   ./run_ui.command   （或 python3 ui_desktop.py）
#
# 依赖：仅 Python 标准库（tkinter）。pandas 为可选（读取状态用，
#       缺失时状态卡片显示 —）。
# ============================================================
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
CONFIG_FILE = os.path.join(_HERE, 'config.json')
DEFAULTS_FILE = os.path.join(_HERE, 'config.defaults.json')

# 结构定义：每个可编辑字段 (JSON路径, 标签, 类型, 校验规则, 说明)
# JSON路径用 "." 分隔；类型: 'text' | 'number' | 'select' | 'bool'
FIELDS = [
    # ── 模型 ──
    ('active_model', '当前模型', 'select', '模型key',
     '推理和重建使用哪个模型（见下方模型列表）'),
    # ── 预测窗口 ──
    ('predict_window.back_days', '往前预测天数 (D-back)', 'number', 'int>=0',
     '从参考日 D 往前预测几天（D-4 表示最近4天没有实际结果的）'),
    ('predict_window.fwd_days', '往后预测天数 (D+fwd)', 'number', 'int>=0',
     '从参考日 D 往后预测几天（D+1 表示次日）'),
    # ── 价差阈值 ──
    ('spread.threshold', '小偏差阈值 (τ_minor, 元/MWh)', 'number', 'float>0',
     '|价差| ≤ 此值视为正常，不预警'),
    ('spread.threshold_big', '大偏差阈值 (τ_big, 元/MWh)', 'number', 'float>0',
     '|价差| > 此值视为大偏差，重点预警（应大于 τ_minor）'),
    # ── 因子重建 ──
    ('factor_rebuild.mode', '因子重建模式', 'select', 'enum:full,tail',
     'full=全量重算（最稳）；tail=只重算最近 N 天（快）'),
    ('factor_rebuild.tail_days', '尾部窗口天数', 'number', 'int>=35',
     'tail 模式下重算最近多少天（滚动因子需 35 天历史）'),
    ('factor_rebuild.rebuild_sp_wow', '重建价差状态因子 sp_wow', 'bool', 'bool',
     '依赖最新实际结果；建议保持开启'),
    # ── 模型训练 ──
    ('training.valid_days', '验证集天数', 'number', 'int>=7',
     '最后多少天做验证集（early stopping）。红线：验证集不混入训练'),
    ('training.n_estimators', '最大树数', 'number', 'int>=50',
     'XGBoost 分类头/回归头最大迭代数'),
    ('training.early_stopping', '早停轮数', 'number', 'int>=10',
     '验证集不再改善多少轮就停止训练'),
    ('training.random_state', '随机种子', 'number', 'int>=0',
     '保证可复现，一般不用改'),
    # ── XGBoost 超参 ──
    ('training.fixed_params.max_depth', 'max_depth 树深度', 'number', 'int 3-15',
     '决策树最大深度'),
    ('training.fixed_params.learning_rate', 'learning_rate 学习率', 'number', 'float 0.001-0.5',
     '步长，越小越稳但越慢'),
    ('training.fixed_params.subsample', 'subsample 子采样', 'number', 'float 0.1-1.0',
     '每棵树用多少比例的样本'),
    ('training.fixed_params.colsample_bytree', 'colsample_bytree 列采样', 'number', 'float 0.1-1.0',
     '每棵树用多少比例的特征'),
    ('training.fixed_params.n_jobs', 'n_jobs 线程数', 'number', 'int 1-64',
     '项目实测 8 线程最快，一般不用改'),
    # ── 高级：路径 ──
    ('data_dir', '数据目录', 'text', 'text',
     '所有数据所在目录（一般不动）'),
    ('project_dir', '项目目录', 'text', 'text',
     '项目根目录（一般不动）'),
    ('model_dir', '模型目录', 'text', 'text',
     '训练好的模型存放处（默认在 deploy/models）'),
    ('upload_dir', '上传目录', 'text', 'text',
     '每日新 xlsx 放置处（默认在 deploy/upload）'),
]

# 配置分成两个 tab：模型与路径 / 模型参数
TAB_MODEL_PATH = ('模型与路径',
                  lambda p: p == 'active_model'
                  or p in ('data_dir', 'project_dir', 'model_dir', 'upload_dir'))
TAB_PARAMS = ('模型参数',
              lambda p: not (p == 'active_model'
                             or p in ('data_dir', 'project_dir', 'model_dir',
                                      'upload_dir')))
CONFIG_TABS = [TAB_MODEL_PATH, TAB_PARAMS]

# ── BUG-4 修复：模型 → 无效配置字段映射 ───────────────────────
# 某些模型训练时不读取 config 的特定字段，改这些字段不生效。
# 切换到这类模型时，把这些字段的输入框置灰（state='disabled'）并加提示，
# 避免用户以为改了会生效（此前 training.valid_days 对 v9/v9.1 是死配置）。
#   key: 模型 key（'*' = 所有模型）
#   value: {字段 path: 提示文本}
MODEL_IGNORED_FIELDS = {
    # v9 / v9.1 训练脚本硬编码 --valid-days 60（2A_rebuild.py），
    # 不读 config 的 training.valid_days
    'v9': {
        'training.valid_days': 'v9/v9.1 训练硬编码验证集 60 天，此设置仅对 v8/v8.1 生效',
    },
    'v9.1': {
        'training.valid_days': 'v9/v9.1 训练硬编码验证集 60 天，此设置仅对 v8/v8.1 生效',
    },
}
MODEL_IGNORED_FIELDS['*'] = {}

# 模型参数页内的分组标题
PARAM_SECTIONS = [
    ('预测窗口',        lambda p: p.startswith('predict_window')),
    ('价差阈值',        lambda p: p.startswith('spread')),
    ('因子重建',        lambda p: p.startswith('factor_rebuild')),
    ('模型训练',        lambda p: p.startswith('training') and 'fixed_params' not in p),
    ('XGBoost 超参',    lambda p: 'fixed_params' in p),
]

# 模型指标卡片字段映射: (metrics key, 中文标签, 格式化类型)
# 值来源: 模型 dict['metrics']['valid']（新模型）或平铺 valid_sign_hit/valid_big_f1（旧模型）
# 格式化类型: 'pct' 百分比 | 'num' 小数3位 | 'rmse' 1位 | 'int'
MODEL_METRIC_FIELDS = [
    # 分类驱动
    ('acc',             '5类准确率',      'pct'),
    ('sign_hit',        '方向命中',       'pct'),
    ('big_f1',          '大偏差F1',       'pct'),
    ('big_recall',      '大偏差召回',     'pct'),
    ('big_precision',   '大偏差精确',     'pct'),
    ('trigger_rate',    '触发率',         'pct'),
    # 数值驱动
    ('tier_acc',        '数值分级命中',   'pct'),
    ('num_recall',      '数值触发召回',   'pct'),
    ('num_precision',   '数值触发精确',   'pct'),
    ('small_f1',        '数值小偏差F1',   'pct'),
    ('big_f1_num',      '数值大偏差F1',   'pct'),
    ('trig_hit50',      '触发中|值|≥τ',   'pct'),
    # 误差
    ('rmse',            '全量RMSE',       'rmse'),
    ('rmse_cond',       '条件RMSE(>τ)',   'rmse'),
]
MODEL_METRIC_GROUPS = [
    ('分类驱动', MODEL_METRIC_FIELDS[:6]),
    ('数值驱动', MODEL_METRIC_FIELDS[6:12]),
    ('误差',     MODEL_METRIC_FIELDS[12:]),
]

# 一键流程：导入 → 重建因子 → 重训 active_model(v9)+v9.1 → 推理(v9.1) → 核对 → 归档
# 每项 = (脚本, [参数]); 仅脚本名则 () 参数。
PIPELINE = [
    ('code/common/1_ingest_xlsx.py', []),
    ('code/common/2A_rebuild.py', ['--factors']),
    ('code/common/2A_rebuild.py', ['--model']),          # 重训 active_model（默认 v9）
    ('code/common/2A_rebuild.py', ['--model', '--model-key', 'v9.1']),  # 重训 v9.1(ATR)
    ('code/common/2B_inference.py', ['--model', 'v9.1']),
    ('code/common/2C_verify.py', []),
]
INFER_ONLY = [
    ('code/common/2B_inference.py', ['--model', 'v9.1']),
    ('code/common/2C_verify.py', []),
]
VERIFY_ONLY = [('code/common/2C_verify.py', [])]


def _get(cfg, path, default=None):
    node = cfg
    for k in path.split('.'):
        if isinstance(node, dict) and k in node:
            node = node[k]
        else:
            return default
    return node


def _set(cfg, path, value):
    node = cfg
    parts = path.split('.')
    for k in parts[:-1]:
        node = node.setdefault(k, {})
    node[parts[-1]] = value


def load_config():
    with open(CONFIG_FILE, encoding='utf-8') as f:
        return json.load(f)


def save_config(cfg):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def build_defaults():
    """从 0_config 的默认值生成默认配置（恢复默认用）"""
    from _cfg import cfg
    return {
        'project_dir': cfg.PROJECT_DIR,
        'data_dir': cfg.DATA_DIR,
        'model_dir': cfg.MODEL_DIR,
        'upload_dir': cfg.UPLOAD_DIR,
        'active_model': cfg.ACTIVE_MODEL,
        'model_registry': cfg.MODEL_REGISTRY,
        'spread': {'threshold': cfg.SPREAD_THRESHOLD,
                   'threshold_big': cfg.SPREAD_THRESHOLD_BIG},
        'predict_window': {'back_days': cfg.PREDICT_BACK_DAYS,
                           'fwd_days': cfg.PREDICT_FWD_DAYS},
        'factor_rebuild': {'mode': cfg.FACTOR_REBUILD_MODE,
                           'tail_days': cfg.TAIL_DAYS,
                           'rebuild_sp_wow': cfg.REBUILD_SP_WOW},
        'training': {'valid_days': cfg.TRAIN_VALID_DAYS,
                     'n_estimators': cfg.MODEL_N_ESTIMATORS,
                     'early_stopping': cfg.MODEL_EARLY_STOPPING,
                     'random_state': cfg.MODEL_RANDOM_STATE,
                     'fixed_params': cfg.MODEL_FIXED_PARAMS},
    }


# ────────────────────────────────────────────────────────────
# 校验（与网页版一致）
# ────────────────────────────────────────────────────────────
def validate_field(path, raw, rule):
    """校验单个字段，返回 (parsed_value, error_or_None)"""
    raw = raw.strip() if isinstance(raw, str) else raw
    try:
        if rule in ('bool',):
            return raw in (True, 'true', 'on', '1'), None
        if rule.startswith('enum:'):
            opts = rule.split(':', 1)[1].split(',')
            if raw not in opts:
                raise ValueError(f"必须是 {' / '.join(opts)} 之一")
            return raw, None
        if rule == 'text':
            return raw, None
        if rule.startswith('float'):
            v = float(raw)
            if '0.001-0.5' in rule and not (0.001 <= v <= 0.5):
                raise ValueError('范围 0.001 ~ 0.5')
            if '0.1-1.0' in rule and not (0.1 <= v <= 1.0):
                raise ValueError('范围 0.1 ~ 1.0')
            if rule.startswith('float>0') and v <= 0:
                raise ValueError('必须 > 0')
            return v, None
        if rule.startswith('int'):
            v = int(float(raw))
            if '>=' in rule:
                lo = int(rule.split('>=')[1].split()[0])
                if v < lo:
                    raise ValueError(f'必须 ≥ {lo}')
            if '3-15' in rule and not (3 <= v <= 15):
                raise ValueError('范围 3 ~ 15')
            if '1-64' in rule and not (1 <= v <= 64):
                raise ValueError('范围 1 ~ 64')
            if '>0' in rule and v <= 0:
                raise ValueError('必须 > 0')
            return v, None
    except (ValueError, TypeError) as e:
        return None, f"字段格式错误: {e}"
    return raw, None


def validate_all(payload):
    """校验整份提交，返回 (cleaned_cfg, errors: {path: msg})"""
    errors = {}
    PATH_FIELDS = ('data_dir', 'project_dir', 'model_dir', 'upload_dir')
    cfg_now = load_config()
    for path, label, ftype, rule, desc in FIELDS:
        if path in PATH_FIELDS:
            continue   # 高级字段单独校验
        if path in payload:
            val, err = validate_field(path, payload[path], rule)
            if err:
                errors[path] = err
            else:
                _set(cfg_now, path, val)
    # 高级字段（路径）只允许已存在目录或非空
    for path in PATH_FIELDS:
        if path in payload and str(payload[path]).strip():
            _set(cfg_now, path, str(payload[path]).strip())
    # 交叉校验
    if not errors:
        t1 = cfg_now['spread']['threshold']
        t2 = cfg_now['spread']['threshold_big']
        if t2 <= t1:
            errors['spread.threshold_big'] = '大偏差阈值必须大于小偏差阈值'
    return cfg_now, errors


def take_status(pd):
    """只读状态信息（在工作线程中调用）。返回 (st, pd_ok)"""
    st = {'latest_disclosure': None, 'latest_actual': None,
          'latest_model': None, 'latest_output': None}
    if pd is None:
        return st, False
    from _cfg import cfg, latest_disclosure_date, latest_actual_date
    # 最新披露日（取宽表，秒级；不再扫矩阵 sorted()[:30]）
    st['latest_disclosure'] = latest_disclosure_date()
    # 最新实际结果日
    st['latest_actual'] = latest_actual_date()
    # 最新模型
    try:
        if os.path.exists(cfg.LATEST_MODEL_FILE):
            ptr = json.load(open(cfg.LATEST_MODEL_FILE))
            st['latest_model'] = ptr.get(cfg.ACTIVE_MODEL) or '（无指针）'
    except Exception:
        pass
    # 最近输出（按预测窗口排序取最新）
    try:
        outs = [f for f in os.listdir(cfg.OUTPUT_DIR) if f.endswith('.csv')]
        if outs:
            st['latest_output'] = _sort_output_files(outs)[0]
    except Exception:
        pass
    return st, True


def resolve_python():
    """优先用上一级目录的 venv，否则系统 python。返回可执行路径/命令。"""
    if sys.platform == 'win32':
        cand = os.path.join(_PARENT, 'venv', 'Scripts', 'python.exe')
    else:
        cand = os.path.join(_PARENT, 'venv', 'bin', 'python')
    if os.path.isfile(cand):
        return cand
    return sys.executable


def _sort_output_files(files):
    """按预测窗口排序，返回「最新优先」列表（取 [0] 即最新结果）。
    规则：窗口结束日最新者在前；结束日相同时，跨度大（开始日更早）者在前。
    文件名形如 预测_2026-07-22_2026-07-27.csv。解析失败按字符串兜底。"""
    from datetime import date
    def key(f):
        try:
            name = f[:-4] if f.endswith('.csv') else f
            dates = [p for p in name.split('_') if len(p) == 10 and p[4] == '-' and p[7] == '-']
            if len(dates) >= 2:
                e = date.fromisoformat(dates[-1]).toordinal()
                s = date.fromisoformat(dates[0]).toordinal()
                return (-e, s)   # 升序：结束日新者在前；同结束日跨度大者在前
            if len(dates) == 1:
                return (-date.fromisoformat(dates[0]).toordinal(), 0)
        except Exception:
            pass
        return (1e15, 0)
    return sorted(files, key=key)


def cfg_output_dir():
    """推理结果输出目录（从 _cfg 读取，失败回退到 deploy/output）"""
    try:
        from _cfg import cfg as _c
        return _c.OUTPUT_DIR
    except Exception:
        return os.path.join(_HERE, 'output')


# ────────────────────────────────────────────────────────────
# 桌面 App
# ────────────────────────────────────────────────────────────
class ConfigApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('电力现货价差预测 · 配置管理')
        self.geometry('1040x820')
        self.minsize(900, 680)
        self._set_app_icon()   # 窗口/任务栏图标（assets/SanYonon.*）

        self.vars = {}      # path -> (StringVar | BooleanVar)
        self.errs = {}      # path -> error StringVar
        self.status_vars = {}   # card key -> StringVar
        self.dirty = False
        self.running = False
        self._busy = False
        try:
            self._cfg = load_config()
        except Exception:
            self._cfg = {}

        # 根布局：grid，明确控制每行高度（日志行固定，notebook 行伸缩）
        self.grid_rowconfigure(2, weight=1)   # notebook 行可伸缩
        self.grid_rowconfigure(3, weight=0)   # 日志行固定
        self.grid_columnconfigure(0, weight=1)

        self._build_menu()         # 菜单栏（系统菜单）
        self._build_toolbar()      # row 0
        self._build_status_cards() # row 1
        self._build_form()         # row 2 (notebook)
        self._build_log()          # row 3
        self._build_footer()       # row 4

        self.protocol('WM_DELETE_WINDOW', self.on_close)
        self.bind_all('<Control-s>', lambda e: self.on_save())
        self.load_into_form()
        # 分步向导状态：步骤 → (notebook tab 索引, 操作)
        #   1=模型与路径  2=模型参数  3=保存(在模型参数页)  4=运行  5=预测结果
        self.steps = [
            {'name': '模型与路径', 'tab': 0},
            {'name': '模型参数',   'tab': 1},
            {'name': '保存配置',   'tab': 1},
            {'name': '运行流程',   'tab': 1},
            {'name': '预测结果',   'tab': 2},
        ]
        self.cur_step = 0
        self._update_step_nav()
        self.after(200, self.refresh_status_async)
        self.after(300, self.load_results)

    # ── 应用图标（assets/SanYonon.*）─────────────────────────
    def _set_app_icon(self):
        """设置窗口/任务栏/Dock 图标（跨平台兜底，失败静默不影响启动）。

        优先级：Windows → .ico (iconbitmap)；macOS → .icns/.png (iconphoto)；
        Linux → .png (iconphoto)。图标缺失/损坏时静默跳过。"""
        import os as _os
        asset_dir = _os.path.join(_HERE, 'assets')
        try:
            if sys.platform == 'win32':
                ico = _os.path.join(asset_dir, 'SanYonon.ico')
                if _os.path.isfile(ico):
                    self.iconbitmap(ico)
            else:
                # macOS / Linux：iconphoto 吃 PNG/icns 里的位图；取 assets 里任一可用
                for cand in ('SanYonon.png', 'SanYonon.ico'):
                    p = _os.path.join(asset_dir, cand)
                    if not _os.path.isfile(p):
                        continue
                    try:
                        img = tk.PhotoImage(file=p)
                        self.iconphoto(True, img)
                        self._app_icon_img = img   # 防 GC
                        break
                    except Exception:
                        continue
        except Exception:
            pass   # 图标加载失败不影响功能

    # ── UI 构建 ────────────────────────────────────────────
    def _build_menu(self):
        m = tk.Menu(self)
        fm = tk.Menu(m, tearoff=0)
        fm.add_command(label='保存配置  (Ctrl+S)', command=self.on_save)
        fm.add_command(label='恢复默认', command=self.on_reset)
        fm.add_separator()
        fm.add_command(label='退出', command=self.on_close)
        m.add_cascade(label='文件', menu=fm)

        dm = tk.Menu(m, tearoff=0)
        dm.add_command(label='打开数据目录…', command=lambda: self.on_pick_dir('data_dir'))
        dm.add_command(label='打开模型目录…', command=lambda: self.on_pick_dir('model_dir'))
        dm.add_command(label='打开上传目录…', command=lambda: self.on_pick_dir('upload_dir'))
        dm.add_command(label='打开项目目录…', command=lambda: self.on_pick_dir('project_dir'))
        m.add_cascade(label='目录', menu=dm)

        rm = tk.Menu(m, tearoff=0)
        rm.add_command(label='一键运行每日流程', command=self.on_run)
        rm.add_command(label='仅预测（2B+2C）', command=self.on_run_infer_only)
        rm.add_command(label='仅核对（2C）', command=self.on_run_verify_only)
        m.add_cascade(label='运行', menu=rm)

        hm = tk.Menu(m, tearoff=0)
        hm.add_command(label='关于', command=self._on_about)
        m.add_cascade(label='帮助', menu=hm)
        self.config(menu=m)

    def _build_toolbar(self):
        bar = ttk.Frame(self, padding=(10, 8))
        bar.grid(row=0, column=0, sticky='ew')
        self.run_btn = ttk.Button(bar, text='一键运行每日流程',
                                  command=self.on_run)
        self.run_btn.pack(side='left')
        self.infer_btn = ttk.Button(bar, text='仅预测（2B）',
                                    command=self.on_run_infer_only)
        self.infer_btn.pack(side='left', padx=(8, 0))
        self.verify_btn = ttk.Button(bar, text='仅核对（2C）',
                                     command=self.on_run_verify_only)
        self.verify_btn.pack(side='left', padx=(8, 0))
        self.progress = ttk.Progressbar(bar, mode='indeterminate', length=160)
        self.progress.pack(side='left', padx=10)
        self.run_status = ttk.Label(bar, text='')
        self.run_status.pack(side='left', padx=5)

    def _toggle_log(self):
        """显示 / 隐藏运行日志区。隐藏后空间让给上方 notebook。"""
        if not hasattr(self, 'log_frame'):
            return
        if self.log_visible.get():
            self.log_frame.grid()
        else:
            self.log_frame.grid_remove()

    def _build_status_cards(self):
        frame = ttk.Frame(self, padding=(10, 0))
        frame.grid(row=1, column=0, sticky='ew')
        keys = [('latest_disclosure', '最新披露日'),
                ('latest_actual', '最新实际结果日'),
                ('latest_model', '最新模型'),
                ('latest_output', '最近预测输出')]
        for i, (key, label) in enumerate(keys):
            card = ttk.LabelFrame(frame, text=label, padding=(10, 6))
            card.grid(row=0, column=i, padx=4, sticky='nsew')
            var = tk.StringVar(value='—')
            self.status_vars[key] = var
            ttk.Label(card, textvariable=var, font=('', 11, 'bold'),
                      wraplength=200).pack(fill='x')
            frame.columnconfigure(i, weight=1)

    def _build_form(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=2, column=0, sticky='nsew', padx=10, pady=8)

        self.widgets = {}
        self.errs = {}
        self.desc_labels = {}   # path -> 说明 Label（BUG-4 锁定提示用）

        # ── Tab 1：模型与路径 ──
        page1 = ttk.Frame(self.notebook, padding=(14, 10))
        self.notebook.add(page1, text='模型与路径')
        page1.columnconfigure(1, weight=1)
        f1 = [(p, l, t, ru, d) for p, l, t, ru, d in FIELDS
              if p == 'active_model' or p in ('data_dir', 'project_dir',
                                              'model_dir', 'upload_dir')]
        for i, (path, label, ftype, rule, desc) in enumerate(f1):
            r = i * 2
            ttk.Label(page1, text=label, font=('', 10, 'bold')).grid(
                row=r, column=0, sticky='w', pady=(6, 2))
            if ftype == 'select':   # active_model
                options = list(self._cfg.get('model_registry', {}).keys())
                var = tk.StringVar()
                cb = ttk.Combobox(page1, textvariable=var, values=options,
                                  state='readonly', width=34)
                cb.grid(row=r, column=1, sticky='w', pady=(6, 2))
                self.widgets[path] = cb
                self.vars[path] = var
                if path == 'active_model':
                    # BUG-4：切换模型后立即刷新无效字段的锁定状态
                    var.trace_add('write', lambda *a: self._apply_model_field_locks())
                else:
                    var.trace_add('write', lambda *a, p=path: self._mark_dirty(p))
                ttk.Label(page1, text=desc, foreground='#888',
                          font=('', 8)).grid(row=r, column=2, sticky='w',
                                             padx=(8, 0))
            else:   # 路径字段：Entry + 浏览按钮 + 说明
                var = tk.StringVar()
                frm = ttk.Frame(page1)
                frm.grid(row=r, column=1, sticky='w', pady=(6, 2))
                ent = ttk.Entry(frm, textvariable=var, width=34)
                ent.pack(side='left')
                ttk.Button(frm, text='浏览…', width=7,
                           command=lambda p=path: self.on_pick_dir(p)).pack(
                    side='left', padx=(6, 0))
                ttk.Button(frm, text='打开', width=5,
                           command=lambda p=path: self._open_path_dir(p)).pack(
                    side='left', padx=(2, 0))
                # 项目目录旁加「自动识别」按钮：读取标的文件自动填入各目录
                if path == 'project_dir':
                    ttk.Button(frm, text='自动识别', width=9,
                               command=self.on_auto_detect).pack(side='left', padx=(6, 0))
                self.widgets[path] = ent
                self.vars[path] = var
                var.trace_add('write', lambda *a, p=path: self._mark_dirty(p))
                ttk.Label(page1, text=desc, foreground='#888',
                          font=('', 8)).grid(row=r + 1, column=1, sticky='w')
            err = tk.StringVar()
            self.errs[path] = err
            ttk.Label(page1, textvariable=err, foreground='#c62828',
                      font=('', 8)).grid(row=r + 1, column=2, sticky='w',
                                         padx=(8, 0))

        # ── Tab 2：模型参数（分组显示，可上下滚动）──
        page2 = ttk.Frame(self.notebook, padding=(14, 10))
        self.notebook.add(page2, text='模型参数')
        # 滚动容器：canvas + 垂直滚动条，内容超出高度时可滚动
        page2_canvas = tk.Canvas(page2, highlightthickness=0)
        page2_vsb = ttk.Scrollbar(page2, orient='vertical', command=page2_canvas.yview)
        page2_canvas.configure(yscrollcommand=page2_vsb.set)
        page2_vsb.pack(side='right', fill='y')
        page2_canvas.pack(side='left', fill='both', expand=True)
        page2_inner = ttk.Frame(page2_canvas)
        page2_inner_id = page2_canvas.create_window((0, 0), window=page2_inner, anchor='nw')
        page2_inner.bind('<Configure>', lambda e, c=page2_canvas: c.configure(
            scrollregion=c.bbox('all')))
        page2_canvas.bind('<Configure>', lambda e, c=page2_canvas, iw=page2_inner_id: c.itemconfigure(
            iw, width=e.width))
        # 鼠标滚轮滚动（Windows / macOS / Linux）
        def _on_mousewheel(event, c=page2_canvas):
            delta = -1 if getattr(event, 'delta', 0) > 0 else 1
            if sys.platform == 'darwin':
                c.yview_scroll(-1 * int(event.delta), 'units')
            else:
                c.yview_scroll(delta, 'units')
        page2_canvas.bind('<Enter>', lambda e, c=page2_canvas: c.bind_all(
            '<MouseWheel>', _on_mousewheel))
        page2_canvas.bind('<Leave>', lambda e: page2_canvas.unbind_all('<MouseWheel>'))
        page2 = page2_inner
        page2.columnconfigure(3, weight=1)
        f2 = [f for f in FIELDS if f not in f1]
        rows = {}   # path -> (page, grid_row, grid_col)
        grid_row = 0
        for sec_name, pred in PARAM_SECTIONS:
            sec = [f for f in f2 if pred(f[0])]
            if not sec:
                continue
            ttk.Label(page2, text='—— ' + sec_name + ' ——', font=('', 11, 'bold'),
                      foreground='#1a5fb4').grid(
                row=grid_row, column=0, columnspan=4, sticky='w',
                pady=(12, 4))
            grid_row += 1
            half = (len(sec) + 1) // 2
            for i, (path, label, ftype, rule, desc) in enumerate(sec):
                col = 0 if i < half else 2
                sub = i % half
                r = grid_row + sub * 2
                ttk.Label(page2, text=label, font=('', 9, 'bold')).grid(
                    row=r, column=col, sticky='w', pady=(4, 0))
                if ftype == 'bool':
                    var = tk.BooleanVar(value=False)
                    cb = ttk.Checkbutton(page2, text=desc, variable=var)
                    cb.grid(row=r + 1, column=col, columnspan=2, sticky='w')
                    self.widgets[path] = cb
                    self.vars[path] = var
                    var.trace_add('write', lambda *a, p=path: self._mark_dirty(p))
                else:
                    var = tk.StringVar()
                    if ftype == 'select':
                        options = rule.split(':', 1)[1].split(',')
                        w = ttk.Combobox(page2, textvariable=var, values=options,
                                         state='readonly', width=20)
                    else:
                        w = ttk.Entry(page2, textvariable=var, width=20)
                    w.grid(row=r + 1, column=col, sticky='w', pady=(2, 2))
                    self.widgets[path] = w
                    self.vars[path] = var
                    var.trace_add('write', lambda *a, p=path: self._mark_dirty(p))
                # 说明（放控件下方，小字）
                if ftype != 'bool' and desc:
                    dl = ttk.Label(page2, text=desc, foreground='#888',
                                   font=('', 8))
                    dl.grid(row=r + 1, column=col + 1, sticky='w',
                            pady=(2, 2), padx=(6, 0))
                    self.desc_labels[path] = dl
                err = tk.StringVar()
                self.errs[path] = err
                ttk.Label(page2, textvariable=err, foreground='#c62828',
                          font=('', 8)).grid(
                    row=r + 1, column=col + 1, sticky='e', pady=(2, 2))
            grid_row += half * 2

        # ── 模型参数页底部：醒目「保存配置」大按钮（对应分步流程第 3 步）──
        grid_row += 1
        save_row = ttk.Frame(page2, padding=(0, 10))
        save_row.grid(row=grid_row, column=0, columnspan=4, sticky='ew')
        ttk.Label(save_row, text='完成参数设置后，点下方按钮保存并继续：',
                  foreground='#666', font=('', 9)).pack(side='left', padx=(0, 10))
        self.save_big_btn = ttk.Button(
            save_row, text='保存配置并继续',
            command=lambda: self._go_next_from_save())
        self.save_big_btn.pack(side='left')

        # ── 预测结果页（放在配置 tab 之后）──
        self._build_results_tab()

    # ── BUG-4：按当前模型锁定无效配置字段 ─────────────────────
    def _apply_model_field_locks(self):
        """当前 active_model 训练时忽略的字段 → 置灰输入框 + 橙色提示。

        切换 active_model 时自动调用（trace）；启动时由 load_into_form 末尾调用。
        锁定是展示层保护：即使字段被禁用，保存逻辑不变；被忽略字段改值不写入生效，
        避免用户误以为改了 training.valid_days 等会对当前模型生效。"""
        model = str(self.vars.get('active_model', tk.StringVar()).get())
        ignored = (MODEL_IGNORED_FIELDS.get(model)
                   or MODEL_IGNORED_FIELDS.get('*', {}))
        for path, w in self.widgets.items():
            is_locked = path in ignored
            state = 'disabled' if is_locked else 'normal'
            try:
                w.configure(state=state)
            except Exception:
                pass
            # 说明文字：锁定时替换为提示，解锁后还原为原始 desc
            dl = self.desc_labels.get(path)
            if dl is not None:
                tip = ignored.get(path)
                if tip:
                    dl.config(text='[已锁定] ' + tip, foreground='#b8860b')
                else:
                    orig = next((d for p, l, t, ru, d in FIELDS if p == path), '')
                    dl.config(text=orig, foreground='#888')

    def _build_results_tab(self):
        page = ttk.Frame(self.notebook, padding=(10, 8))
        self.notebook.add(page, text='预测结果')
        page.columnconfigure(0, weight=1)
        page.rowconfigure(4, weight=1)   # 明细表行伸缩

        # 顶部：文件选择 + 操作按钮
        top = ttk.Frame(page)
        top.grid(row=0, column=0, sticky='ew', pady=(0, 6))
        ttk.Label(top, text='结果文件:').pack(side='left')
        self.res_file = ttk.Combobox(top, state='readonly', width=42)
        self.res_file.pack(side='left', padx=(4, 8))
        self.res_file.bind('<<ComboboxSelected>>', lambda e: self._load_selected_csv())
        ttk.Button(top, text='刷新', command=self.load_results).pack(side='left', padx=2)
        ttk.Button(top, text='打开输出目录',
                   command=lambda: self._open_dir()).pack(side='left', padx=2)
        ttk.Button(top, text='打开前端仪表盘',
                   command=self.open_results_dashboard).pack(side='left', padx=2)
        # 显示模型指标开关（默认开；关闭后指标卡隐藏，明细表自动变高）
        self.show_metrics = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text='显示模型指标', variable=self.show_metrics,
                        command=self._toggle_metrics).pack(side='left', padx=(14, 0))

        # 摘要
        self.summary_var = tk.StringVar(value='—')
        ttk.Label(page, textvariable=self.summary_var, foreground='#333',
                  font=('', 10)).grid(row=1, column=0, sticky='ew', pady=(0, 6))

        # 模型指标卡片（Valid 集，grid 到 page 顶部，可用开关隐藏）
        self.metrics_card_frame = ttk.Frame(page)
        self.metrics_card_frame.grid(row=2, column=0, sticky='ew', pady=(0, 6))
        self.metrics_card_frame.columnconfigure(0, weight=1)
        self._build_model_metrics_card(self.metrics_card_frame)

        # 明细表（列随 CSV 动态重建，见 _rebuild_tree）
        self.tree_wrap = ttk.Frame(page)
        self.tree_wrap.grid(row=4, column=0, sticky='nsew', padx=4, pady=4)
        self.res_tree = ttk.Treeview(self.tree_wrap, columns=('日期', '小时'),
                                     show='headings', height=12)
        self.res_tree.tag_configure('red', foreground='#c62828')
        self.res_tree.tag_configure('yellow', foreground='#b8860b')
        self.res_tree.tag_configure('green', foreground='#2e7d32')
        vsb = ttk.Scrollbar(self.tree_wrap, command=self.res_tree.yview)
        hsb = ttk.Scrollbar(self.tree_wrap, orient='horizontal',
                            command=self.res_tree.xview)
        self.res_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.res_tree.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')
        hsb.pack(side='bottom', fill='x')
        self.res_tree.bind('<<TreeviewSelect>>', self._on_result_row_click)

        self.res_rows = {}   # iid -> row dict（存原始行数据）

    def _toggle_metrics(self):
        """显示 / 隐藏模型指标卡片。隐藏后明细表自动长高占满。"""
        if not hasattr(self, 'metrics_card_frame'):
            return
        if self.show_metrics.get():
            self.metrics_card_frame.grid()
        else:
            self.metrics_card_frame.grid_remove()

    # ── 模型指标卡片（Valid 集）────────────────────────────
    def _build_model_metrics_card(self, parent):
        """在预测结果页构建「模型指标（Valid 集）」卡片。

        读取活动模型的 metrics['valid']（新模型）或平铺 valid_sign_hit/valid_big_f1（旧模型）。
        值在后台线程加载（joblib.load 8.5MB 较慢），主线程只更新 StringVar。"""
        self.metrics_vars = {}   # metric key -> StringVar
        card = ttk.LabelFrame(parent, text='模型指标（Valid 集）', padding=(10, 6))
        card.pack(fill='x')
        self.metrics_card = card

        # 头部：模型名 + 训练时间 + 阈值
        self.metric_meta = tk.StringVar(value='—')
        ttk.Label(card, textvariable=self.metric_meta, foreground='#555',
                  font=('', 9)).grid(row=0, column=0, columnspan=8, sticky='w', pady=(0, 4))

        # 分组网格：分类驱动 / 数值驱动 / 误差
        row0 = 1
        for gi, (gname, fields) in enumerate(MODEL_METRIC_GROUPS):
            ttk.Label(card, text='—— ' + gname + ' ——', foreground='#1a5fb4',
                      font=('', 8, 'bold')).grid(row=row0 + gi * 2, column=0,
                                                 columnspan=8, sticky='w', pady=(4, 0))
            for fi, (key, label, fmt) in enumerate(fields):
                r = row0 + gi * 2 + 1
                c = fi * 2
                ttk.Label(card, text=label, foreground='#888',
                          font=('', 8)).grid(row=r, column=c, sticky='w', padx=(2, 0))
                var = tk.StringVar(value='—')
                self.metrics_vars[key] = var
                ttk.Label(card, textvariable=var, font=('', 10, 'bold'),
                          foreground='#333').grid(row=r + 1, column=c, sticky='w', padx=(2, 0))

    @staticmethod
    def _fmt_metric_val(v, fmt):
        """指标值格式化：None/NaN → '—'。fmt: 'pct'/'num'/'rmse'/'int'"""
        if v is None:
            return '—'
        try:
            fv = float(v)
            if fv != fv:   # NaN
                return '—'
        except (TypeError, ValueError):
            return '—'
        if fmt == 'pct':
            return f'{fv * 100:.1f}%'
        if fmt == 'rmse':
            return f'{fv:.1f}'
        if fmt == 'int':
            return f'{int(fv)}'
        return f'{fv:.3f}'

    def _load_model_metrics_async(self):
        """后台线程加载活动模型指标（不阻塞 UI）。"""
        threading.Thread(target=self._metrics_worker, daemon=True).start()

    def _metrics_worker(self):
        info = {'meta': None, 'values': {}, 'note': None, 'kind': 'classic'}
        try:
            from _cfg import cfg, active_model_path
            path = active_model_path()
            if not path:
                info['note'] = '未找到活动模型'
                self.after(0, lambda: self._apply_metrics(info))
                return
            import joblib
            m = joblib.load(path)
            label = (cfg.MODEL_REGISTRY.get(cfg.ACTIVE_MODEL) or {}).get('label', cfg.ACTIVE_MODEL)
            ts = m.get('trained_at', '—')
            t1 = m.get('threshold_minor', cfg.SPREAD_THRESHOLD)
            t2 = m.get('threshold_big', cfg.SPREAD_THRESHOLD_BIG)

            if m.get('model_type') == 'v9_direction':
                # v9 指标结构：metrics.valid = {acted, base, dir_head} + 顶层平铺
                # v9.1 结构：metrics.valid = {c_base, c_atr} + 顶层 valid_c_atr_*
                #   （BUG-1 修复：v9.1 字段名不同，按 version 兼容取数）
                is_v91 = m.get('version') == 'v9.1' or 'valid_c_atr_hit' in m
                info['kind'] = 'v9'
                valid = (m.get('metrics') or {}).get('valid') or {}
                if is_v91:
                    acted = valid.get('c_atr') or {}
                    base = valid.get('c_base') or {}
                    dh = {}
                    c_hit = m.get('valid_c_atr_hit')
                    c_trig = acted.get('trigger_rate', base.get('trig'))
                    dhit = acted.get('dir_hit')
                else:
                    acted = valid.get('acted') or {}
                    base = valid.get('base') or {}
                    dh = valid.get('dir_head') or {}
                    c_hit = m.get('valid_c_hit')
                    c_trig = acted.get('trig_global', base.get('trig'))
                    dhit = m.get('valid_dir_hit', dh.get('dir_hit'))
                fmt = self._fmt_metric_val
                info['values'] = {
                    'c_hit': fmt(c_hit, 'pct'),
                    'trig': fmt(c_trig, 'pct'),
                    'trig_ge50': fmt(acted.get('trig_ge50'), 'pct'),
                    'net_win': fmt(acted.get('net_win_rate'), 'pct'),
                    'n': fmt(acted.get('n'), 'int'),
                    'pnl': fmt(acted.get('pnl_total'), 'num'),
                    'dd': fmt(acted.get('max_drawdown'), 'num'),
                    'dir_hit': fmt(dhit, 'pct'),
                    'dir_trig': fmt(dh.get('trigger_rate'), 'pct'),
                    'mean_conf': fmt(dh.get('mean_conf'), 'num'),
                }
                info['meta'] = f'{label} · 训练于 {ts} · τ={t1}/{t2} · 规则层C(出手)'
            else:
                vm = (m.get('metrics') or {}).get('valid') or m.get('valid_metrics')
                if vm:
                    for key, _label, _fmt in MODEL_METRIC_FIELDS:
                        if key in vm:
                            info['values'][key] = self._fmt_metric_val(vm[key], _fmt)
                elif 'valid_sign_hit' in m:
                    # 旧模型降级：只展示平铺 2 项
                    info['values'] = {
                        'sign_hit': self._fmt_metric_val(m.get('valid_sign_hit'), 'pct'),
                        'big_f1': self._fmt_metric_val(m.get('valid_big_f1'), 'pct'),
                    }
                    info['note'] = '旧模型仅存 2 项指标（请重跑 2A_rebuild 后自动刷新）'
                info['meta'] = f'{label} · 训练于 {ts} · τ={t1}/{t2}'
        except Exception as e:
            info['note'] = f'模型指标加载失败: {e}'
        self.after(0, lambda: self._apply_metrics(info))

    def _apply_metrics(self, info):
        """回主线程更新指标卡片。v9 布局与 classic 不同，按 kind 分派。"""
        if info.get('kind') == 'v9':
            self._apply_metrics_v9(info)
            return
        if hasattr(self, 'metrics_vars'):
            for key, var in self.metrics_vars.items():
                var.set(info['values'].get(key, '—'))
        if hasattr(self, 'metric_meta'):
            meta = info['meta'] or ''
            note = info.get('note')
            self.metric_meta.set((meta + '  ' if meta else '') + (note or ''))

    def _apply_metrics_v9(self, info):
        """v9 指标卡片：重建为 v9 专用网格（规则层C / 方向头）。主线程调用。"""
        card = getattr(self, 'metrics_card', None)
        if card is None:
            return
        # 清掉 classic 网格（保留头部 meta 行与 LabelFrame 本身）
        for w in card.winfo_children():
            w.destroy()
        # BUG-7 修复：meta 同时写入 self.metric_meta StringVar，
        # 避免重建后该变量停留在旧值 '—'（原本只用 text= 硬编码在 Label）
        if info['meta']:
            ttk.Label(card, text=info['meta'], foreground='#555',
                      font=('', 9)).grid(row=0, column=0, columnspan=8,
                                         sticky='w', pady=(0, 4))
        if hasattr(self, 'metric_meta'):
            self.metric_meta.set(info['meta'] or '—')
        groups = [
            ('规则层 C（出手）', [('c_hit', '方向命中', 'pct'),
                                ('trig', '全局触发率', 'pct'),
                                ('trig_ge50', '触发中|实际|≥τ', 'pct'),
                                ('net_win', '净胜率', 'pct'),
                                ('n', '样本数', 'int'),
                                ('pnl', 'P&L', 'num'),
                                ('dd', '最大回撤', 'num')]),
            ('方向头（模型信号）', [('dir_hit', '方向命中', 'pct'),
                                  ('dir_trig', '触发率', 'pct'),
                                  ('mean_conf', '平均置信度', 'num')]),
        ]
        self.metrics_vars = {}
        row = 1
        for gname, fields in groups:
            ttk.Label(card, text='—— ' + gname + ' ——', foreground='#1a5fb4',
                      font=('', 8, 'bold')).grid(row=row, column=0,
                                                 columnspan=8, sticky='w',
                                                 pady=(4, 0))
            row += 1
            for fi, (key, label, fmt) in enumerate(fields):
                c = (fi % 4) * 2
                r = row + (fi // 4) * 2
                ttk.Label(card, text=label, foreground='#888',
                          font=('', 8)).grid(row=r, column=c, sticky='w',
                                             padx=(2, 0))
                var = tk.StringVar(value='—')
                self.metrics_vars[key] = var
                ttk.Label(card, textvariable=var, font=('', 10, 'bold'),
                          foreground='#333').grid(row=r + 1, column=c,
                                                  sticky='w', padx=(2, 0))
            row += ((len(fields) + 3) // 4) * 2
        for key, var in self.metrics_vars.items():
            var.set(info['values'].get(key, '—'))

    def _open_dir(self):
        import subprocess as _sp
        out_dir = cfg_output_dir()
        try:
            if sys.platform == 'win32':
                _sp.Popen(['explorer', out_dir])
            elif sys.platform == 'darwin':
                _sp.Popen(['open', out_dir])
            else:
                _sp.Popen(['xdg-open', out_dir])
        except Exception as e:
            messagebox.showwarning('打开失败', str(e))

    def open_results_dashboard(self):
        """启动本地 Web 前端仪表盘（results_server.py）并在浏览器打开。

        服务后台运行（不阻塞 UI），防重复启动：已运行则直接开浏览器。
        端口默认 8301；若被占用自动提示换端口。"""
        port = 8301
        url = f'http://127.0.0.1:{port}'
        # 已在运行？直接开浏览器
        if getattr(self, '_results_server_alive', False):
            webbrowser.open(url)
            return
        # 探测端口是否已被占用（可能上次启动遗留的服务）
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        try:
            s.connect(('127.0.0.1', port))
            s.close()
            self._results_server_alive = True
            webbrowser.open(url)
            return
        except OSError:
            pass
        finally:
            s.close()
        # 启动服务（后台，随本进程退出）
        py = resolve_python()
        server = os.path.join(_HERE, 'results_server.py')
        try:
            proc = subprocess.Popen(
                [py, server, '--port', str(port)],
                cwd=_HERE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._results_server_proc = proc
            self._results_server_alive = True
        except Exception as e:
            messagebox.showerror('启动失败', f'无法启动前端服务：{e}')
            return
        # 等端口就绪再开浏览器（最多 ~2s）
        import time
        def _open_when_ready():
            deadline = time.time() + 3
            while time.time() < deadline:
                try:
                    s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s2.settimeout(0.3)
                    try:
                        s2.connect(('127.0.0.1', port))
                        s2.close()
                        self.after(0, lambda: webbrowser.open(url))
                        return
                    except OSError:
                        pass
                    finally:
                        s2.close()
                except Exception:
                    pass
                time.sleep(0.15)
            self.after(0, lambda: messagebox.showinfo(
                '前端服务', f'服务似乎未就绪，请手动打开 {url}'))
        threading.Thread(target=_open_when_ready, daemon=True).start()
        self.status_var.set('已启动前端仪表盘 → ' + url)

    def load_results(self):
        """刷新预测结果页：列出输出目录 csv，选中最新一个并加载。"""
        out_dir = cfg_output_dir()
        files = _sort_output_files(
            f for f in os.listdir(out_dir) if f.endswith('.csv'))
        self.res_file['values'] = files
        # 同步刷新模型指标卡片（后台线程，不阻塞）
        if hasattr(self, 'metrics_vars'):
            self._load_model_metrics_async()
        if not files:
            self.summary_var.set('尚无预测结果。运行「一键运行」后自动生成。')
            self.res_tree.delete(*self.res_tree.get_children())
            return
        latest = files[0]   # _sort_output_files 最新优先
        self.res_file.set(latest)
        self._load_selected_csv()

    def _load_selected_csv(self):
        fname = self.res_file.get()
        if not fname:
            return
        self._render_csv(os.path.join(cfg_output_dir(), fname))

    def _render_csv(self, path):
        try:
            import csv
            with open(path, encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                header = reader.fieldnames or []
                rows = list(reader)
        except Exception as e:
            self.summary_var.set(f'读取失败: {e}')
            return
        n = len(rows)
        if not n:
            self.summary_var.set('文件为空')
            return
        # 动态列结构判定（v9 10 列 / classic 6 列 / price 3 列）
        cols, is_v9, named = self._classify_columns(header)
        self._cols = cols
        self._is_v9 = is_v9
        self._named = named
        self._current_csv_path = path
        # 小时统一成字符串（2B 里可能是 '00:00' 或 int）
        for r in rows:
            r['小时'] = str(r.get('小时', ''))[:5]

        self._render_summary(rows, is_v9, named)
        self._rebuild_tree(path, rows, cols, is_v9, named)

    def _classify_columns(self, header):
        """判定 CSV 结构：返回 (展示列列表, is_v9, named)。

        is_v9 = 含『模型方向』『方向核对』；v9 的『方向核对』列在 2B 实际矩阵
        加载失败时会缺失，这里自动补齐（值 '—'）。"""
        cols = list(dict.fromkeys(header))   # 保序去重
        is_v9 = '模型方向' in cols
        if is_v9 and '方向核对' not in cols:
            cols.append('方向核对')
        named = {c: c for c in cols}
        return cols, is_v9, named

    def _render_summary(self, rows, is_v9, named):
        """摘要行：classic 统计预警/大偏差；v9 统计出手/方向核对对错 + 套利。"""
        import collections
        n = len(rows)
        dates = sorted({r.get('日期', '') for r in rows})
        head = f'预测窗口: {dates[0]} ~ {dates[-1]}（{len(dates)} 天 · {n} 小时）'
        # 核对标记（从当前文件名解析 a/b/c）
        flag_lbl = ''
        try:
            from _verify import CHECK_LABEL, parse_filename
            meta = parse_filename(os.path.basename(self._current_csv_path))
            if meta:
                flag_lbl = f'  |  核对: {CHECK_LABEL.get(meta["flag"], "—")}'
        except Exception:
            pass
        if is_v9:
            acted = sum(1 for r in rows if r.get('是否出手') == '是')
            ck = collections.Counter(r.get('方向核对', '') or '待实际' for r in rows)
            ok = ck.get('对·实际正', 0) + ck.get('对·实际负', 0)
            bad = ck.get('错·实际正', 0) + ck.get('错·实际负', 0)
            neu = ck.get('实际中性', 0)
            pend = ck.get('待实际', 0)
            na = ck.get('未出手', 0)
            arb = collections.Counter(r.get('套利结果', '') or '待实际' for r in rows)
            pro = sum(v for k, v in arb.items() if str(k).startswith('盈利'))
            loss = sum(v for k, v in arb.items() if str(k).startswith('亏损'))
            miss = sum(v for k, v in arb.items() if str(k).startswith('错过'))
            pnl = 0.0
            for r in rows:
                try:
                    v = float(r.get('套利盈亏(元/MWh)', ''))
                    pnl += v if not math.isnan(v) else 0.0
                except (TypeError, ValueError):
                    continue
            arb_txt = ''
            if '套利结果' in (self._cols or []):
                arb_txt = (f'  套利 买{sum(1 for r in rows if r.get("套利时机") == "日前买")}'
                           f'/卖{sum(1 for r in rows if r.get("套利时机") == "日前卖")}'
                           f'  盈{pro}/亏{loss}/错{miss}'
                           f'  盈亏{pnl:+.1f}')
            self.summary_var.set(
                f'{head}{flag_lbl}  |  出手 {acted} 小时  对 {ok} / 错 {bad}  '
                f'实际中性 {neu}  待实际 {pend}  未出手 {na}{arb_txt}')
        else:
            levels = collections.Counter(r.get('预警等级', '') for r in rows)
            warns = sum(1 for r in rows if '是' in str(r.get('是否预警', '')))
            big = sum(1 for r in rows if r.get('预警等级', '').startswith('大'))
            self.summary_var.set(
                f'{head}{flag_lbl}  |  预警 {warns}/{n} 小时  大偏差 {big} 小时  '
                f'|  {dict(levels)}')

    def _rebuild_tree(self, path, rows, cols, is_v9, named):
        """按 CSV 动态重建明细表：表头=全部列，横竖滚动条，双格式配色。"""
        # 重建 Treeview（列结构随 CSV 变化，最干净的方式是销毁重建）。
        # 必须把旧的 tree + 两个滚动条一起销毁，否则每次切换文件后滚动条堆积，
        # 挤占 tree_wrap 空间 → 表格窗口被压缩、滚动条错乱（BUG-8 修复）。
        for w in self.tree_wrap.winfo_children():
            w.destroy()
        tree = ttk.Treeview(self.tree_wrap, columns=cols, show='headings', height=8)
        vsb = ttk.Scrollbar(self.tree_wrap, command=tree.yview)
        hsb = ttk.Scrollbar(self.tree_wrap, orient='horizontal', command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.tag_configure('red', foreground='#c62828')
        tree.tag_configure('yellow', foreground='#b8860b')
        tree.tag_configure('green', foreground='#2e7d32')
        tree.tag_configure('buy', foreground='#c62828', background='#fdecea')
        tree.tag_configure('sell', foreground='#2e7d32', background='#e8f5e9')
        # 列宽：日期/小时固定，其余按内容最长值自适应，封顶 220
        for c in cols:
            tree.heading(c, text=c)
            if c == '日期':
                tree.column(c, width=100, anchor='center')
            elif c == '小时':
                tree.column(c, width=60, anchor='center')
            else:
                maxlen = max([len(str(r.get(c, ''))) for r in rows[:100]] + [len(c)])
                w = min(220, max(90, 8 * maxlen))
                tree.column(c, width=w, anchor='center')
        tree.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')
        hsb.pack(side='bottom', fill='x')
        tree.bind('<<TreeviewSelect>>', self._on_result_row_click)
        self.res_tree = tree
        self.res_vsb = vsb
        self.res_hsb = hsb

        # 填行 + 配色
        self.res_rows = {}
        for i, r in enumerate(rows):
            tag = self._row_tag(r, is_v9)
            iid = f'r{i}'
            self.res_rows[iid] = r
            tree.insert('', 'end', iid=iid, tags=(tag,) if tag else (),
                        values=[r.get(c, '') for c in cols])

    def _row_tag(self, r, is_v9):
        """行配色：v9 出手小时按套利时机（日前买红/日前卖绿），否则按方向核对；
        classic 按『预警等级』。"""
        if is_v9:
            arb = r.get('套利时机', '') or ''
            if arb == '日前买':
                return 'buy'
            if arb == '日前卖':
                return 'sell'
            ck = r.get('方向核对', '') or ''
            if ck.startswith('对'):
                return 'green'
            if ck.startswith('错'):
                return 'red'
            return ''
        lvl = r.get('预警等级', '')
        if '正常' in lvl:
            return 'green'
        if lvl.startswith('大'):
            return 'red'
        return 'yellow'

    def _on_result_row_click(self, _evt):
        sel = self.res_tree.selection()
        if not sel:
            return
        r = self.res_rows.get(sel[0])
        if not r:
            return
        # 遍历 CSV 全部列动态展示（v9 10 列 / classic 6 列 / price 3 列）
        cols = getattr(self, '_cols', None) or [c for c in r.keys()]
        lines = []
        for c in cols:
            v = r.get(c, '')
            if c == 'class_code' and str(v).strip() not in ('', '—'):
                try:
                    code = int(v)
                    v = f'{v} ({["大负偏差","负偏差","中性","正偏差","大正偏差"][code] if 0 <= code <= 4 else v})'
                except (TypeError, ValueError):
                    pass
            lines.append(f'{c}: {v if str(v).strip() else "—"}')
        messagebox.showinfo(f"预测详情 · {r.get('日期')} {r.get('小时')}",
                            '\n'.join(lines))

    def _build_log(self):
        lf = ttk.LabelFrame(self, text='运行日志（一键运行 / 仅预测输出）', padding=(8, 6))
        lf.grid(row=3, column=0, sticky='ew', padx=10, pady=(4, 8))
        self.log_frame = lf
        self.log = tk.Text(lf, height=8, state='disabled', wrap='none',
                           font=('Menlo', '10'))
        sb = ttk.Scrollbar(lf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.grid(row=0, column=0, sticky='nsew')
        sb.grid(row=0, column=1, sticky='ns')
        lf.rowconfigure(0, weight=1)
        lf.columnconfigure(0, weight=1)
        self.append_log(f'将使用 Python: {resolve_python()}')
        self.append_log('点上方「一键运行」执行 1→2A→2B→2C；「仅预测」跑 2B+2C；'
                        '「仅核对（2C）」对新实际数据重核对。输出实时显示在下方。')

    def _build_footer(self):
        foot = ttk.Frame(self, padding=(10, 6))
        foot.grid(row=4, column=0, sticky='ew')
        self.status_var = tk.StringVar(value='就绪')
        ttk.Label(foot, textvariable=self.status_var).pack(side='left')
        # 运行日志显示开关（放在最下方页脚）
        self.log_visible = tk.BooleanVar(value=True)
        ttk.Checkbutton(foot, text='显示运行日志',
                        variable=self.log_visible,
                        command=self._toggle_log).pack(side='left', padx=(14, 0))
        ttk.Button(foot, text='恢复默认', command=self.on_reset).pack(side='right')
        ttk.Button(foot, text='保存配置', command=self.on_save).pack(side='right', padx=6)

        # ── 分步向导导航（上一步 / 下一步 / 步骤指示）──
        nav = ttk.Frame(foot)
        nav.pack(side='right', padx=(20, 4))
        self.step_label = ttk.Label(nav, text='', font=('', 9, 'bold'),
                                    foreground='#1a5fb4')
        self.step_label.pack(side='left', padx=(0, 10))
        self.prev_btn = ttk.Button(nav, text='← 上一步', command=self._go_prev)
        self.prev_btn.pack(side='left')
        self.next_btn = ttk.Button(nav, text='下一步 →', command=self._go_next)
        self.next_btn.pack(side='left', padx=(6, 0))

    def _update_step_nav(self):
        """刷新底部导航：步骤指示、按钮可用性。"""
        n = len(self.steps)
        self.step_label.config(text=f'步骤 {self.cur_step + 1}/{n} · {self.steps[self.cur_step]["name"]}')
        # 上一步：第0步不可用
        self.prev_btn.config(state='disabled' if self.cur_step == 0 else 'normal')
        # 下一步：运行中禁用；最后一步(预测结果)隐藏式禁用
        if self.running:
            self.next_btn.config(state='disabled')
        else:
            self.next_btn.config(state='disabled' if self.cur_step >= n - 1 else 'normal')

    def _go_prev(self):
        if self.running:
            return
        if self.cur_step > 0:
            self.cur_step -= 1
            self.notebook.select(self.steps[self.cur_step]['tab'])
            self._update_step_nav()

    def _go_next_from_save(self):
        """模型参数页「保存配置并继续」：保存 → 跳到『运行流程』步骤。"""
        if self.running:
            return
        if not self._try_save():
            return
        self.cur_step = 3    # 运行流程（保存已完成）
        self.notebook.select(self.steps[3]['tab'])
        self._update_step_nav()

    def _go_next(self):
        """下一步：按步骤引导（模型与路径→模型参数→保存→运行→预测结果）。"""
        if self.running:
            return
        n = len(self.steps)
        if self.cur_step >= n - 1:
            return
        nxt = self.cur_step + 1
        step = self.steps[nxt]
        if step['name'] == '保存配置':
            # 先保存（静默：有错则停在模型参数页提示）
            ok = self._try_save()
            if not ok:
                self.cur_step = 1   # 停在模型参数页
                self.notebook.select(self.steps[1]['tab'])
                self._update_step_nav()
                return
            self.status_var.set('配置已保存，可运行每日流程')
        elif step['name'] == '运行流程':
            # 保存后自动切到模型参数页（隐藏提示），用户可点「一键运行」或直接下一步看结果
            self.notebook.select(self.steps[1]['tab'])
        self.cur_step = nxt
        # 进入预测结果前，若已有输出文件则刷新
        if step['name'] == '预测结果':
            self.after(50, self.load_results)
        self.notebook.select(step['tab'])
        self._update_step_nav()

    def _try_save(self):
        """保存配置（静默版，失败返回 False 并在表单显示错误）。"""
        if self.running:
            messagebox.showwarning('运行中', '每日流程正在运行，请稍后再保存配置。')
            return False
        for v in self.errs.values():
            v.set('')
        payload = self.collect_from_form()
        cleaned, errors = validate_all(payload)
        if errors:
            for path, msg in errors.items():
                if path in self.errs:
                    self.errs[path].set(msg)
            messagebox.showerror('配置有误',
                                 '\n'.join(f'{k}: {v}' for k, v in errors.items()))
            return False
        for p in ('data_dir', 'project_dir', 'model_dir', 'upload_dir'):
            v = str(cleaned.get(p, '')).strip().replace('\\', '/')
            if v and not os.path.isdir(v):
                if not messagebox.askyesno('目录不存在',
                                           f'{p} = {v}\n\n该目录当前不存在。仍要保存吗？'):
                    return False
            cleaned[p] = v
        try:
            save_config(cleaned)
            self.dirty = False
            self.status_var.set(f'已保存 {datetime.now():%H:%M:%S}')
        except Exception as e:
            messagebox.showerror('保存失败', str(e))
            return False
        self.after(100, self.refresh_status_async)
        return True

    # ── 表单加载 / 收集 ────────────────────────────────────
    def load_into_form(self):
        try:
            cfg = load_config()
        except Exception as e:
            messagebox.showerror('读取配置失败', str(e))
            return
        # 加载表单期间抑制 dirty 标记：vars.set() 会触发 trace → _mark_dirty，
        # 导致启动即显示"有未保存的修改"（原有 bug，顺手修复）
        self._loading = True
        try:
            for path, label, ftype, rule, desc in FIELDS:
                if path not in self.vars:
                    continue
                val = _get(cfg, path, '')
                if ftype == 'bool':
                    self.vars[path].set(bool(val))
                else:
                    self.vars[path].set('' if val is None else str(val))
        finally:
            self._loading = False
        self.dirty = False
        # BUG-4：按当前 active_model 锁定无效字段（启动即生效）
        self._apply_model_field_locks()

    def collect_from_form(self):
        payload = {}
        for path, label, ftype, rule, desc in FIELDS:
            if path not in self.vars:
                continue
            var = self.vars[path]
            payload[path] = var.get()
        return payload

    def _mark_dirty(self, path):
        if getattr(self, '_loading', False):
            return   # 表单加载中，不标脏（load_into_form 已抑制）
        self.dirty = True
        if path in self.errs:
            self.errs[path].set('')
        self.status_var.set('有未保存的修改')

    # ── 保存 / 恢复 ────────────────────────────────────────
    def on_save(self):
        self._try_save()

    def on_reset(self):
        if self.running:
            messagebox.showwarning('运行中', '每日流程正在运行，请稍后再操作。')
            return
        if not messagebox.askyesno('恢复默认', '确定恢复默认配置？当前修改会丢失。'):
            return
        try:
            save_config(build_defaults())
        except Exception as e:
            messagebox.showerror('恢复默认失败', str(e))
            return
        self.load_into_form()
        self.dirty = False
        self.status_var.set(f'已恢复默认 {datetime.now():%H:%M:%S}')
        self.after(100, self.refresh_status_async)

    def on_close(self):
        if self.running:
            messagebox.showwarning('运行中', '每日流程正在运行，请等待完成后再退出。')
            return
        if self.dirty and not messagebox.askyesno('未保存', '有未保存的修改，确定退出？'):
            return
        self.destroy()

    # ── 选目录 ────────────────────────────────────────────
    def on_pick_dir(self, path):
        current = str(self.vars[path].get()).strip()
        initial = current if (current and os.path.isdir(current)) else os.path.expanduser('~')
        sel = filedialog.askdirectory(title=f'选择 {path}', initialdir=initial)
        if sel:
            self.vars[path].set(sel.replace('\\', '/'))
            self._mark_dirty(path)

    def _open_path_dir(self, path):
        """在系统文件管理器中打开指定路径字段对应的目录（不存在则提示）。"""
        import subprocess as _sp
        target = str(self.vars[path].get()).strip()
        if not target:
            messagebox.showwarning('打开', f'尚未填写 {path}，请先用「浏览…」选择。')
            return
        if not os.path.isdir(target):
            messagebox.showwarning(
                '打开', f'目录不存在：{target}\n\n可先用「浏览…」选择正确的 {path}。')
            return
        try:
            if sys.platform == 'win32':
                _sp.Popen(['explorer', target])
            elif sys.platform == 'darwin':
                _sp.Popen(['open', target])
            else:
                _sp.Popen(['xdg-open', target])
        except Exception as e:
            messagebox.showwarning('打开失败', str(e))

    def on_auto_detect(self):
        """自动识别项目根：找当前项目目录内/上级目录的 deploy_root.marker，
        读取其中各相对路径，解析为绝对路径后自动填入 项目目录/模型目录/上传目录/数据目录。"""
        marker = 'deploy_root.marker'
        base = os.path.dirname(os.path.abspath(__file__))
        # 候选位置：本目录 / 上级目录 / 数据目录上级（../data 的上级）
        cands = [base]
        for _ in range(3):
            base = os.path.dirname(base)
            cands.append(base)
        cands.append(os.path.join(_PARENT, 'data'))
        found = None
        for d in cands:
            p = os.path.join(d, marker)
            if os.path.exists(p):
                found = p
                break
        if not found:
            messagebox.showinfo(
                '自动识别',
                f'未找到 {marker} 标的文件。\n'
                f'请在项目根目录放置 {marker}（含 project_dir/model_dir/upload_dir/data_dir 的相对路径），\n'
                f'或先把项目目录填好再点「自动识别」。')
            return
        try:
            with open(found, encoding='utf-8') as f:
                rel = json.load(f)
        except Exception as e:
            messagebox.showerror('自动识别失败', f'读取 {marker} 出错：{e}')
            return
        root = os.path.dirname(found)
        mapping = {'project_dir': 'project_dir', 'model_dir': 'model_dir',
                   'upload_dir': 'upload_dir', 'data_dir': 'data_dir'}
        filled = []
        for key, vkey in mapping.items():
            rv = rel.get(vkey)
            if not isinstance(rv, str) or not rv:
                continue
            abspath = os.path.normpath(os.path.join(root, rv)).replace('\\', '/')
            if key in self.vars:
                self.vars[key].set(abspath)
                self._mark_dirty(key)
                filled.append(f'{key} = {abspath}')
        if filled:
            self.status_var.set(f'已自动识别: {os.path.basename(found)}')
            # 自动保存：识别出的路径直接写入 config.json 生效（失败则回退为手动保存提示）
            ok = self._try_save()
            if ok:
                messagebox.showinfo(
                    '自动识别完成',
                    f'已从 {found} 读取并填入：\n\n' + '\n'.join(filled) +
                    '\n\n配置已自动保存并生效。')
            else:
                messagebox.showinfo(
                    '自动识别完成',
                    f'已从 {found} 读取并填入：\n\n' + '\n'.join(filled) +
                    '\n\n保存校验未通过，请检查表单后点击「保存配置」。')
        else:
            messagebox.showinfo('自动识别', '标的文件存在，但没有可填入的相对路径字段。')

    # ── 状态刷新（后台线程）───────────────────────────────
    def refresh_status_async(self):
        threading.Thread(target=self._status_worker, daemon=True).start()

    def _status_worker(self):
        try:
            import pandas as pd
        except ImportError:
            pd = None
        st, pd_ok = take_status(pd)
        self.after(0, lambda: self._apply_status(st, pd_ok))

    def _apply_status(self, st, pd_ok):
        for key, var in self.status_vars.items():
            v = st.get(key)
            var.set('—' if v is None else str(v))
        if not pd_ok:
            self.run_status.config(text='（未安装 pandas，状态不可用）')

    # ── 一键运行 / 仅预测（后台线程）───────────────────────
    def on_run(self):
        self._start_pipeline(PIPELINE)

    def on_run_infer_only(self):
        self._start_pipeline(INFER_ONLY)

    def on_run_verify_only(self):
        self._start_pipeline(VERIFY_ONLY)

    def _start_pipeline(self, pipeline):
        if self.running or self._busy:
            return
        self.running = True
        self._busy = True
        self.run_btn.config(state='disabled')
        self.infer_btn.config(state='disabled')
        if hasattr(self, 'verify_btn'):
            self.verify_btn.config(state='disabled')
        self.progress.start(12)
        self.run_status.config(text='运行中…')
        self.clear_log()
        # 拼命令行日志：pipeline 元素是 (script, args) 元组，先展平成命令串再 join
        cmd_parts = [resolve_python()]
        for step in pipeline:
            s, a = (step if isinstance(step, tuple) else (step, []))
            cmd_parts.append(s + (' ' + ' '.join(a) if a else ''))
        self.append_log('$ ' + ' '.join(cmd_parts))
        threading.Thread(target=self._run_worker, args=(pipeline,), daemon=True).start()

    def _run_worker(self, pipeline):
        py = resolve_python()
        n_total = len(pipeline)
        for idx, step in enumerate(pipeline, start=1):
            script, args = (step if isinstance(step, tuple) else (step, []))
            step_label = f'第 {idx}/{n_total} 步 · {script}'
            self.after(0, lambda s=step_label: self.run_status.config(text=s + ' …'))
            self.after(0, self.append_log, f'\n── [{script}] ──\n')
            try:
                p = subprocess.Popen(
                    [py, '-u', script] + list(args), cwd=_HERE,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding='utf-8', errors='replace', bufsize=1)
            except Exception as e:
                self.after(0, self.append_log, f'[启动失败] {e}\n')
                break
            for line in p.stdout:
                self.after(0, self.append_log, line)
            p.wait()
            if p.returncode != 0:
                self.after(0, self.append_log, f'── [{script}] 失败（退出码 {p.returncode}），已停止 ──\n')
                self.after(0, lambda s=script: self.run_status.config(text=f'失败: {s}'))
                break
        else:
            self.after(0, lambda: self.run_status.config(text='流程完成'))
            self.after(0, self.append_log, '\n流程完成。')
        self.after(0, self.refresh_status_async)
        self.after(0, self.load_results)     # 运行结束自动刷新预测结果页
        self.after(0, self.finish_run)

    def finish_run(self):
        self.running = False
        self._busy = False
        self.progress.stop()
        self.run_btn.config(state='normal')
        self.infer_btn.config(state='normal')
        if hasattr(self, 'verify_btn'):
            self.verify_btn.config(state='normal')
        if self.run_status.cget('text') in ('', '运行中…'):
            self.run_status.config(text='')
        self._update_step_nav()   # 运行结束重新启用「下一步」

    # ── 日志 ──────────────────────────────────────────────
    def clear_log(self):
        self.log.config(state='normal')
        self.log.delete('1.0', 'end')
        self.log.config(state='disabled')

    def append_log(self, text):
        self.log.config(state='normal')
        self.log.insert('end', text)
        self.log.see('end')
        self.log.config(state='disabled')

    # ── 关于 ──────────────────────────────────────────────
    def _on_about(self):
        messagebox.showinfo(
            '关于',
            f'电力现货价差预测 · 配置管理（桌面版）\n\n'
            f'配置文件：{CONFIG_FILE}\n'
            f'默认配置：{DEFAULTS_FILE}\n\n'
            '修改保存后，下一次运行每日流程时自动生效。')


if __name__ == '__main__':
    # 生成默认配置备份（仅当缺失时；绝不覆盖 config.json）
    if not os.path.exists(DEFAULTS_FILE):
        try:
            shutil.copyfile(CONFIG_FILE, DEFAULTS_FILE)
        except Exception:
            pass
    app = ConfigApp()
    app.mainloop()
