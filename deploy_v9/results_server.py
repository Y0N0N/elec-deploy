#!/usr/bin/env python
# ============================================================
# deploy/results_server.py — 预测结果 Web Dashboard 服务
#
# 作用：给「预测结果」提供本地 web 前端（纯本地零依赖，无 CDN）。
#   - GET /               → dashboard 页面（results_dashboard.html）
#   - GET /api/results    → 扫描 output/*.csv，返回全部预测结果
#   - GET /api/model      → 活动模型信息（model_type/阈值/v9 指标）
#   - GET /api/status     → 状态（披露日/实际日/模型/输出）
#
# 由 ui_desktop.py 的「打开前端仪表盘」按钮启动，或命令行直接跑：
#   python3 results_server.py              # http://127.0.0.1:8301
#   python3 results_server.py --port 9000  # 换端口
# ============================================================
import argparse
import csv
import io
import json
import os
import sys
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from flask import Flask, jsonify, send_from_directory

try:
    from _cfg import cfg
except Exception:
    cfg = None

try:
    from _verify import CHECK_LABEL, parse_filename
except Exception:
    def parse_filename(fname): return None
    CHECK_LABEL = {}

app = Flask(__name__, static_folder=None)

# 允许跨源（本机浏览器打开页面访问本机 API，同源，其实不需要 CORS，
# 但留开发便利）
@app.after_request
def _cors(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


def _load_config():
    """加载 config.json（复用 ui_client 的逻辑，独立进程避免 import 它的 Flask app）。"""
    try:
        p = os.path.join(_HERE, 'config.json')
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _sort_output_files(files):
    """按预测窗口排序，最新优先（文件名 预测_YYYY-MM-DD_YYYY-MM-DD.csv）。"""
    from datetime import date

    def key(f):
        try:
            name = f[:-4] if f.endswith('.csv') else f
            dates = [p for p in name.split('_')
                     if len(p) == 10 and p[4] == '-' and p[7] == '-']
            if len(dates) >= 2:
                e = date.fromisoformat(dates[-1]).toordinal()
                s = date.fromisoformat(dates[0]).toordinal()
                return (-e, s)
            if len(dates) == 1:
                return (-date.fromisoformat(dates[0]).toordinal(), 0)
        except Exception:
            pass
        return (1e15, 0)
    return sorted(files, key=key)


def _read_csv(path):
    """读 output CSV（utf-8-sig BOM），返回 {columns, rows, is_v9}。"""
    with open(path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)
    cols = list(dict.fromkeys(header))
    is_v9 = '模型方向' in cols
    return cols, rows, is_v9


def _parse_dates(fname):
    """从文件名解析预测窗口日期。"""
    try:
        name = fname[:-4]
        dates = [p for p in name.split('_')
                 if len(p) == 10 and p[4] == '-' and p[7] == '-']
        return (dates[0], dates[-1]) if len(dates) >= 2 else (None, None)
    except Exception:
        return (None, None)


# ────────────────────────────────────────────────────────────
# API
# ────────────────────────────────────────────────────────────
@app.route('/api/results')
def api_results():
    """全部预测结果：文件名 + 窗口 + 格式 + 元数据(模型/核对标记) + 行数据。"""
    out_dir = os.path.join(_HERE, 'output')
    results = []
    if os.path.isdir(out_dir):
        for fn in _sort_output_files(f for f in os.listdir(out_dir)
                                      if f.endswith('.csv')):
            try:
                cols, rows, is_v9 = _read_csv(os.path.join(out_dir, fn))
                d0, d1 = _parse_dates(fn)
                meta = parse_filename(fn)
                results.append({
                    'file': fn,
                    'start': d0,
                    'end': d1,
                    'model': (meta or {}).get('model'),
                    'flag': (meta or {}).get('flag'),
                    'flag_label': CHECK_LABEL.get((meta or {}).get('flag'), '—'),
                    'legacy': (meta or {}).get('legacy', True),
                    'days': len(sorted({r.get('日期', '') for r in rows})),
                    'hours': len(rows),
                    'is_v9': is_v9,
                    'columns': cols,
                    'rows': rows,
                })
            except Exception as e:
                results.append({'file': fn, 'error': str(e)})
    return jsonify({'results': results})


@app.route('/api/model')
def api_model():
    """活动模型信息：model_type / 阈值 / 训练时间 / 指标。"""
    cfg_now = _load_config()
    active = cfg_now.get('active_model', 'v7')
    reg = (cfg_now.get('model_registry') or {}).get(active, {})
    spread = cfg_now.get('spread', {})
    info = {
        'active': active,
        'label': reg.get('label', active),
        'threshold': spread.get('threshold', 50.0),
        'threshold_big': spread.get('threshold_big', 100.0),
        'model_type': None,
        'trained_at': None,
        'metrics': None,
        'predict_mode': reg.get('predict_mode', 'spread'),
    }
    # 读最新模型 joblib 拿 model_type / 指标
    try:
        import joblib
        model_dir = cfg_now.get('model_dir') or os.path.join(_HERE, 'models')
        pattern = reg.get('pattern', f'xgb_{active}_{{ts}}.joblib')
        prefix = pattern.split('{ts}')[0]
        cands = [f for f in os.listdir(model_dir)
                 if f.startswith(prefix) and f.endswith('.joblib')]
        if cands:
            cands.sort(key=lambda f: os.path.getmtime(os.path.join(model_dir, f)))
            m = joblib.load(os.path.join(model_dir, cands[-1]))
            info['model_type'] = m.get('model_type')
            info['trained_at'] = m.get('trained_at')
            info['metrics'] = m.get('metrics')
            info['valid_c_hit'] = m.get('valid_c_hit')
            info['valid_trigger_rate'] = m.get('valid_trigger_rate')
            info['valid_dir_hit'] = m.get('valid_dir_hit')
    except Exception:
        pass
    return jsonify(info)


@app.route('/api/status')
def api_status():
    """状态卡片：最新披露日 / 实际日 / 模型 / 输出。"""
    st = {'latest_disclosure': None, 'latest_actual': None,
          'latest_model': None, 'latest_output': None}
    if cfg is not None:
        try:
            import pyarrow.feather as pyf
            import pandas as pd
            raw = cfg.DISCLOSURE_RAW
            if os.path.exists(raw):
                df = pyf.read_table(raw).to_pandas()
                dates = df.index.get_level_values(0).astype(str)
                st['latest_disclosure'] = str(sorted(dates.unique())[-1])
        except Exception:
            pass
        try:
            import pandas as pd
            p = os.path.join(cfg.ACTUAL_MATRIX, '日前统一结算价.feather')
            if os.path.exists(p):
                df = pd.read_feather(p)
                st['latest_actual'] = str(df.index.max())
        except Exception:
            pass
    cfg_now = _load_config()
    try:
        ptr_file = os.path.join(_HERE, 'latest_model.json')
        if os.path.exists(ptr_file):
            ptr = json.load(open(ptr_file))
            st['latest_model'] = ptr.get(cfg_now.get('active_model'))
    except Exception:
        pass
    try:
        out_dir = os.path.join(_HERE, 'output')
        if os.path.isdir(out_dir):
            outs = [f for f in os.listdir(out_dir) if f.endswith('.csv')]
            if outs:
                st['latest_output'] = _sort_output_files(outs)[0]
    except Exception:
        pass
    return jsonify(st)


@app.route('/')
def index():
    return send_from_directory(_HERE, 'results_dashboard.html')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='预测结果 Web Dashboard 服务')
    ap.add_argument('--host', type=str, default='127.0.0.1',
                    help='监听地址（默认 127.0.0.1）')
    ap.add_argument('--port', type=int, default=8301, help='端口（默认 8301）')
    args = ap.parse_args()
    print('=' * 56)
    print(f'  预测结果 Dashboard: http://{args.host}:{args.port}')
    print('  输出目录: ' + os.path.join(_HERE, 'output'))
    print('=' * 56)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
