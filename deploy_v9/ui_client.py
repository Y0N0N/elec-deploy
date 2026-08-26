#!/usr/bin/env python
# ============================================================
# deploy/ui_client.py — 配置管理 UI 客户端（网页版）
#
# 作用：用一个简单的网页表单查看/修改 config.json，
#       不用手改配置文件，避免改错导致脚本出错。
#
# 用法：
#   python ui_client.py                  # 默认 http://127.0.0.1:8300
#   python ui_client.py --port 9000      # 换端口
#   python ui_client.py --host 0.0.0.0   # 允许局域网/远程访问（注意安全）
#
# 打开浏览器访问对应地址即可。保存后立即生效（下一次运行脚本时读取）。
# ============================================================
import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd          # 主线程预加载原生库，避免 Flask 请求线程首次 import 段错误

_HERE = os.path.dirname(os.path.abspath(__file__))
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
]


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
    """从 0_config 的默认值生成 config.defaults.json（重置用）"""
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
# 校验
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
    # 先基于现有 config 复制，再覆盖提交的字段（保留 registry 等结构性内容）
    cfg_now = load_config()
    for path, label, ftype, rule, desc in FIELDS:
        if path in ('data_dir', 'project_dir'):
            continue   # 高级字段单独校验
        if path in payload:
            val, err = validate_field(path, payload[path], rule)
            if err:
                errors[path] = err
            else:
                _set(cfg_now, path, val)
    # 高级字段（路径）只允许已存在目录或非空
    for path in ('data_dir', 'project_dir'):
        if path in payload and str(payload[path]).strip():
            _set(cfg_now, path, str(payload[path]).strip())
    # 交叉校验
    if not errors:
        t1 = cfg_now['spread']['threshold']
        t2 = cfg_now['spread']['threshold_big']
        if t2 <= t1:
            errors['spread.threshold_big'] = '大偏差阈值必须大于小偏差阈值'
    return cfg_now, errors


# ────────────────────────────────────────────────────────────
# Flask 应用
# ────────────────────────────────────────────────────────────
from flask import Flask, jsonify, request, render_template_string

app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>电力价差预测 · 配置管理</title>
<style>
  :root { --bg:#0f1420; --card:#171e2e; --line:#26304a; --txt:#e6eaf3;
          --dim:#8b95ad; --accent:#4c8dff; --ok:#37c08c; --warn:#e8b93c; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--txt);
         font-family: -apple-system,'PingFang SC','Microsoft YaHei',sans-serif; }
  header { padding:20px 28px; border-bottom:1px solid var(--line); }
  header h1 { margin:0; font-size:20px; }
  header p { margin:6px 0 0; color:var(--dim); font-size:13px; }
  main { max-width:860px; margin:24px auto; padding:0 20px 60px; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
           gap:12px; margin:18px 0 26px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:12px 16px; }
  .card .k { font-size:12px; color:var(--dim); }
  .card .v { font-size:16px; margin-top:4px; font-weight:600; }
  .card .v small { font-size:12px; color:var(--dim); font-weight:400; }
  section { background:var(--card); border:1px solid var(--line); border-radius:12px;
            padding:18px 22px; margin-bottom:16px; }
  section h2 { margin:0 0 4px; font-size:15px; }
  section .sec-desc { margin:0 0 14px; color:var(--dim); font-size:12px; }
  .field { margin-bottom:12px; }
  .field label { display:block; font-size:13px; margin-bottom:4px; }
  .field label small { color:var(--dim); font-weight:400; margin-left:6px; }
  .field input[type=number], .field input[type=text], .field select {
    width:100%; max-width:320px; background:#0c1220; color:var(--txt);
    border:1px solid var(--line); border-radius:6px; padding:8px 10px; font-size:14px; }
  .field input[type=checkbox] { width:18px; height:18px; accent-color:var(--accent); }
  .field .desc { font-size:12px; color:var(--dim); margin-top:3px; }
  .field .err { color:#ff7b72; font-size:12px; margin-top:3px; }
  .actions { display:flex; gap:12px; align-items:center; margin-top:6px; }
  button { background:var(--accent); color:#fff; border:none; border-radius:8px;
           padding:10px 22px; font-size:14px; cursor:pointer; }
  button.secondary { background:#2a3550; }
  button:hover { filter:brightness(1.1); }
  #msg { font-size:13px; margin-left:8px; }
  #msg.ok { color:var(--ok); } #msg.err { color:#ff7b72; }
  .model-table { width:100%; border-collapse:collapse; font-size:13px; }
  .model-table th,.model-table td { text-align:left; padding:6px 8px;
    border-bottom:1px solid var(--line); }
  .model-table .active { color:var(--accent); font-weight:600; }
  .hint { font-size:12px; color:var(--dim); margin-top:14px; line-height:1.7; }
</style>
</head>
<body>
<header>
  <h1>⚡ 电力现货价差预测 · 配置管理</h1>
  <p>修改后点「保存」立即生效（下一次运行 run_all.sh 时读取）。无需重启。</p>
</header>
<main>
  <div class="cards">
    <div class="card"><div class="k">最新披露日</div><div class="v">{{ st.latest_disclosure or '—' }}</div></div>
    <div class="card"><div class="k">最新实际结果日</div><div class="v">{{ st.latest_actual or '—' }}</div></div>
    <div class="card"><div class="k">最新模型</div><div class="v" style="font-size:13px">{{ st.latest_model or '—' }}</div></div>
    <div class="card"><div class="k">最近预测输出</div><div class="v" style="font-size:13px">{{ st.latest_output or '—' }}</div></div>
  </div>

  <form id="cfg-form">
    <section>
      <h2>📦 模型</h2>
      <p class="sec-desc">推理/重建用哪个模型。各模型说明见下方列表。</p>
      <div class="field">
        <label>当前模型</label>
        <select name="active_model">
          {% for k, v in cfg.model_registry.items() %}
          <option value="{{ k }}" {% if k == cfg.active_model %}selected{% endif %}>{{ k }}</option>
          {% endfor %}
        </select>
        <div class="desc">选择后，2A 会重训该模型、2B 会用它推理</div>
      </div>
      <table class="model-table">
        <tr><th>key</th><th>说明</th><th>预测类型</th><th>文件名模板</th></tr>
        {% for k, v in cfg.model_registry.items() %}
        <tr class="{{ 'active' if k == cfg.active_model }}">
          <td>{{ k }}</td><td>{{ v.label }}</td><td>{{ v.predict_mode }}</td><td>{{ v.pattern }}</td>
        </tr>
        {% endfor %}
      </table>
    </section>

    <section>
      <h2>🗓 预测窗口</h2>
      <p class="sec-desc">参考日 D = 最新披露日 - 1 天。窗口 = [D - 往前天数, D + 往后天数]</p>
      {% for path, label, ftype, rule, desc in fields %}
        {% if path.startswith('predict_window') %}
        <div class="field">
          <label>{{ label }}</label>
          <input type="number" name="{{ path }}" value="{{ val(path) }}">
          <div class="desc">{{ desc }}</div>
        </div>
        {% endif %}
      {% endfor %}
    </section>

    <section>
      <h2>🎯 价差阈值</h2>
      <p class="sec-desc">判定"正常 / 有偏差 / 大偏差"的界限</p>
      {% for path, label, ftype, rule, desc in fields %}
        {% if path.startswith('spread') %}
        <div class="field">
          <label>{{ label }}</label>
          <input type="number" step="0.5" name="{{ path }}" value="{{ val(path) }}">
          <div class="desc">{{ desc }}</div>
        </div>
        {% endif %}
      {% endfor %}
    </section>

    <section>
      <h2>🔧 因子重建</h2>
      <p class="sec-desc">重建因子时的策略</p>
      <div class="field">
        <label>重建模式</label>
        <select name="factor_rebuild.mode">
          <option value="full" {% if cfg.factor_rebuild.mode == 'full' %}selected{% endif %}>full（全量，最稳）</option>
          <option value="tail" {% if cfg.factor_rebuild.mode == 'tail' %}selected{% endif %}>tail（尾部增量，快）</option>
        </select>
      </div>
      <div class="field">
        <label>尾部窗口天数</label>
        <input type="number" name="factor_rebuild.tail_days" value="{{ cfg.factor_rebuild.tail_days }}">
        <div class="desc">tail 模式下重算最近多少天（滚动因子需 35 天历史）</div>
      </div>
      <div class="field">
        <label><input type="checkbox" name="factor_rebuild.rebuild_sp_wow"
          {% if cfg.factor_rebuild.rebuild_sp_wow %}checked{% endif %}> 重建价差状态因子 sp_wow</label>
      </div>
    </section>

    <section>
      <h2>🎓 模型训练</h2>
      <p class="sec-desc">重建模型时的数据划分与训练参数（红线：验证集绝不混入训练集）</p>
      {% for path, label, ftype, rule, desc in fields %}
        {% if path.startswith('training') and 'fixed_params' not in path %}
        <div class="field">
          <label>{{ label }}</label>
          <input type="number" name="{{ path }}" value="{{ val(path) }}">
          <div class="desc">{{ desc }}</div>
        </div>
        {% endif %}
      {% endfor %}
    </section>

    <section>
      <h2>🌲 XGBoost 超参</h2>
      {% for path, label, ftype, rule, desc in fields %}
        {% if 'fixed_params' in path %}
        <div class="field">
          <label>{{ label }}</label>
          <input type="number" step="0.01" name="{{ path }}" value="{{ val(path) }}">
          <div class="desc">{{ desc }}</div>
        </div>
        {% endif %}
      {% endfor %}
    </section>

    <section>
      <h2>📁 高级：路径</h2>
      <p class="sec-desc">一般不要改。改了要确保目录真实存在。</p>
      {% for path, label, ftype, rule, desc in fields %}
        {% if path in ('data_dir', 'project_dir') %}
        <div class="field">
          <label>{{ label }}</label>
          <input type="text" name="{{ path }}" value="{{ val(path) }}">
          <div class="desc">{{ desc }}</div>
        </div>
        {% endif %}
      {% endfor %}
    </section>

    <div class="actions">
      <button type="button" onclick="save()">💾 保存配置</button>
      <button type="button" class="secondary" onclick="reset()">↺ 恢复默认</button>
      <span id="msg"></span>
    </div>
  </form>

  <div class="hint">
    💡 保存后无需重启，下一次运行 <code>bash run_all.sh</code> 时自动生效。<br>
    ⚠️ 脚本运行中请勿修改配置（可能导致读写冲突）。<br>
    配置文件位置：<code>deploy/config.json</code>（也可直接手改）。
  </div>
</main>
<script>
function save() {
  const form = document.getElementById('cfg-form');
  const data = {};
  new FormData(form).forEach((v, k) => { data[k] = v; });
  fetch('/api/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  }).then(r => r.json()).then(res => {
    const msg = document.getElementById('msg');
    if (res.ok) {
      msg.className = 'ok'; msg.textContent = '✅ ' + res.message;
      setTimeout(() => location.reload(), 800);
    } else {
      msg.className = 'err';
      msg.textContent = '❌ ' + (res.errors ? Object.values(res.errors).join('；') : res.message);
    }
  });
}
function reset() {
  if (!confirm('确定恢复默认配置？当前修改会丢失。')) return;
  fetch('/api/reset', {method: 'POST'}).then(r => r.json()).then(res => {
    location.reload();
  });
}
</script>
</body>
</html>
"""


@app.route('/')
def index():
    cfg_now = load_config()
    st = get_status()
    return render_template_string(
        HTML, cfg=cfg_now, st=st, fields=FIELDS,
        val=lambda p: _get(cfg_now, p, ''))


@app.route('/api/status')
def api_status():
    return jsonify(get_status())


@app.route('/api/save', methods=['POST'])
def api_save():
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify(ok=False, message='请求格式错误'), 400
    cleaned, errors = validate_all(payload)
    if errors:
        return jsonify(ok=False, errors=errors), 400
    save_config(cleaned)
    return jsonify(ok=True, message='已保存，下次运行生效')


@app.route('/api/reset', methods=['POST'])
def api_reset():
    save_config(build_defaults())
    return jsonify(ok=True, message='已恢复默认配置')


def get_status():
    """只读状态信息"""
    from _cfg import cfg
    st = {'latest_disclosure': None, 'latest_actual': None,
          'latest_model': None, 'latest_output': None}
    # 最新披露日
    try:
        mx = None
        for f in sorted(os.listdir(cfg.DISCLOSURE_MATRIX))[:30]:
            if f.endswith('.feather'):
                df = pd.read_feather(os.path.join(cfg.DISCLOSURE_MATRIX, f))
                if mx is None or str(df.index.max()) > mx:
                    mx = str(df.index.max())
        st['latest_disclosure'] = mx
    except Exception:
        pass
    # 最新实际结果日
    try:
        df = pd.read_feather(os.path.join(cfg.ACTUAL_MATRIX, '日前统一结算价.feather'))
        st['latest_actual'] = str(df.index.max())
    except Exception:
        pass
    # 最新模型
    try:
        if os.path.exists(cfg.LATEST_MODEL_FILE):
            ptr = json.load(open(cfg.LATEST_MODEL_FILE))
            st['latest_model'] = ptr.get(cfg.ACTIVE_MODEL) or '（无指针）'
    except Exception:
        pass
    # 最近输出
    try:
        outs = sorted(f for f in os.listdir(cfg.OUTPUT_DIR) if f.endswith('.csv'))
        st['latest_output'] = outs[-1] if outs else None
    except Exception:
        pass
    return st


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='配置管理 UI 客户端')
    ap.add_argument('--host', type=str, default='127.0.0.1',
                    help='监听地址（默认 127.0.0.1；局域网访问用 0.0.0.0）')
    ap.add_argument('--port', type=int, default=8300, help='端口（默认 8300）')
    args = ap.parse_args()
    # 生成默认配置备份（仅当缺失时；绝不覆盖 config.json，避免清掉用户已改的配置）
    if not os.path.exists(DEFAULTS_FILE):
        try:
            with open(DEFAULTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(build_defaults(), f, ensure_ascii=False, indent=2)
            print(f"默认配置备份已生成: {DEFAULTS_FILE}")
        except Exception:
            pass
    print(f"配置 UI 已启动: http://{args.host}:{args.port}")
    print(f"配置文件: {CONFIG_FILE}")
    app.run(host=args.host, port=args.port, debug=False, threaded=False)
