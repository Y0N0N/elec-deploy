# deploy_v9

本目录包含本地运行的预测流水线和结果查看工具。发布仓库只保留源代码与示例配置；原始市场数据、上传文件、模型权重和运行结果需要在本地另行准备。

## 目录

- `code/common/`：数据导入、特征重建、推理、核对和归档逻辑
- `code/v9/`：v9 模型特征与包装器
- `code/v8.1/`：兼容旧模型的包装器
- `results_server.py` / `results_dashboard.html`：本地结果仪表盘
- `ui_client.py` / `ui_desktop.py`：配置与运行入口

## 使用

1. 将 `config.json` 中的路径改为本地数据目录。
2. 将外部数据放入 `upload/`，将训练模型放入 `models/`。
3. 在本目录执行 `bash run_all.sh`，或单独运行 `code/common/` 中的脚本。
4. 执行 `python results_server.py` 查看本地仪表盘。

运行产物、原始数据和模型文件均被 Git 忽略。
