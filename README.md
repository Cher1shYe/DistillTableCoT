# DistillTableCoT

本项目旨在探索和评估小模型(Qwen3-1.7B)在表格问答任务表现上的最优蒸馏方案，并支持混合推理（如 SQL-Agent 与 CoT 结合）的模型蒸馏微调。

## 支持的任务

- `wikitableqa`: 表格问答
- `tabfact`: 表格事实判断
- `fetaqa`: 自由格式的表格问答
- `hitab`: 层次化表格问答与计算

## 项目结构

```text
DistillTableCoT/
├── configs/                  # 训练配置文件目录
├── data_loader/              # 数据加载相关
├── local_datasets/           # 本地数据集存储
├── outputs/                  # 存放所有模型输出与预测结果
├── scripts/                  # 训练蒸馏脚本目录
├── utils_train/              # 训练辅助函数
├── configs.py                # 任务配置中心 (Prompt及后处理逻辑)
├── utils.py                  # 通用辅助函数
├── run_inference.py          # 运行大模型推理并保存结果
├── run_evaluate.py           # 读取结果并进行评估
└── README.md                 # 本说明文件
```

## 使用说明

### 1. 环境准备

请确保您已安装 Python 3.8+。

**安装依赖:**
```bash
pip install -r requirements.txt
```

本项目在数据处理和评估阶段需要 NLTK 的 `punkt` 和 `punkt_tab` 数据包，可通过以下快捷命令下载：
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

**设置 API Key (如使用 DeepSeek API 进行推理蒸馏数据生成):**
将您的 DeepSeek API 密钥设置为环境变量：
```bash
# Linux / macOS
export DEEPSEEK_API_KEY="your_api_key_here"

# Windows (CMD)
set DEEPSEEK_API_KEY=your_api_key_here

# Windows (PowerShell)
$env:DEEPSEEK_API_KEY="your_api_key_here"
```

### 2. 生成预测结果

使用 `run_inference.py` 脚本来调用 API（或模型）生成预测。需指定任务名称和样本数量。

**示例:** 为 `wikitableqa` 任务的前 5 个样本生成预测。
```bash
python run_inference.py --task_name wikitableqa --num_samples 5
```
运行后，结果将保存在 `outputs/wikitableqa/predictions.json` 或类似路径下。

**其他任务示例:**
```bash
python run_inference.py --task_name tabfact --num_samples 5
python run_inference.py --task_name fetaqa --num_samples 5
python run_inference.py --task_name hitab --num_samples 5
```

### 3. 评估预测结果

生成预测文件后，使用 `run_evaluate.py` 脚本来计算评估指标（如 Exact Match, Accuracy, ROUGE 等）。

**示例:** 评估 `wikitableqa` 下文件`predicion.json`的预测结果。
```bash
python run_evaluate.py --task_name wikitableqa --pred_file prediction.json
```
脚本会自动读取对应的预测结果数据并打印评估分数。

### 4. 训练蒸馏模型

设置好 `configs/` 下的配置文件后，使用 `scripts/train_distill.py` 脚本训练模型，本项目支持多种范式的训练，配置参数在configs文件夹下：

```bash
# 以混合范式蒸馏 Qwen3-1.7B 模型
python scripts/train_distill.py --config configs/qwen3-mixed.yaml

```
训练输出的模型和日志默认保存在 `outputs/models/` 目录下。

### 5. 测试微调模型

在网络访问 Hugging Face 通畅的情况下，使用 `scripts/test_model.py` 输出评估任务的结果，测试代码支持多种参数，如提示词范式，max_tokens等：

```bash
python scripts/test_model.py \
    --task_name wikitableqa \
    --num_samples 100 \
    --model_path outputs/models/your_model_dir \
    --out_name predictions_qwen.json \
    --max_new_tokens 2048 \
    --prompt_mode mixed_agent \
    --split test
```

### 6. 评估模型结果

本项目设计了自动化评估脚本`scripts/batch_eval_qwen3.py`，其会自动阅读`outputs/数据集/`目录下相应小模型与大模型的预测结果文件，并生成评估报告，示例命令如下：

```bash
python scripts/batch_eval_qwen3.py
```