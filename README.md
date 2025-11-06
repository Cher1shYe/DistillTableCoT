# 基本改动
添加了
- `ExamplePool`文件夹：存储example_pool的构建逻辑`example_pool.py`，以及一个调用示例池方法来构建示例池的脚本`build_ep.py`
- `ep_instances`文件夹：存储生成的基本示例池。
- `run_k_shot_experiments.py`：一个测试k_shot功能的样例脚本。
修改了 
- `run_evaluate`：为evaluate函数添加了一个新参数k_shot，用于相关的shot评测。
- `run_inference`：添加了生成k_shot_prompt的哈桑农户

## 如何运行
### 1. 构建示例池：
   ```bash
    python build_example_pool.py --task wikitableqa --num_examples 50 --pool_dir ep_instances
   ```
### 2. 对比实验
```bash
# Zero-shot 基线
python run_inference.py --task wikitableqa --num_samples 20 --k_shot 0

# One-shot 实验  
python run_inference.py --task wikitableqa --num_samples 20 --k_shot 1

# 评估对比
python run_evaluate.py --task wikitableqa
```



---

# DistillTableCoT
Distill Chain-of-Thought (CoT) from LLMs into a small language model for table reasoning(particularly TableQA) and inference.

## 支持的任务

- `wikitableqa`: 表格问答
- `tabfact`: 表格事实判断
- `fetaqa`: 自由格式的表格问答

## 项目结构

```
deepseek-table-eval/
├── outputs/                  # 存放所有模型输出的文件夹
├── ├── fetaqa/
│   │   └── predictions.json
│   ├── tabfact/
│   │   └── predictions.json
│   └── wikitableqa/
│       └── predictions.json
├── configs.py                # 任务配置中心
├── utils.py                  # 辅助函数
├── run_inference.py          # 运行推理并保存结果
├── evaluate.py               # 读取结果并进行评估
├── requirements.txt          # 项目依赖
└── README.md                 # 本说明文件
```

## 使用说明

### 1. 环境准备

首先，请确保你已经安装了 Python 3.8+。

**安装依赖:**
```bash
pip3 install -r requirements.txt
```

本项目在数据处理和评估阶段需要 NLTK 的 punkt 和 punkt_tab 数据包。你可以通过运行一个简单的 Python 命令来下载它们。

进入 Python 交互环境：
```python
python3
```
然后在 Python 中执行：

```
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
exit()
```
**设置 API Key:**
你需要将你的 DeepSeek API 密钥设置为环境变量。
```bash
# 在 Linux/macOS
export DEEPSEEK_API_KEY="your_api_key_here"

# 在 Windows (CMD)
set DEEPSEEK_API_KEY=your_api_key_here

# 在 Windows (PowerShell)
$env:DEEPSEEK_API_KEY="your_api_key_here"
```

### 2. 生成预测结果

使用 `run_inference.py` 脚本来调用 API 并生成预测。你需要指定任务名称和样本数量。

**示例:** 为 `wikitableqa` 任务的前 5 个样本生成预测。
```bash
python3 run_inference.py --task_name wikitableqa --num_samples 5
```
运行后，结果将保存在 `outputs/wikitableqa/predictions.json`。

**其他任务示例:**
```bash
# 为 tabfact 生成 5 个样本的预测
python3 run_inference.py --task_name tabfact --num_samples 5

# 为 fetaqa 生成 5 个样本的预测
python3 run_inference.py --task_name fetaqa --num_samples 5
```

### 3. 评估预测结果

生成预测文件后，使用 `evaluate.py` 脚本来计算评估指标。

**示例:** 评估 `wikitableqa` 的预测结果。
```bash
python3 evaluate.py --task_name wikitableqa
```
脚本会自动从 `outputs/wikitableqa/predictions.json` 读取数据并打印评估分数。

**评估其他任务:**
```bash
python3 evaluate.py --task_name tabfact
python3 evaluate.py --task_name fetaqa
```