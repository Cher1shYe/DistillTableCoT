#!/usr/bin/env python3
"""
cost-aware 路由的 GRPO 训练入口 (对应 cost_aware.pdf §6.2)。

为什么用 GRPO:route-SFT 只是模仿 teacher 的选路标签，探针证明学不出逐题路由
(组间差≈0、推理塌缩成全 Direct)。GRPO 改用强化学习，奖励直接来自"学生走这条路
能不能答对 + 这条路贵不贵"(见 utils_train/route_reward.py)，路由信号来自学生自身
可解性，天然 cost-aware。

不动现有 SFT 流水线:复用 utils_train.config_loader (YAML 继承) 与
utils_train.route_scoring (判分口径)，LoRA / 输出目录约定与 train_distill.py 一致。

强烈建议从 route-SFT 检查点热启动 (模型已会输出 <ROUTE>/<ANSWER> 格式，
GRPO 只需把"选路"从塌缩里拉出来):
    在 config 里设 model.sft_adapter_path 指向 SFT 的 final_model (LoRA adapter 目录)。

用法 (GPU / Colab):
    pip install "trl>=0.16.0"
    python3 scripts/train_grpo.py --config configs/qwen3_route_grpo.yaml
"""
import argparse
import dataclasses
import json
import os
import random
import sys
from glob import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from utils_train.config_loader import load_config  # noqa: E402
from utils_train.route_reward import make_cost_aware_reward  # noqa: E402
from utils_train.grpo_sql_rollout import make_sql_rollout_func  # noqa: E402


def resolve_model_path(model_cfg):
    """有本地权重就用本地，否则回退 HF 名称 (与 qwen_trainer 一致)。"""
    local = model_cfg.get("local_path")
    if local and os.path.exists(local):
        return local
    return model_cfg["model_name"]


def load_tokenizer(path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        if "<|endoftext|>" in tok.get_vocab():
            tok.pad_token = "<|endoftext|>"
        else:
            tok.pad_token = tok.eos_token
    return tok


def _load_tables(task, split, ids):
    """按 run_teacher_evaluate / eval_route 同样的方式加载原数据集表 (id=样本序号)。"""
    from datasets import load_dataset
    from configs import TASK_CONFIGS
    ds = load_dataset(TASK_CONFIGS[task]["dataset_name"], split=split)
    max_id = max(ids) if ids else -1
    tables = {}
    for i, sample in enumerate(ds):
        if i in ids:
            tables[i] = (sample.get("table") or sample.get("table_content")
                         or sample.get("table_text"))
        if i >= max_id:
            break
    return tables


def build_dataset(cfg, tokenizer, exec_sql=False, exec_split="train"):
    """读 route_sft jsonl，渲染成 GRPO 的 prompt 数据集 (prompt/task/reference)。

    prompt 用与 RouteSFTDataset 完全一致的 enable_thinking=False 渲染 (训练/推理一致)。
    GRPO 只需 prompt，模型自己采样补全；task/reference 透传给奖励函数判分。
    多路径数据里同一题有多行 → 按 (task,id) 去重，避免同题被 GRPO 重复采样。

    exec_sql 时同时建 prompt_lookup = {prompt_str: (task, raw_table)}，供多轮 SQL rollout
    真执行;表数据按 task 从原数据集 exec_split 加载 (GRPO 训练 v2 → train 切分)。
    返回 (dataset, prompt_lookup)。
    """
    from collections import defaultdict
    from datasets import Dataset

    seen, rows, p2meta = set(), [], {}
    for dp in cfg["data"]["data_paths"]:
        for fp in (glob(dp) if "*" in dp else [dp]):
            if not os.path.exists(fp):
                print(f"⚠️ 数据文件不存在，跳过: {fp}")
                continue
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    s = json.loads(line)
                    if not s.get("input"):
                        continue
                    key = (s.get("task"), s.get("id"))
                    if key in seen:
                        continue
                    seen.add(key)
                    msgs = [{"role": "user", "content": s["input"]}]
                    try:
                        prompt = tokenizer.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False)
                    except TypeError:
                        prompt = tokenizer.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True)
                    rows.append({"prompt": prompt,
                                 "task": s.get("task"),
                                 "reference": str(s.get("reference"))})
                    p2meta[prompt] = (s.get("task"), s.get("id"))

    random.seed(cfg["training"].get("seed", 42))
    random.shuffle(rows)
    max_samples = cfg["data"].get("max_samples", -1)
    if max_samples and max_samples > 0:
        rows = rows[:max_samples]
    print(f"GRPO 训练 prompt 数 (按 task,id 去重): {len(rows)}")

    prompt_lookup = None
    if exec_sql:
        ids_by_task = defaultdict(set)
        for r in rows:
            t, i = p2meta[r["prompt"]]
            ids_by_task[t].add(i)
        tables = {}
        for t, ids in ids_by_task.items():
            print(f"📊 加载表数据 {t} (split={exec_split}, n={len(ids)}) ...")
            tables[t] = _load_tables(t, exec_split, ids)
        prompt_lookup = {}
        for r in rows:
            t, i = p2meta[r["prompt"]]
            prompt_lookup[r["prompt"]] = (t, tables.get(t, {}).get(i))
    return Dataset.from_list(rows), prompt_lookup


def load_policy_model(cfg):
    """加载策略模型，返回 (model, is_peft)。

    配了 SFT adapter 时:加载 base + 该 adapter 作为**可训练** PeftModel，GRPO **继续训练
    这个 adapter**(不 merge)。这样 final_model 存的是相对**裸基座**的 adapter，
    eval_route 直接 base+adapter 加载即正确。(若先 merge 进 base 再叠新 adapter，
    eval 时只 base+新adapter 会缺这层 merge → 权重错位。)
    """
    from transformers import AutoModelForCausalLM

    model_path = resolve_model_path(cfg["model"])
    dtype = getattr(torch, cfg["model"].get("torch_dtype", "bfloat16"))
    print(f"加载基座: {model_path} (dtype={dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype)

    sft_adapter = cfg["model"].get("sft_adapter_path")
    if sft_adapter and os.path.exists(os.path.join(sft_adapter, "adapter_config.json")):
        from peft import PeftModel
        print(f"🔥 从 route-SFT 检查点热启动:继续训练其 LoRA adapter (不 merge): {sft_adapter}")
        model = PeftModel.from_pretrained(model, sft_adapter, is_trainable=True)
        return model, True
    if sft_adapter:
        print(f"⚠️ sft_adapter_path 无 adapter_config.json，按裸基座启动: {sft_adapter}")
    return model, False


def build_lora_config(cfg):
    if not cfg["training"].get("use_lora", False):
        return None
    from peft import LoraConfig, TaskType
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["training"].get("lora_r", 16),
        lora_alpha=cfg["training"].get("lora_alpha", 32),
        lora_dropout=cfg["training"].get("lora_dropout", 0.05),
        target_modules=cfg["training"].get(
            "lora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        bias="none",
    )


def build_grpo_config(cfg):
    """把 YAML 的 training/grpo 段映射成 GRPOConfig，按版本过滤掉不支持的字段。"""
    from trl import GRPOConfig

    t, g = cfg["training"], cfg.get("grpo", {})
    lr = t["learning_rate"]
    if isinstance(lr, str):
        lr = float(lr.strip())

    desired = dict(
        output_dir=t["output_dir"],
        num_train_epochs=t.get("num_epochs", 1),
        per_device_train_batch_size=t.get("batch_size", 8),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 4),
        learning_rate=lr,
        warmup_ratio=t.get("warmup_ratio", 0.1),
        logging_steps=t.get("logging_steps", 10),
        save_steps=t.get("save_steps", 100),
        save_total_limit=t.get("save_total_limit", 2),
        bf16=t.get("bf16", True),
        fp16=t.get("fp16", False),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        report_to=t.get("report_to", "none"),
        seed=t.get("seed", 42),
        optim=t.get("optim", "adamw_torch"),
        max_steps=t.get("max_steps", -1),   # >0 时按步数停 (smoke test 用)，-1=按 epoch
        # ---- GRPO 专属 ----
        num_generations=g.get("num_generations", 8),
        max_prompt_length=g.get("max_prompt_length", 1024),
        max_completion_length=g.get("max_completion_length", 640),
        temperature=g.get("temperature", 0.9),
        beta=g.get("beta", 0.04),
        use_vllm=g.get("use_vllm", False),
        log_completions=g.get("log_completions", True),
    )
    # 不同 trl 版本字段名略有出入，过滤掉 GRPOConfig 不认的，避免报错
    valid = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = [k for k in desired if k not in valid]
    if dropped:
        print(f"ℹ️ 当前 trl 版本不支持以下字段，已忽略: {dropped}")
    return GRPOConfig(**{k: v for k, v in desired.items() if k in valid})


def main():
    ap = argparse.ArgumentParser(description="cost-aware 路由 GRPO 训练")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--data_paths", nargs="+", default=None)
    ap.add_argument("--max_samples", type=int, default=None, help="只取前 N 条 prompt (smoke test)")
    ap.add_argument("--max_steps", type=int, default=None, help="只训 N 步就停 (smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["training"]["output_dir"] = args.output_dir
    if args.data_paths:
        cfg["data"]["data_paths"] = args.data_paths
    if args.max_samples is not None:
        cfg["data"]["max_samples"] = args.max_samples
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps
    os.makedirs(cfg["training"]["output_dir"], exist_ok=True)

    import yaml
    with open(os.path.join(cfg["training"]["output_dir"], "config_used.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    from trl import GRPOTrainer

    g = cfg.get("grpo", {})
    exec_sql = g.get("exec_sql", True)

    model_path = resolve_model_path(cfg["model"])
    tokenizer = load_tokenizer(model_path)
    train_dataset, prompt_lookup = build_dataset(
        cfg, tokenizer, exec_sql=exec_sql, exec_split=g.get("exec_split", "train"))
    model, is_peft = load_policy_model(cfg)

    # token 归一化基准默认 = max_completion_length (生成达到上限即扣满 max_token_penalty)
    token_norm = g.get("token_norm") or g.get("max_completion_length", 640)
    correct_reward = g.get("correct_reward", 1.0)
    max_token_penalty = g.get("max_token_penalty", 0.3)
    print(f"🎯 奖励:答对={correct_reward} - 至多{max_token_penalty}的token惩罚(norm={token_norm}) | "
          f"答错={g.get('format_reward', 0.0)} → 答对(最贵)≥{correct_reward - max_token_penalty:.2f} 恒>答错")
    reward_func = make_cost_aware_reward(
        correct_reward=correct_reward,
        max_token_penalty=max_token_penalty,
        token_norm=token_norm,
        tokenizer=tokenizer,
        format_reward=g.get("format_reward", 0.0),
    )

    # SQL 路径真执行:多轮 ReAct rollout (与教师一致),否则 None=TRL 默认单次生成
    rollout_func = None
    if exec_sql:
        rollout_func = make_sql_rollout_func(
            prompt_lookup,
            max_new_tokens=g.get("max_completion_length", 640),
            max_turns=g.get("max_sql_turns", 5),
            max_empty=g.get("max_empty", 2))
        print(f"🔧 SQL 多轮真执行 rollout 已启用 (max_sql_turns={g.get('max_sql_turns', 5)})")

    grpo_config = build_grpo_config(cfg)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=(None if is_peft else build_lora_config(cfg)),  # 热启动时继续训已有adapter
        rollout_func=rollout_func,
    )

    # 续训:output_dir 里已有 checkpoint 时自动恢复 (Colab 断线重跑不从头)
    last_ckpt = None
    if cfg["training"].get("resume_from_checkpoint", True):
        from transformers.trainer_utils import get_last_checkpoint
        if os.path.isdir(grpo_config.output_dir):
            last_ckpt = get_last_checkpoint(grpo_config.output_dir)
        if last_ckpt:
            print(f"Resuming from checkpoint: {last_ckpt}")
    print("Starting GRPO training...")
    trainer.train(resume_from_checkpoint=last_ckpt)

    final_dir = os.path.join(cfg["training"]["output_dir"], "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"✅ GRPO 训练完成，模型已保存: {final_dir}")


if __name__ == "__main__":
    main()
