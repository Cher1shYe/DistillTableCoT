"""
GRPO 的多轮 SQL ReAct rollout —— 让 GRPO 采样时**真执行 SQL**,与教师 sql_agent 一致。

接 TRL 1.6.0 GRPOTrainer 的 rollout_func 钩子 (GRPOTrainer(..., rollout_func=...))。
对每个 prompt 多轮:生成停在 </SQL> → 在该题 SQLite 表上真执行 → 注入真实
<EXECUTION_RESULT> → 续写,直到写出 <ANSWER> 或到 max_turns (连续空结果 max_empty 次早停),
与 run_teacher_evaluate 的 sql_agent 完全对齐。

返回 TRL 要求的 dict:prompt_ids / completion_ids / logprobs,**外加 env_mask**
(1=模型生成的 token、0=注入的执行结果 token)。TRL 内部把 env_mask 当 tool_mask,
在 loss 里 completion_mask * tool_mask,于是**注入的执行结果不参与梯度**——模型只学
"怎么写 SQL、怎么读结果作答",不会被训练去"背"执行结果。

注:rollout 逐 prompt 串行 + 每条末尾一次前向算 logprob,正确性优先、速度一般;
direct/cot 路径没有 <SQL> → 一轮即结束,不受影响。
"""
import re

import torch

_SQL_RE = re.compile(r"<SQL>\s*(.*?)\s*</SQL>", re.DOTALL | re.IGNORECASE)


def _exec_sql(raw_table, task, sql):
    """在该题表上真执行一条 SQL,返回反馈文本 (复用 teacher/eval 同一套 utils)。"""
    if not sql:
        return "SQL Error: no <SQL> block found."
    if raw_table is None:
        return "SQL Error: table unavailable."
    from utils import table_to_sqlite, execute_sql
    conn, _ = table_to_sqlite(raw_table, task_name=task)
    if conn is None:
        return "SQL Error: failed to build database."
    try:
        _, feedback = execute_sql(conn, sql)
    finally:
        conn.close()
    return feedback


@torch.no_grad()
def rollout_one(model, tok, prompt_ids, task, raw_table, device, gen_kwargs,
                max_new_tokens, max_turns=5, max_empty=2):
    """对单个 prompt 跑多轮 ReAct,返回 (completion_ids, env_mask, tool_calls)。

    completion_ids 是 prompt 之后的全部 token (模型生成 + 注入的执行结果交错);
    env_mask 与之等长,1=模型生成 (参与训练)、0=注入 (屏蔽)。
    """
    comp_ids, env_mask = [], []
    tool_calls = 0
    empty = 0
    for turn in range(max_turns):
        cur = prompt_ids + comp_ids
        input_ids = torch.tensor([cur], device=device)
        attn = torch.ones_like(input_ids)
        last = (turn == max_turns - 1) or (empty >= max_empty)
        stop = None if last else ["</SQL>"]
        gen = model.generate(input_ids=input_ids, attention_mask=attn,
                             max_new_tokens=max_new_tokens,
                             stop_strings=stop,
                             tokenizer=(tok if stop else None),
                             **gen_kwargs)
        new_ids = gen[0][input_ids.shape[1]:].tolist()
        # 去掉尾部 pad；遇到 eos 视为生成结束
        while new_ids and new_ids[-1] == tok.pad_token_id:
            new_ids.pop()
        finished_eos = tok.eos_token_id in new_ids
        comp_ids += new_ids
        env_mask += [1] * len(new_ids)

        text = tok.decode(new_ids, skip_special_tokens=True)
        if last or finished_eos or "</sql>" not in text.lower():
            break  # 收尾 (<ANSWER>/eos) 或到上限

        # 取轨迹里最后一条 (=最新、未执行的) SQL,真执行,注入真实结果
        sqls = _SQL_RE.findall(tok.decode(comp_ids, skip_special_tokens=True))
        feedback = _exec_sql(raw_table, task, sqls[-1].strip() if sqls else None)
        if sqls:
            tool_calls += 1
        if "no results" in str(feedback).lower():
            empty += 1
        inj = tok(f"\n<EXECUTION_RESULT>\n{feedback}\n</EXECUTION_RESULT>\n",
                  add_special_tokens=False)["input_ids"]
        comp_ids += inj
        env_mask += [0] * len(inj)   # 注入的执行结果:不训练
    return comp_ids, env_mask, tool_calls


@torch.no_grad()
def completion_logps(model, prompt_ids, comp_ids, device, temperature):
    """对 prompt+completion 前向一遍,取 completion 各 token 在采样策略下的 logprob。

    GRPO 要 rollout 回传 logprobs (变成 sampling_per_token_logps 做重要性采样校正)。
    用与采样一致的 temperature 缩放 logits,口径与 TRL 内部一致。
    """
    if not comp_ids:
        return []
    full = torch.tensor([prompt_ids + comp_ids], device=device)
    logits = model(full).logits[0]                       # (L, V)
    plen = len(prompt_ids)
    sel = logits[plen - 1: plen - 1 + len(comp_ids)]     # 预测每个 completion token 的位置
    logp = torch.log_softmax(sel.float() / max(temperature, 1e-6), dim=-1)
    comp_t = torch.tensor(comp_ids, device=device).unsqueeze(1)
    return logp.gather(1, comp_t).squeeze(1).tolist()


def make_sql_rollout_func(prompt_lookup, max_new_tokens=640, max_turns=5, max_empty=2):
    """构造 TRL GRPOTrainer 用的 rollout_func。

    Args:
        prompt_lookup: {prompt_str: (task, raw_table)}，rollout 据此知道每个 prompt 该用哪张表执行。
                       prompt_str 必须与数据集 "prompt" 列逐字一致 (TRL 原样透传)。
        max_new_tokens/max_turns/max_empty: 每轮生成上限 / 最大轮数 / 连续空结果早停 (镜像教师)。
    """
    def rollout(prompts, trainer):
        from trl.models import unwrap_model_for_generation

        tok = trainer._tokenizer
        device = trainer.accelerator.device
        temperature = max(getattr(trainer, "temperature", 1.0) or 1.0, 1e-6)
        gen_kwargs = {"do_sample": True, "temperature": temperature,
                      "pad_token_id": tok.pad_token_id}
        for k in ("top_p", "top_k", "min_p", "repetition_penalty"):
            v = getattr(trainer, k, None)
            if v is not None:
                gen_kwargs[k] = v

        out = {"prompt_ids": [], "completion_ids": [], "logprobs": [], "env_mask": []}
        with unwrap_model_for_generation(
                trainer.model_wrapped, trainer.accelerator,
                gather_deepspeed3_params=trainer.args.ds3_gather_for_generation) as model:
            for prompt in prompts:
                task, raw_table = prompt_lookup.get(prompt, (None, None))
                p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
                comp_ids, env_mask, _ = rollout_one(
                    model, tok, p_ids, task, raw_table, device, gen_kwargs,
                    max_new_tokens, max_turns, max_empty)
                logps = completion_logps(model, p_ids, comp_ids, device, temperature)
                out["prompt_ids"].append(p_ids)
                out["completion_ids"].append(comp_ids)
                out["logprobs"].append(logps)
                out["env_mask"].append(env_mask)
        return out

    return rollout
