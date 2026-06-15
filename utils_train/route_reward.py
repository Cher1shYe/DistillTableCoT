"""
GRPO 的 cost-aware 奖励函数 (对应 cost_aware.pdf §6.2 的 RL 化路由学习)。

核心思想:SFT 模仿 teacher 选路标签学不出逐题路由 (探针证明组间差≈0)，
改用 RL —— 奖励来自"学生自己走这条路能不能答对 + 用了多少 token":

    答错 → reward = 0           (一切白搭，与 token / route 无关)
    答对 → reward = correct_reward - token 惩罚   (实际生成 token 越多，扣越多)

**accuracy 绝对主导**:只要保证 correct_reward - max_token_penalty > 0，
任何"答对"(哪怕走最贵、最长的路) 都恒优于任何"答错"。所以模型永远倾向能答对的 route，
不管它多贵；token 只在"几条路都能答对"时做 tie-break，把它推向更省的那条。
GRPO 用组内相对优势更新，于是同一题里"答对且更省"的采样胜出 —— 天然 cost-aware。

判分严格复用 utils_train.route_scoring.score_answer，与 batch_eval / eval_route 同口径。
"""
import re

from utils_train.route_scoring import score_answer

_ROUTE_RE = re.compile(r"<ROUTE>\s*(DIRECT|COT|SQL)\s*</ROUTE>", re.IGNORECASE)
_ANSWER_RE = re.compile(r"<ANSWER>(.*?)</ANSWER>", re.IGNORECASE | re.DOTALL)
_EXEC_RE = re.compile(r"<EXECUTION_RESULT>.*?</EXECUTION_RESULT>", re.IGNORECASE | re.DOTALL)


def _to_text(completion):
    """兼容 TRL 的两种 completion 形态：纯字符串 (standard) 或消息列表 (conversational)。"""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):  # [{"role": "assistant", "content": "..."}]
        return "".join(m.get("content", "") for m in completion if isinstance(m, dict))
    return str(completion or "")


def parse_route(text):
    """抽出 <ROUTE>X</ROUTE> 里的路径名 (小写)，没有则 None。"""
    m = _ROUTE_RE.search(text or "")
    return m.group(1).lower() if m else None


def parse_answer(text):
    """抽出最后一个非空 <ANSWER>...</ANSWER>，没有则 None。"""
    for a in reversed(_ANSWER_RE.findall(text or "")):
        if a.strip():
            return a.strip()
    return None


def make_cost_aware_reward(correct_reward=1.0, max_token_penalty=0.3,
                           token_norm=640, tokenizer=None, format_reward=0.0):
    """构造 TRL GRPOTrainer 的奖励函数 —— accuracy 主导，token 仅作惩罚。

    答错 → reward = format_reward (默认 0，与 token 无关)；
    答对 → reward = correct_reward - max_token_penalty * min(实际token / token_norm, 1)。

    须保证 correct_reward - max_token_penalty > format_reward，则"任何答对"恒优于"任何答错"，
    模型永远优先选能答对的 route，token 只在都能答对时做 tie-break。

    Args:
        correct_reward:    答对基础分。
        max_token_penalty: token 惩罚上限 (生成达到 token_norm 时扣满)。须 < correct_reward。
        token_norm:        token 归一化基准，一般设 = max_completion_length。
        tokenizer:         用于数实际生成 token；None 时退化为按空白分词近似。
        format_reward:     答错但格式合法 (<ROUTE>+<ANSWER>) 时给的极小分，仅为 GRPO 冷启动
                           留点梯度，默认 0 (严格"答错一切白搭")。
    """
    def _count_tokens(text, ids):
        # SQL 真执行 rollout 时 completion 里含注入的 <EXECUTION_RESULT> (非模型生成)，
        # 只数模型生成的部分,token 惩罚口径才与教师 completion_tokens 一致。
        if "<EXECUTION_RESULT>" in text:
            model_text = _EXEC_RE.sub("", text)
            if tokenizer is not None:
                return len(tokenizer(model_text, add_special_tokens=False)["input_ids"])
            return len(model_text.split())
        if ids is not None:          # 无注入:completion_ids 最准 (TRL 直接给)
            return len(ids)
        if tokenizer is not None:
            return len(tokenizer(text, add_special_tokens=False)["input_ids"])
        return len(text.split())     # 兜底:词数近似

    def cost_aware_reward(prompts=None, completions=None, task=None, reference=None,
                          completion_ids=None, **kwargs):
        completions = completions or []
        n = len(completions)
        tasks = task if task is not None else [None] * n
        refs = reference if reference is not None else [None] * n
        cids = completion_ids if completion_ids is not None else [None] * n

        rewards = []
        for comp, tk, ref, cid in zip(completions, tasks, refs, cids):
            text = _to_text(comp)
            answer = parse_answer(text)

            correct = False
            if answer is not None and tk is not None and ref is not None:
                try:
                    correct = bool(score_answer(tk, answer, ref))
                except Exception:
                    correct = False

            if correct:
                ntok = _count_tokens(text, cid)
                penalty = max_token_penalty * min(ntok / token_norm, 1.0) if token_norm else 0.0
                rewards.append(correct_reward - penalty)
            else:
                # 答错:与 token / route 全无关，至多给个格式分 (默认 0)
                has_format = parse_route(text) is not None and answer is not None
                rewards.append(format_reward if has_format else 0.0)
        return rewards

    return cost_aware_reward
