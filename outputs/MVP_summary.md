# Cost-Aware 表格推理路径蒸馏 MVP 阶段汇报

## 一句话目标
让小模型 (Qwen3-1.7B) 在表格问答上**按题选最便宜的正确推理路径** (Direct < CoT < SQL)，
在尽量不掉精度的前提下省 token / 省工具调用。

## 方法
三阶段：① teacher (DeepSeek) 在 4 任务上各跑 Direct/CoT/SQL 三条路径 → ②
按 oracle (答对路径里最便宜的) 构造 `<ROUTE>X</ROUTE> + 轨迹 + <ANSWER>` 的 SFT 数据
→ ③ 干净基座Qwen3-1.7B模型LoRA微调出"既会选路又会执行"的学生模型，再评估。

---

## 一、动机验证：没有单一路径全局最优

**大模型 (teacher / DeepSeek) 三路径结果 (精度 / 平均 token)** —— 训练数据来源 + oracle 选路依据
(评估子集 n≈90，因子集为 oracle 可解，精度偏高)：

| 任务 | Direct | CoT | SQL |
|---|---|---|---|
| hitab | **87%** / 7tok | 84% / 963tok | 83% / 1521tok |
| fetaqa | **94%** / 32tok | 90% / 420tok | 91% / 1344tok |
| tabfact | 69% / 5tok | **97%** / 1138tok | 96% / 1820tok |
| wikitableqa | 76% / 6tok | **92%** / 539tok | 90% / 1124tok |

→ 两个事实同时成立：(1) **成本阶梯极陡** —— Direct 几 token，CoT 几百，SQL 上千 + 工具调用；
(2) **最优路径因任务而异** —— tabfact/wikitableqa 上 CoT/SQL 碾压 Direct (69%→97%)，
hitab/fetaqa 上 Direct 就够。既然便宜路径常常够用，**按题路由能省大量成本** —— cost-aware 命题成立。

## 二、route-SFT 训练：路由塌缩
连训三版 (原始比例 / 下采样 / 改挂式均衡配比)，三版全是学生在推理时**几乎永远只选 Direct**。
设计概率探针，证明：模型在路由 token 上输出的≈训练集的整体先验，**与具体题目无关** (组间差≈0)。
→ 结论：teacher 标签 + 轨迹混训，**学不出逐题路由**。这是当前方法的核心瓶颈。

## 三、执行层是好的：各路径真实执行力 (精度 / 平均 token)
用采样强制走每条路 + **真实 SQL 执行**，测出各路径被走到时的平均能力 (同 oracle 子集，n≈90)：

| 任务 | Direct | CoT | SQL |
|---|---|---|---|
| hitab | **65%** / 21tok | 54% / 761tok | 56% / 78tok |
| fetaqa | **90%** / 46tok | 85% / 463tok | 75% / 96tok |
| tabfact | 52% / 18tok | **85%** / 814tok | 67% / 196tok |
| wikitableqa | 31% / 20tok | 67% / 508tok | **71%** / 55tok |

- 按"每任务选最强路径"做**任务级路由**，估算总精度 ≈ **70%**，远高于塌缩学生的 63.5%。

## 四、蒸馏的价值：路径内能力增益
同样输入下，对比未微调基座的 Direct 与蒸馏后学生的 Direct：

| 任务 | 基座 Direct (零蒸馏) | 学生 Direct (蒸馏后) | 蒸馏增量 |
|---|---|---|---|
| hitab | 43.8% | 62.9% | **+19.1** |
| fetaqa | 70.8% | 89.9% | **+19.1** |
| tabfact | 48.4% | 61.1% | **+12.6** |
| wikitableqa | 31.9% | 40.7% | **+8.8** |

→ 蒸馏在**路径内**实打实加 9~19 个点；cost-aware 路由是**路径间**的选择增益。两个维度正交。

## 五、成本对比：route-SFT vs 最强同输入 baseline (Mixed)
Mixed 是最强的旧 baseline (多轮 SQL agent + CoT 回退，与 route-SFT 同输入)。
对比它和 route-SFT 当前形态 (已塌缩为近似全 Direct)：

| 任务 | Mixed baseline | route-SFT (free) | 成本变化 |
|---|---|---|---|
| hitab | 73% / 615tok / 1.9工具 | 63% / 64tok / 0工具 | **token −90%，零工具**，精度 −10 |
| fetaqa | 85% / 431tok / 2.0工具 | **90%** / 47tok / 0工具 | **token −89%，零工具，精度反超 +5** |
| tabfact | 82% / 816tok / 3.1工具 | 61% / 18tok / 0工具 | token −98%，精度 −21 |
| wikitableqa | 64% / 933tok / 2.1工具 | 41% / 20tok / 0工具 | token −98%，精度 −23 |

→ route-SFT **砍掉 90~98% 的 token、彻底免去工具调用**；fetaqa 精度还反超。
代价是 tabfact/wikitableqa 精度掉得多 —— 因为它俩走 CoT/SQL 的时候accuracy一定会更高，却被塌缩逼成了 Direct。
**这正说明：成本已经压到极低，只要把路由学对 (第二节的瓶颈)，就能用低成本逼近 Mixed 的精度。**

---

## 结论
1. **动机成立**：成本阶梯极陡 (Direct 几 token vs SQL 上千)，且最优路径因任务而异 —— 路由有精度+成本双重价值。
2. **成本已极低**：route-SFT 比最强 baseline 省 90~98% token、零工具调用，fetaqa 精度反超。
3. **执行层 OK，路由层是瓶颈**：各路径都有真本事 (尤其真执行后的 SQL)，但学生学不会逐题选路，拖累了需要贵路径的任务。
4. **蒸馏有效**：同输入下路径内 +9~19pt。

## 下一步
1. **换 student-oracle 标签**：用学生自己的可解性贴标签 (而非 teacher 能力)，标签与题目特征关联更强、更可学。
2. **路由训练方法**：可以结合实验指导中的6.2和6.3给出的DPO和Budget-conditioned方法尝试。
3. 兜底选择：**级联路由** (先 Direct，置信度低再升级)。