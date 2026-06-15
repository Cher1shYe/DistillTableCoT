# Oracle 与各模型结果汇总

> 范围:teacher 的 oracle 上界(oracle accuracy / oracle avg token / oracle path 分布)
> + direct(塌缩后的 route-SFT)与之前几个蒸馏模型的实测结果。
>
> **判分口径**:`utils_train/route_scoring`(与 batch_eval 同口径,hitab/wikitableqa=Exact Match、
> tabfact=Accuracy、fetaqa=ROUGE-L≥0.3 算对)。
> **oracle 定义**:逐题取"最便宜的答对路径"(成本 direct < cot < sql);token = 整条轨迹的
> `completion_tokens`;**分母含三路全错的题**(所以 oracle accuracy 即"完美路由的准确率上界")。

---

## 一、Teacher oracle 上界(选路依据 / 准确率天花板)

### 测试集 (v1, n=100/任务)

| 任务 | oracle accuracy | oracle avg token | oracle path 分布(可解题上) |
|---|---:|---:|---|
| hitab | 87.0% | 244 | direct 89% / cot 6% / sql 6% |
| fetaqa | 92.0% | 59 | direct 95% / cot 3% / sql 2% |
| tabfact | 95.0% | 285 | direct 69% / cot 29% / sql 1% |
| wikitableqa | 91.0% | 152 | direct 76% / cot 23% / sql 1% |

### 训练集 (v2)

| 任务 | n | oracle accuracy | oracle avg token | oracle path 分布(可解题上) |
|---|---:|---:|---:|---|
| hitab | 1500 | 79.7% | 249 | direct 86% / cot 10% / sql 5% |
| fetaqa | 1500 | 91.3% | 169 | direct 88% / cot 7% / sql 5% |
| tabfact | 2500 | 93.6% | 253 | direct 72% / cot 27% / sql 2% |
| wikitableqa | 1000 | 88.8% | 255 | direct 74% / cot 23% / sql 2% |

**要点**:
- oracle avg token 仅 59~285,而单走 CoT 要数百、单走 SQL 上千 —— 完美路由能把成本压到最贵单路的 1/5~1/30。
- 最优路因任务而异:hitab/fetaqa 的 oracle 几乎全是 direct;tabfact/wikitableqa 有 23%~29% 必须上 CoT。
- 剩下 5%~20% 是三路全错的题(no-solution),是任何路由都够不到的天花板。

---

## 二、Direct(塌缩后的 route-SFT)与最强旧基线 —— route_scoring 同口径(测试集,n≈90 oracle 子集)

> **"direct"的来历**:route-SFT 路由塌缩成几乎全选 Direct,所以它 free 模式的输出≈Direct 路径,
> 当时直接拿它当 "direct/学生直答" 用。Mixed 是最强的旧 baseline(多轮 SQL agent + CoT 回退,与
> route-SFT 同输入)。fetaqa 此处为"答对率(ROUGE-L≥0.3)"。

| 任务 | Mixed baseline(最强旧基线) | Direct(=塌缩 route-SFT free) | 成本变化 |
|---|---|---|---|
| hitab | 73% / 615tok / 1.9 工具 | 63% / 64tok / 0 工具 | token −90%、零工具,精度 −10 |
| fetaqa | 85% / 431tok / 2.0 工具 | 90% / 47tok / 0 工具 | token −89%、零工具,精度反超 +5 |
| tabfact | 82% / 816tok / 3.1 工具 | 61% / 18tok / 0 工具 | token −98%,精度 −21 |
| wikitableqa | 64% / 933tok / 2.1 工具 | 41% / 20tok / 0 工具 | token −98%,精度 −23 |

**要点**:Direct(塌缩)把成本砍到极低(省 90~98% token、零工具),fetaqa 精度还反超;
代价是 tabfact/wikitableqa 掉得多 —— 这俩本该走 CoT/SQL,却被塌缩逼成 Direct。

---

## 三、之前几个蒸馏模型(测试集,n=100/任务)

> 旧的分模式蒸馏学生。指标:hitab/wikitableqa=EM、tabfact=Accuracy、fetaqa=ROUGE-L(分数)。

| 模型(变体) | hitab | tabfact | wikitableqa | fetaqa(ROUGE-L) | 输入是否完整 |
|---|---:|---:|---:|---:|---|
| basic | 0.02 ⚠️ | 0.70 | 0.36 | 0.354 | ❌ hitab 86/100 没喂表(数字无效) |
| cot | 0.07 ⚠️ | 0.73 | 0.53 | 0.471 | ❌ hitab 59/100 没喂表(数字无效) |
| agent(=SQL 路) | 0.49 | 0.77 | 0.48 | 0.364 | ✅ 完整 |
| mixed(=路由) | 0.69 | 0.79 | 0.59 | 0.460 | ✅ 完整 |
| *teacher cot(参考)* | 0.74 | 0.89 | 0.82 | 0.519 | — |
| *teacher mixed(参考)* | — | 0.89 | 0.83 | 0.487 | — |

**注意**:
- `basic` 名为 basic 但 100% 仍带 `<think>`,**不是真正的 direct(直接作答)模型**;且 hitab 输入残缺。
- `cot`/`basic` 在 hitab 上表没喂进去(预测里写 "the table isn't here"),hitab 的 0.02/0.07 **不是真实能力**,
  公平对比只能用 `agent`/`mixed`(输入完整)。

---

## 四、学生(小模型)oracle —— 对同一题挑最便宜答对路(测试集子集 n≈90)

> **三路来源**(按 (task,id) 取交集,以 route_eval 的 ~90 子集=teacher 可解子集为准):
> direct = 塌缩 route-SFT free 输出(~98% 走 direct,有 `correct`/`output_tokens`);
> cot = `predictions_qwen3_1.7b_cot`;sql = `predictions_qwen3_1.7b_agent`(多轮真执行)。
> 判分全程 route_scoring;token 取模型生成口径(agent=Σ各轮 response 的 token)。

| 任务 | n | oracle accuracy | oracle avg token | oracle path 分布(可解题上) |
|---|---:|---:|---:|---|
| hitab | 89 | 83.1% | 178 | direct 76% / cot 8% / sql 16% |
| fetaqa | 89 | 98.9% | 77 | direct 91% / cot 7% / sql 2% |
| tabfact | 95 | 91.6% | 122 | direct 67% / cot 30% / sql 3% |
| wikitableqa | 91 | 74.7% | 127 | direct 54% / cot 37% / sql 9% |

**各单路 学生 acc / avg token**:

| 任务 | direct | cot | sql |
|---|---|---|---|
| hitab | 63% / 64 | 8% / 394 ⚠️ | 52% / 726 |
| fetaqa | 90% / 47 | 88% / 305 | 69% / 618 |
| tabfact | 61% / 18 | 74% / 276 | 78% / 1032 |
| wikitableqa | 41% / 20 | 58% / 325 | 52% / 1025 |

### 路由头部空间(GRPO 的目标值)

direct 单路(=现在塌缩模型 free 模式的实际表现)对照学生 oracle(完美路由天花板),
两者之差 = **改好路由能夺回多少精度**:

| 任务 | 现状(塌缩≈全 direct) | 学生 oracle | 路由可夺回 |
|---|---:|---:|---:|
| hitab | 63% | 83.1% | **+20pt** |
| fetaqa | 90% | 98.9% | **+9pt** |
| tabfact | 61% | 91.6% | **+31pt** |
| wikitableqa | 41% | 74.7% | **+34pt** |

→ 达到天花板的成本极低(oracle avg token 仅 77~178,远低于单走 sql 的 600~1000)。
**这是"执行层 OK、路由是瓶颈"的量化证据,也是 GRPO 的明确目标。**

**口径说明**:
- 子集为 teacher 可解的 ~90 题/任务(因 direct 数据只在该子集上有),故 teacher 在此子集上 oracle≈100%;
  学生 oracle < 100% 反映的是"在 teacher 能解的题里,学生用最优路能解多少"。
- direct 来自塌缩 route-SFT(并非单独训的纯 direct 模型);hitab 的 cot(8%)受输入残缺影响不可靠,
  但 oracle 取各路最好者,受单路损坏影响很小(只在"该题仅 cot 可解"时才丢分)。
- 真正干净的"同一模型三条强制路"的学生 oracle,需 cost-aware 路由模型训完后用 `--force_route` 跑出来。
