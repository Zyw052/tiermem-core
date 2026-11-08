<div align="center">

# 🧠 TierMem Core

### 给你的 AI Agent 装一套「不会失忆、也不会乱编」的大脑

**极简 · 零容器 · 防幻觉 · 纯 Python 的分层长期记忆内核**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![SQLite](https://img.shields.io/badge/Storage-SQLite%20WAL-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Docker](https://img.shields.io/badge/Docker-0%20containers-2496ED?logo=docker&logoColor=white)](#-为什么要有-tiermem-core)
[![Footprint](https://img.shields.io/badge/RAM-~30MB-31c48d)](#-架构对比轻量到什么程度)
[![Tests](https://img.shields.io/badge/Tests-63%20passed-2ea44f)](#-测试与质量保障)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

</div>

<a id="nav"></a>

## 📖 目录

| 章节 | 你会看到什么 |
| :-- | :-- |
| [为什么要有 TierMem Core](#why) | 为什么你的机器人像个金鱼 |
| [一句话定位](#pitch) | 这玩意儿到底是干嘛的 |
| [三层记忆](#tiers) | 用「人的大脑」讲清 L1 / L2 / L3 |
| [源流与致敬](#origin) | 我们继承了谁、又砍掉了什么 |
| [生产级超越](#plan-a) | 方案 A 的五道硬核防线 |
| [工业级引擎升级](#plan-b) | 方案 B：FTS5 倒排 + 向量矩阵缓存 + 死信闭环 |
| [30 秒快速上手](#quickstart) | 复制粘贴就能跑 |
| [架构对比](#compare) | 轻量化到底轻到什么程度 |
| [配置说明](#config) | 兼容任意模型 / 本地 Embedding |
| [测试与质量](#tests) | 63 个用例锁死的行为契约 |
| [开源演进路线图](#roadmap) | 已落地阶段 + 方案 C 要做什么 |
| [English TL;DR](#english) | 英文速览 |

<a id="english"></a>

## 🌐 English TL;DR

**TierMem Core** is a lightweight, zero-container, anti-hallucination **tiered long-term memory kernel** for AI agents, written in pure Python.

- **Inspired by** Tencent's open-source `TencentDB-Agent-Memory` (30k+ stars): we keep its two best ideas — the **L0~L3 tiered memory model** and **BM25 + vector hybrid retrieval**.
- **Rebuilt for humans**: the original needs 4 Docker containers, Node.js 22 and a dedicated vector database (multi-GB). Ours is **one SQLite file, ~30 MB RAM, zero infrastructure**.
- **Production hardened (Plan A)**: pinned memories can never be killed by a confused LLM, 5 dialogue turns collapse into a single batched LLM + embedding call (**≈80% cheaper**), every turn is persisted immediately, sessions are strictly isolated, and retrieval is lock-free read-only.
- **Plan B landed (industrial engine)**: native SQLite `FTS5` inverted index with CJK pre-tokenization (≈13x faster keyword recall), an in-memory **vector matrix cache** (`matrix @ query`, microsecond vector scoring, auto-fallback to pure Python), and a **closed-loop dead-letter queue** with exponential backoff retries.
- **Roadmap**: Plan C — a true temporal knowledge graph.

> 中文读者请继续往下读，下面全部是中文主线，穿插英文术语便于检索。

---

<a id="why"></a>

## 🤔 为什么要有 TierMem Core

### 你的 AI 有点像个金鱼

大模型没有「记忆」这个器官。每一次 API 请求，对它来说都是**一次全新的投胎**。

你昨天跟它聊了两小时，今天它开口第一句是「你好，我是 AI 助手，有什么可以帮你？」

这不是它笨，是它**真的不记得**。

### 开发者的四种「土办法」，都不太灵

| 土办法 | 听起来还行 | 实际踩的坑 |
| :-- | :-- | :-- |
| **把历史全塞进 Prompt** | 最省事 | Token 费爆炸；上下文越长模型越糊涂（*lost in the middle*）；迟早撞上窗口上限 |
| **自己写个日志文件** | 数据留下了 | **存得下 ≠ 找得出**。1000 条日志里翻不出「主人不吃香菜」 |
| **直接上向量数据库** | 语义检索很香 | 「主人」和「主人翁」向量很近，精确匹配垮掉；且为一句话装一整套服务 |
| **上重型分层记忆方案** | 功能确实强 | 为了记住「主人爱喝龙井」，你先得部署 4 个 Docker 容器 |

### 具体到日常，是这四种翻车

- 🐟 **重复自我介绍** —— 每换个会话就要重新交代一遍「我是谁、说话什么风格」；
- 🙈 **推翻自己** —— 昨天说了「我讨厌香菜」，今天推荐菜还是「香菜拌牛肉」；
- 💥 **记错铁律** —— 你三令五申「叫我主人」，某天它突然改口喊你「亲」；
- 🚨 **串台事故** —— A 群的聊天内容，被 B 群的人问出来了。

TierMem Core 就是为了把这四件事一次性解决。

---

<a id="pitch"></a>

## 🎯 一句话定位

> **tiermem-core 是一个极简、零容器、防幻觉的 Python Agent 分层记忆内核：**
> **用「一个 SQLite 文件」管好「短期流水 → 中期事实 → 长期铁律」三层记忆，**
> **检索时只把最该被记得的那几条，精准塞回 Prompt。**

三个关键词，展开说：

- **极简** —— 没有 Redis、没有 Milvus、没有 Neo4j。整个"数据库"就是一个 `.db` 文件，`cp` 一下就是备份。
- **零容器** —— `pip` 装完就能跑，不需要 `docker-compose up`，不需要 Node.js，不需要运维。
- **防幻觉** —— 记忆是**可追溯、可推翻、可置顶**的。它会告诉你"这条记忆从哪来"，也会在你说"我不喜欢了"之后优雅地废弃旧记忆，而不是把两条矛盾事实一起塞给模型让它自己精神分裂。

---

<a id="tiers"></a>

## 🧠 三层记忆：人的大脑是怎么记账的

先别管什么 L0~L3、什么 embedding 维度。**人的记忆本来就是分层的**，这个项目只是把大脑的做法搬进了代码。

| 层级 | 🧠 大脑里的对应物 | 存什么 | 举例 | 特点 |
| :-- | :-- | :-- | :-- | :-- |
| **L1 工作记忆** | **桌边的草稿纸 / 流水账本** | 原始对话逐字全量保存 | 「周三 21:14 · 主人：今天加班好累」 | 量大、不做加工、**绝不会丢**（进门即落盘） |
| **L2 情景记忆** | **随身笔记本** | 从流水里提炼出的**事实原子** | 「主人最近在赶一个项目，压力大」 | 精炼、带主题标签、带向量、可被推翻 |
| **L3 核心记忆** | **刻在墙上的座右铭 / 铁律** | 必须长期遵守的偏好与硬约束 | 「称呼必须是"主人"」「严禁提及密码」 | 少量、**强力置顶**、**免死金牌** |

### 记忆是怎么"升级"的

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  用户说的一句话                                                  │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────┐   每轮实时落盘，断电不丢                            │
│  │  L1 流水 │ ──────────────────────────────►  SQLite: l1_docs   │
│  └────┬────┘                                                    │
│       │  攒够 N 轮（默认 5 轮）→ 整窗打包成 1 次调用              │
│       ▼                                                         │
│  ┌──────────────┐   LLM 提炼出独立事实                          │
│  │  事实抽取     │   + 廉价判官打分/查冲突                        │
│  └──────┬───────┘                                               │
│         ▼                                                       │
│  ┌─────────┐   普通事实                                          │
│  │  L2 原子 │ ──────────────────────────────►  SQLite: l2_atoms  │
│  └────┬────┘                                                    │
│       │  命中「喜欢/讨厌/必须/严禁/密码/习惯」                      │
│       │  或重要度 ≥ 0.85                                        │
│       ▼                                                         │
│  ┌─────────┐                                                   │
│  │  L3 铁律 │  pinned=1 · tier=3 ────────────►  SQLite: l2_atoms │
│  └─────────┘  （检索时永远排在最前面，且不可能被覆盖）             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

> **为什么非要分层？**
> 因为**检索时的优先级不一样**。L3 是「无论如何都得遵守」，L2 是「这条可能相关，看看」，L1 是「要溯源/要复述原话时才翻」。三层分开放，检索时才能各按各的权重处理；全塞一张大表里，就只能一刀切。

### L3 是怎么自动晋升的

命中以下关键词，或模型给出的重要度 ≥ `0.85`，就自动钉到墙上，**不需要你手动维护**：

```text
喜欢 · 讨厌 · 偏好 · 必须 · 严禁 · 密码 · 习惯
```

命中即 `importance = 0.90`、`pinned = 1`、`tier = 3`。

也可以随时手动干预：

```bash
/memory pin 42     # 把第 42 条记忆钉上墙（或取消钉）
/memory recall 龙井 # 手动试一下"关于龙井，它记得什么"
/memory status     # 看一眼现在脑子里装了多少东西
```

---

<a id="origin"></a>

## 🌱 源流与致敬：站在巨人的肩膀上

这个项目不是凭空想出来的。它有两个明确的老师。

### 老师一：腾讯开源的 `TencentDB-Agent-Memory`

腾讯开源的 `TencentDB-Agent-Memory`（GitHub **30,000+ Star**）是目前业界做得最扎实的 Agent 记忆工程之一。我们从中**认真抄了两样真本事**：

| 学到的精髓 | 为什么重要 |
| :-- | :-- |
| **L0~L3 分层记忆模型** | 记忆不是一堆平铺的向量，而是有层级的：原始记录 / 提炼事实 / 核心画像。分层决定了检索优先级、决定了什么能丢、什么不能丢。 |
| **BM25 + 向量双路混合检索** | 两条腿走路：**BM25 保精确**（关键词、人名、专有名词一个都不能漏），**向量保语义**（"我不爱吃香菜"≈"香菜我不行"）。任何单路检索都会瘸腿。 |

**没有这两位老师，这个项目不会存在。** 如果你要做的是超大规模、多租户、企业级的记忆中台，请直接用原版。

### 但原版对我们来说太重了

原版的定位是「工业级记忆基础设施」，代价是：

```text
4 个 Docker 容器 + Node.js 22 运行时 + 专用向量数据库 …… 镜像动辄数 GB
```

如果你只是想给自己的 QQ 机器人、Telegram Bot、或者一个本地小助手加个「记得住事」的能力，
为了这个目标去维护一整套分布式记忆服务，**性价比是负的**。

### 所以我们做了一次「极致轻量化降维重构」

一句话概括：**留下架构精髓，扔掉部署负担。**

| 维度 | 原版重型记忆系统 | 🟢 TierMem Core |
| :-- | :-- | :-- |
| **部署形态** | 4 个 Docker 容器 | **0 个容器**，一个 Python 进程 |
| **运行时** | Node.js 22 | **Python 3.10+** |
| **存储** | 专用向量数据库（独立服务、独立运维） | **单个 SQLite 文件**（WAL 模式，`cp` 即备份） |
| **硬依赖** | 一整套服务栈 | **纯标准库**（`sqlite3` 内置）；`aiohttp` / `numpy` 均为**可选加速项**，缺失自动降级 |
| **内存占用** | 数 GB 级 | **常驻约 30 MB** |
| **上手成本** | 读半天部署文档 | **3 行配置**跑起来 |
| **断网/无密钥** | 基本不可用 | **全降级链路**：规则抽取 + 关键词检索 + 本地 RRF 重排，照常工作 |
| **检索能力** | BM25 + 向量 + 精排 | **向量 + 关键词双路粗排**，`BGE-Reranker` 精排；无密钥时降级本地 RRF 融合 |

> 💡 **降维不是阉割。** 我们把「分布式服务」换成了「单文件数据库」，把「在线向量库」换成了「存成 BLOB 的向量 + 内存余弦计算」——对个人/小团队场景（万级记忆、单机部署），**效果没有肉眼可见的差别，运维成本趋近于零**。

---

<a id="plan-a"></a>

## 🛡️ 生产级超越：方案 A 的五道硬核防线

轻量化最容易死的姿势是「能跑，但一上线就翻车」。为此我们做了一轮**止血热修（方案 A）**，
专门解决分层记忆系统在真实环境里的五个经典事故。

每一条都不是"理论上应该"，而是**有专项测试锁死**的（`tests/test_stage6_hardening.py`，7 个用例，编号 R-01 ~ R-06）。

### 🛡️ 1. L3 置顶免死金牌 —— 绝不因模型抽风误杀铁律

**事故现场**：你设了「称呼必须是主人」。某天模型在判官环节抽风，或者用户开了句玩笑说「别叫我主人了」，
于是这条核心设定的旧记录被标记为「已推翻」——**铁律当场蒸发**，机器人从此喊你「您好」。

**我们的做法**：

- 冲突检测的**规则层直接跳过** `pinned=1` 的候选，从源头不给它被推翻的机会；
- 数据库层的软废弃 SQL 追加硬守卫 `AND COALESCE(pinned,0) = 0`，**双保险**；
- 结论：`pinned=1` 的记忆**只能被手动 `unpin`**，任何自动流程都杀不掉它。

```sql
-- 软废弃时的硬守卫（R-01）
UPDATE l2_atoms
   SET status = 'superseded'
 WHERE id = ? AND COALESCE(pinned, 0) = 0   -- ← 置顶记录在此被物理排除
```

### ⚡ 2. 真·多轮合并批处理 —— API 账单立省 80%

**事故现场**：朴素实现是「每轮对话 → 1 次事实抽取 + 1 次门禁判定 + 1 次向量化」。
5 轮对话就是 **15 次 API 调用**。这叫 **N+1 账单风暴**，跑一天就能给你账单干出心理阴影。

**我们的做法**：把 N 轮**合并进同一个 Context Window**，一次做完：

```text
朴素实现（5 轮）：          本实现（5 轮）：
  轮1 → LLM + 门禁 + Embed     ┌──────────────────────────┐
  轮2 → LLM + 门禁 + Embed     │  5 轮打包成 1 个窗口      │
  轮3 → LLM + 门禁 + Embed  →  │  ① 1 次事实抽取           │
  轮4 → LLM + 门禁 + Embed     │  ② 1 次门禁判定           │
  轮5 → LLM + 门禁 + Embed     │  ③ 1 次批量向量化         │
  ─────────────────────       └──────────────────────────┘
  合计 15 次调用                 合计 3 次调用 → 省 80%
```

**为什么质量不降反升**：模型看到的是一整段连贯对话，而不是 5 个被截断的碎片，**提炼事实的上下文更完整**。

### 💾 3. L1 实时落盘 —— 断电、重启、崩溃，零丢失

**事故现场**：为了"性能"，实现里往往先把对话攒在内存缓冲池，攒够 5 轮再写库。
结果——**没攒够的 4 轮，在重启的那一刻全没了**。用户觉得这机器人有阿尔茨海默症。

**我们的做法**：

- `enqueue_dialogue()` **进门即写 L1**，不攒、不猜、不等；
- L2 析出后回写 `l2_processed` 标记，避免重复提炼；
- 额外加一个 **idle 超时刷盘**（默认 180s），长时间没人说话的会话尾巴也不会滞留内存。

> 一句话：**提炼可以慢，落盘必须快。**

### 🔒 4. 严格会话隔离 —— 杜绝隐私串台

**事故现场**：A 群聊了体检报告，B 群的人随便问一句，记忆就被检索出来了。这是**事故级**的问题。

**我们的做法**：

- `strict_session_isolation` **默认开启**（默认值就是 `True`，不需要你记得去打开）；
- 检索时只允许召回 **本会话** + `scope_key = 'global'`（显式标记为全局共享的）的记忆；
- 跨群、跨私聊**物理隔离**，不是靠 Prompt 里写一句"请不要提及其他群的内容"来祈祷。

### 🚀 5. 连接池复用 + 纯读检索 —— 告别 SQLite 事务打架

**两个独立的性能坑，一起修掉：**

| 坑 | 症状 | 修法 |
| :-- | :-- | :-- |
| **TLS 握手风暴** | 每次 Embedding/Rerank/判官调用都新建 `ClientSession`，高频下全是 TCP/TLS 握手开销 | 全程复用**持久 `aiohttp.ClientSession` 连接池**，`close()` 幂等释放 |
| **SQLite 写锁争抢** | 检索时顺手更新"访问计数"，读操作变成写操作 → 和总结 Worker 抢写锁 → `database is locked` | 检索链路**纯读**，访问计数在内存累加、**异步定期刷盘**，`close()` 时兜底刷净 |

> 结果：**检索零阻塞**，对话不会因为"正在记东西"而卡住。

---

<a id="plan-b"></a>

## 🚀 工业级引擎升级：方案 B 的三大核心改造

方案 A 解决了「不翻车」，方案 B 解决「跑得够快」。三项改造全部**零外部依赖、向下兼容、失败自动降级**，
专项测试锁死在 `tests/test_stage7_engine_b.py`（22 个用例）。

### ⚡ 1. SQLite 原生 FTS5 全文索引 —— 毫秒级倒排召回

**事故现场**：朴素的关键词召回是「把全部记忆拉进 Python，用正则逐条切词，再跑双重循环算 BM25」。
千级记忆时单次检索要 **35ms+ 的纯 CPU 开销**，而且完全无法利用数据库能力。

**我们的做法**：

- 启动即探测 SQLite 是否编入 FTS5（实测创建临时虚拟表，比解析 `PRAGMA compile_options` 更可靠）；
- 创建虚拟表 `l2_atoms_fts(atom_id UNINDEXED, content, key_facts)`，`tokenize='unicode61 remove_diacritics 2'`；
- **关键技术点**：`unicode61` 不会把中文切成单字（整段 CJK 视为一个 token），`MATCH '咖啡'` 命中不了「美式咖啡」。
  因此索引侧与查询侧统一做 **CJK 逐字预分词**（`storage.fts_preprocess`），写入 `l2_atoms.fts_text`：

```python
>>> fts_preprocess("主人爱喝美式咖啡 tiermem-core")
'tiermem core 主 人 爱 喝 美 式 咖 啡'
```

- **四类自动同步触发器**覆盖 `INSERT / UPDATE / DELETE / status 软废弃`，索引与主表永不脱节：

```sql
CREATE TRIGGER trg_l2_atoms_fts_ai AFTER INSERT ON l2_atoms BEGIN
  INSERT INTO l2_atoms_fts(atom_id, content, key_facts)
  VALUES (new.atom_id,
          CASE WHEN COALESCE(new.fts_text,'') <> '' THEN new.fts_text ELSE new.content END,
          CASE WHEN COALESCE(new.fts_text,'') <> '' THEN new.fts_text ELSE new.key_facts_json END);
END;
```

- 检索直接 `MATCH` + `bm25(l2_atoms_fts, 1.0, 0.5)` 倒排打分，分数归一化到 0~1 参与融合；
- **软废弃即时摘除**：`status='superseded'` 时触发器立刻从倒排表删除，检索期无需 `JOIN` 过滤，恢复 `active` 自动回灌；
- **零破坏回退**：SQLite 无 FTS5 时自动回退原 Python BM25，行为完全不变。

| 千级记忆单次检索 | 耗时 | 相对加速 |
| :-- | --: | --: |
| Python BM25（原实现） | 35.1 ms | 1.0x |
| **FTS5 + bm25（方案 B）** | **2.7 ms** | **~13x** |

### 🧮 2. 内存向量矩阵缓存 —— 从「逐条反序列化」到「一次矩阵乘法」

**事故现场**：每次检索都要 `SELECT *` 全表拉取，对每行 `struct.unpack` 反序列化浮点向量，
再逐条调用 `cosine_similarity`。千级记忆下这是实打实的 CPU 瓶颈，且 99% 的向量根本没变化。

**我们的做法**：`EmbeddingMatrixCache`

- 维护 `atom_id -> index` 映射 + 二维矩阵 `np.ndarray(n, dim)`（**存单位向量**，余弦相似度退化为纯点积）;
- 启动/首次检索一次性构建；检索时 `matrix @ query` **一次算出全部相似度**（微秒级）；
- 写入侧**增量维护**：新原子 `upsert` 均摊 O(1) 追加；融合更新原位覆盖；软废弃 `invalidate` 置掩码失效，
  **不搬移数据、不重建矩阵**；
- 候选池因此可以只 `SELECT` 必要列（跳过 `embedding_blob`），进入精排前再从缓存按需回填向量；
- 维度变更（换 embedding 模型）按**维度众数**重建，旧维度脏数据安全跳过；
- 5 秒一次的轻量指纹校验（`COUNT` + `MAX(updated_at)`）兜住绕过缓存的直接 SQL 写入；
- **无 numpy 自动回退**纯 Python 单遍点积，数值与 numpy 后端误差 < 1e-5。

```text
批量余弦一致性：max_diff = 3.1e-08（float32 精度极限）
```

### 🔁 3. 死信队列闭环 —— 从「只写不读的黑洞」到「自动重试消费」

**事故现场**：方案 A 把失败任务写进 `task_dlq` 后就再没人看过它——**死信永久埋葬**，
本该析出的记忆永远丢失，直到有人手工翻库。

**我们的做法**：后台守护协程 `_dlq_retry_loop` + `sweep_dlq_once()`

- 定期扫描 `status='pending' AND next_retry_at <= now AND retry_count < max_retries`；
- 重新投递并调用原处理链路消费；成功 → `completed`；
- 失败 → **指数退避** `base * 2^(n-1)`（受 `max_backoff` 封顶）后重新排队；
- 重试耗尽 → `failed` 终态，不再消耗资源；
- **防死信爆炸**：重试路径以 `raise_on_error=True` 上抛异常，由重试器统一改状态，
  绝不会「重试一次就多写一条新死信」；
- 脏 payload / 未知 `task_type` / 超限历史脏数据一律收敛为 `failed`，绝不卡死队列。

```text
[OK] 指数退避生效: 第1次≈10.0s，第2次≈20.0s
[OK] DLQ 端到端闭环：失败落库 → 到期重试 → completed，析出记忆=...，L1 无重复(1)
```

---

<a id="quickstart"></a>

## 🚀 30 秒快速上手

### 1. 拿到代码

```bash
git clone https://github.com/Zyw052/tiermem-core.git
cd tiermem-core
```

### 2. 零硬依赖，直接验证

```bash
# 不需要 pip install 任何东西就能跑完整个测试套件
python3 tests/test_stage4_e2e.py
```

看到 `[闭环 OK] 冷重启持久化验证通过` 就说明：**记忆真的写进磁盘了，重启还在。**

### 3. 可选：装上加速依赖

```bash
pip install aiohttp    # 仅在调用远程 Embedding / Rerank / LLM 时需要
pip install numpy      # 可选，向量运算加速（缺失时自动回退纯 Python 实现）
```

### 4. 最小可用配置

**只需要 3 行**，而且这三行全是默认值——也就是说，**什么都不配也能跑**：

```jsonc
{
  "enable_memory": true,
  "summary_turn_interval": 5,
  "retrieval": { "inject_position": "system_head" }
}
```

此状态下走**全降级链路**（不需要任何模型密钥）：

```text
规则分句抽取事实 → 关键词（BM25 + 词重合）粗排 → 本地 RRF 精排 → 注入 Prompt
```

### 5. 接上真正的模型（可选，但强烈建议）

```jsonc
{
  "models": {
    "embedding": {
      "api_base": "https://api.siliconflow.cn/v1",
      "api_key": "sk-xxx",
      "model": "BAAI/bge-m3",
      "dim": 1024
    },
    "rerank": {
      "api_base": "https://api.siliconflow.cn/v1",
      "api_key": "sk-xxx",
      "model": "BAAI/bge-reranker-v2-m3"
    }
  }
}
```

### 6. 作为 AstrBot 插件使用

把整个目录丢进 AstrBot 的插件路径，重启，开聊：

```bash
/AstrBot/data/plugins/astrbot_plugin_agent_memory/
```

然后试试管理指令：

```bash
/memory status                  # 看看脑子里装了什么
/memory recall 咖啡             # 手动检索
/memory pin 42                  # 把某条记忆钉成铁律
/memory migrate                 # 从旧版 livingmemory 无损搬家
```

---

<a id="compare"></a>

## 📊 架构对比：轻量到什么程度

### 存储与部署

| 维度 | 🐳 原生重型内存系统 | 🟢 TierMem Core |
| :-- | :-- | :-- |
| 容器数量 | 4+ | **0** |
| 数据库 | 专用向量数据库（独立进程） | **1 个 SQLite 文件** |
| 运行时 | Node.js 22 | **Python 3.10+** |
| 磁盘占用 | 数 GB（镜像 + 模型） | **MB 级**（只有你的对话数据） |
| 内存占用 | GB 级 | **~30 MB** |
| 备份方式 | 快照 / 导出工具 | **`cp agent_memory.db backup.db`** |
| 离线可用 | 否 | **是**（全降级链路） |

### 能力对比（和"常见做法"正面碰一碰）

| 能力 | 全塞上下文 | 纯向量库 | 重型分层方案 | 🟢 **TierMem Core** |
| :-- | :--: | :--: | :--: | :--: |
| 长期记忆 | ⚠️ 撞窗口上限 | ✅ | ✅ | ✅ |
| 精确匹配（人名/术语） | ⚠️ 靠模型运气 | ❌ 向量近义混淆 | ✅ BM25 | ✅ **BM25 + 词重合双路** |
| 语义匹配 | ✅ | ✅ | ✅ | ✅ **向量余弦** |
| 记忆冲突处理 | ❌ | ❌ | ⚠️ 部分 | ✅ **软废弃 + `superseded` 可追溯** |
| 噪声过滤 | ❌ | ❌ | ⚠️ 部分 | ✅ **廉价判官前置门禁** |
| 铁律不可被推翻 | ❌ | ❌ | ❌ | ✅ **L3 pinned 免死金牌** |
| 会话隔离 | ⚠️ 要自己写 | ⚠️ 要自己写 | ⚠️ 要自己写 | ✅ **默认开启，物理隔离** |
| 部署成本 | ✅ 零 | ⚠️ 要起服务 | ❌ 4 个容器起 | ✅ **零容器** |
| 数据隐私 | ✅ 本地 | ⚠️ 看部署 | ⚠️ 看部署 | ✅ **单文件本地 SQLite** |

### 数据流全景

```text
                     ┌──────────────── 对话链路 ────────────────┐
                     │                                          │
   用户消息 ──► on_llm_request ──► [ 检索 + 注入 Prompt ] ──► LLM ──► on_llm_response
                     │            ▲                                    │
                     │            │                                    ▼
                     │     MemoryRetriever                    多轮缓冲池（内存）
                     │      ① L3 置顶捞取                              │ 满 N 轮打包
                     │      ② 向量 + 关键词双路粗排                    ▼
                     │      ③ Cross-Encoder 精排              MemorySummarizer
                     │      ④ 组装 Markdown 记忆块                  │  ├─ 事实抽取
                     │            ▲                                │  ├─ 判官门禁
                     │            │                                │  ├─ 批量向量化
                     │     l2_atoms (L2 + L3)  ◄───────────────────┤  └─ L3 晋升
                     │            ▲                                │
                     │            └── l1_docs (L1 实时落盘)  ◄─────┤
                     │                                             └─► task_dlq（失败死信）
                     └────────────────────────────────────────────────────┘
```

### 模块地图

| 文件 | 一句话职责 |
| :-- | :-- |
| `main.py` | 主类装配：配置规范化、组件总装、生命周期托管、双向钩子、`/memory` 指令层 |
| `storage.py` | SQLite 引擎：DDL、WAL 连接管理、分层统计、`toggle_atom_pin()` 置顶切换 |
| `summarizer.py` | 多轮缓冲池 + 异步 Worker：事实抽取、门禁接入、向量去重融合、L3 晋升、DLQ 降级 |
| `evaluator.py` | **判官门禁**：信息价值打分、事实冲突/推翻检测、超时与异常平滑降级 |
| `retriever.py` | 两阶段混合检索：L3 置顶、粗排召回、精排重排、Markdown 上下文组装 |
| `models_client.py` | BGE-M3 向量客户端 & BGE-Reranker 客户端，含维度对齐校验与 RRF 降级 |
| `migrator.py` | 旧版 `livingmemory` 资产无损迁移（dry-run / 备份快照 / 孤儿边跳过） |
| `tests/` | Stage 1~6 全量测试套件 |

---

<a id="config"></a>

## ⚙️ 配置说明

### 兼容各大模型：只要 OpenAI 兼容协议

`api_base` + `api_key` + `model` 三件套，**任何 OpenAI 兼容端点都能接**：

| 场景 | 推荐配置 |
| :-- | :-- |
| **硅基流动**（性价比首选） | `api_base: https://api.siliconflow.cn/v1` · `model: BAAI/bge-m3` |
| **DeepSeek / Qwen**（总结用） | 填官方兼容端点，`model: deepseek-chat` / `qwen-plus` |
| **OpenAI** | `api_base: https://api.openai.com/v1` · `model: text-embedding-3-small` |
| **本地 Embedding**（Ollama / Xinference / vLLM） | `api_base: http://127.0.0.1:11434/v1` · `model: bge-m3` · `api_key` 随便填 |
| **本地判官模型** | `evaluator.model_name` 指向本地小模型（默认 `jev`），成本几乎为零 |
| **完全离线** | 什么都不配 → 自动走规则抽取 + 关键词检索 + 本地 RRF |

> 若你的宿主（如 AstrBot）已经配置过 Provider，可直接填 `summarizer.provider_id` /
> `summarizer.embedding_provider_id` / `summarizer.rerank_provider_id`，
> 插件会自动复用其中的 `api_base` / `api_key` / `model`，**不用重复填密钥**。

### 三态注入：记忆塞在哪里，决定模型听不听话

| 取值 | 写入位置 | 推荐模型 | 适用场景 |
| :-- | :-- | :-- | :-- |
| **`system_head`**（默认） | System Prompt **最前端** | Claude / GPT-4o | 记忆即最高优先级上下文，**指令权重最高** |
| **`system_tail`** | System Prompt **最末端** | 通用 | 人设优先，记忆作为补充，避免稀释人格 |
| **`user_head`** | 用户 Query **最前端** | DeepSeek / Qwen | 对即时提问敏感的模型，记忆与问题紧邻 |

> **调优口诀**：人格被记忆稀释 → 改 `system_tail`；模型忽略记忆 → 改 `system_head`；上下文被挤爆 → 下调 `prompt_injection_max_chars`。

### 关键参数速查

| 配置项 | 默认值 | 说明 |
| :-- | :-- | :-- |
| `enable_memory` | `true` | 记忆总开关 |
| `db_path` | `''` | 留空 → 插件数据目录；填目录 → 目录下 `agent_memory.db`；填文件 → 直接用 |
| `summary_turn_interval` | `5` | 多轮缓冲打包阈值（**这个值直接决定 API 账单**） |
| `summary_idle_flush_seconds` | `180` | 静默超时自动刷盘（0 关闭） |
| `retrieval.top_k` | `6` | 单次注入的记忆条数上限 |
| `retrieval.prompt_injection_max_chars` | `2000` | 记忆块字符上限，0 = 不限制 |
| `retrieval.retrieve_timeout` | `3.0` | 检索超时（秒），超时静默放弃本次注入 |
| `retrieval.strict_session_isolation` | `true` | **严格会话隔离**，仅本会话 + `global` 可召回 |
| `evaluator.enable` | `true` | 是否启用入库前判官门禁 |
| `evaluator.min_value_score` | `0.35` | 信息价值过滤阈值，低于即丢弃 |
| `evaluator.fail_open` | `true` | 判官挂掉时是否保留事实（**宁多记，不丢记忆**） |

### 注入效果长什么样

```markdown
<!-- agent_memory:injected:v1 -->
## 🧠 长期记忆上下文 (Agent Memory Context)
### 📌 核心记忆与约束 (L3 Pinned)
1. **[核心/preference]** 主人喜欢喝龙井，不加糖。 *(重要度: 0.90)*
2. **[核心/preference]** 主人必须每天早上喝一杯温水。 *(重要度: 0.90)*
### 🔍 相关情景记忆 (L2 Retrieved)
- 上周我们讨论了西湖龙井的采摘季节。 *(相关度: 0.7341)*
```

> 隐藏标记 `<!-- agent_memory:injected:v1 -->` 保证**同一请求绝不重复注入**（包括工具调用的多轮往返）。

---

<a id="tests"></a>

## 🧪 测试与质量保障

**7 个套件 · 63 个用例 · 全部可脱离宿主独立运行**（内置宿主桩包，零外部依赖）。

```bash
python3 tests/test_stage1_buffer.py          # 多轮缓冲打包 / 强制冲洗 / Worker 消费
python3 tests/test_stage2_hooks.py           # 双向钩子装配 / 注入 / 记录 / 降级 / 格式化
python3 tests/test_stage3_commands.py        # 三态注入路由 / 四指令 / 指令组装配
python3 tests/test_stage4_e2e.py             # 全链路端到端闭环 + 冷重启持久化
python3 tests/test_stage5_evaluator_gate.py  # 判官门禁：价值打分 / 冲突废弃 / 超时降级
python3 tests/test_stage6_hardening.py       # 方案 A 防线：R-01 ~ R-06 六项专项
python3 tests/test_stage7_engine_b.py        # 方案 B 引擎：FTS5 / 向量矩阵 / DLQ 闭环 22 项专项
```

> 全量回归同时兼容 **numpy 存在 / 缺失** 两种运行环境（矩阵缓存自动切换后端），
> 以及 **SQLite 带 / 不带 FTS5** 两种构建（全文检索自动回退 Python BM25）。

### 实测闭环输出（Stage 4）

```text
[① OK] 插件初始化完成 | turn_interval=5 | worker=agent_memory_summarizer_worker
[② OK] 5 轮对话已缓冲并触发打包 | batches=1 | buffered=5
[③ OK] Worker 消费完成 | L1=5 · L2=5（其中 L3 置顶 3） · DLQ=0
[④ OK] 前置注入成功 | position=system_head | 记忆块 345 字符 | inject_ok=1 / inject_error=0
[⑤-1 OK] /memory status 输出正常（体量与数据库一致）
[⑤-2 OK] /memory recall 输出正常（含 L3 置顶 + L2 精排分数）
[⑤-3 OK] /memory pin 状态切换可逆 | L3 总数 3→4→3
[⑥ OK] 优雅停机完成 | 尾部冲洗 2 轮 | L1 7→7（实时落盘） · L2 5→7 · DLQ=0
[闭环 OK] 冷重启持久化验证通过 | 复用 5 条 L2 记忆
[闭环 OK] 二次停机后资源同样完全释放
```

### 设计原则：永不阻断主链路

```text
任何记忆操作出错（超时 / 报错 / 返回非法 JSON / 模型不可用）
        ↓
静默降级，绝不影响正常对话
```

- 检索超时（默认 3s）→ 放弃本次注入，正常聊天；
- 判官挂掉 → 回退本地规则判定，`fail_open=true` 默认保留事实；
- 向量服务不可用 → 降级为关键词检索 + 本地 RRF 融合；
- 后台处理异常 → 写入 `task_dlq` 死信队列，单条失败不阻断整批。

---

<a id="roadmap"></a>

## 🗺️ 开源演进路线图

### ✅ 已完成

| 阶段 | 主题 | 交付 |
| :--: | :-- | :-- |
| **Stage 1** | 多轮窗口缓冲 | 窗口打包、异步 Worker、生命周期托管 |
| **Stage 2** | 双向 LLM 钩子 | 注入 + 记录、静默降级、注入幂等 |
| **Stage 3** | 三态注入 + 指令层 | `system_head/tail` + `user_head`、`/memory` 四指令 |
| **Stage 4** | 端到端集成 | 全链路闭环测试、冷重启持久化 |
| **Stage 5** | 判官门禁 | 廉价判官 `jev`、价值打分、冲突软废弃 |
| **方案 A** | 生产级加固 | R-01 ~ R-06：置顶免死、真批量、实时落盘、严格隔离、连接池、纯读检索 |
| **Stage 7 / 方案 B** | 工业级引擎 | FTS5 倒排索引（中文预分词 + 四类同步触发器）、内存向量矩阵缓存、死信队列闭环重试 |

### ✅ 方案 B：原生 SQLite FTS5 引擎（已落地）

**原定目标**：把「手写字词重合 + BM25 近似」换成 SQLite 原生 FTS5 倒排检索。

**实际交付**（`tests/test_stage7_engine_b.py`，22 个用例）：

```sql
CREATE VIRTUAL TABLE l2_atoms_fts USING fts5(
    atom_id UNINDEXED, content, key_facts,
    tokenize = 'unicode61 remove_diacritics 2'
);
```

- **零依赖 BM25**：`MATCH` + `bm25()` 由 SQLite 内置完成，不再依赖手写打分（千级记忆 **35ms → 2.7ms**）；
- **中文分词管道**：实测 `unicode61` 不切分 CJK，因此索引/查询两侧统一做 **CJK 逐字预分词**
  （`fts_preprocess`），**无需 jieba 等任何第三方依赖**，中英文混合查询均可命中；
- **四类自动同步触发器**：`INSERT / UPDATE / DELETE / status 软废弃`，索引与主表永不脱节；
- **优雅回退**：SQLite 未编入 FTS5 时自动回退原 Python BM25，**零破坏变更**；
- **附赠两项**：内存向量矩阵缓存（`matrix @ query` 微秒级向量打分）与死信队列闭环重试
  ——详见 [工业级引擎升级](#plan-b)。

> **实测收益**：关键词召回约 **13x 加速**，向量召回从「全表反序列化 + 逐条余弦」降为**一次矩阵乘法**，
> 且依然是**零容器、单文件、零额外依赖**。

### 🔮 方案 C：真·时序知识图谱（再下一步）

**要解决的问题**：现在的记忆是「一堆独立事实原子」，缺少**事实之间的关系**和**时间维度**。
这两样东西缺失，就答不了这类问题：

- 「主人和小李是什么关系？」
- 「主人上个月喜欢喝美式，这个月改喜欢龙井了——哪个是现在的？」

**基础设施已经预留好了**（`kg_nodes` / `kg_edges` 表结构已建，运行期待接线）：

```text
                    ┌──────────────┐
                    │   主人 (人)   │
                    └──────┬───────┘
              ┌────────────┼────────────┐
              │ 喜欢(2026-09)│            │ 认识(2025-03)
              ▼            ▼            ▼
        ┌─────────┐  ┌─────────┐  ┌─────────┐
        │  美式    │  │  龙井    │  │  小李    │
        └─────────┘  └─────────┘  └─────────┘
          ↑ 已过期      ↑ 当前有效
```

- **时序边（temporal edges）**：每条关系带 `valid_from` / `valid_to`，不再需要把旧事实删掉；
- **冲突自动收敛**：「美式」的边被标记 `valid_to = now`，检索时只返回"当前有效"的事实；
- **多跳推理**：支持「主人 → 同事 → 小李」这类 1~2 跳查询，回答"关系型"问题；
- **时间旅行查询**：能回答「三个月前主人喜欢喝什么」。

### 🎯 其他待办

- [ ] 专属 WebUI 面板（配置项已预留）
- [ ] 检索权重细粒度可调（当前为内置 `0.55` 向量 + `0.45` 关键词）
- [ ] 记忆遗忘曲线 / 时间衰减（让久不用的记忆自动降权）
- [ ] 记忆导出 / 导入（JSON / Markdown）
- [ ] 更多宿主适配器（不限于 AstrBot）

---

<a id="privacy"></a>

## 🔒 数据安全与隐私

- **数据全在本地**：所有记忆存在你自己机器上的一个 SQLite 文件里，**不上传任何第三方**；
- **只发必要内容**：调用模型 API 时，仅发送待提炼的对话文本，不发送数据库、不发送其他会话内容；
- **物理隔离**：`strict_session_isolation` 默认开启，跨群/跨私聊记忆不互通；
- **可追溯**：被推翻的旧事实标记为 `superseded` 而非物理删除，可查"这条记忆是什么时候、被什么替代的"；
- **可备份**：`cp agent_memory.db backup.db`，一个文件搞定备份与迁移；
- **安全停机**：关闭流程严格保证 `flush → drain → close` 顺序，正常停机不丢记忆。

---

<a id="faq"></a>

## ❓ FAQ

<details>
<summary><b>数据量大了会不会慢？</b></summary>

SQLite 在万级记忆规模、带索引的情况下，检索是毫秒级的。真到十万级以上，方案 B 的 FTS5 + 方案 C 的图谱会接上。如果你的规模真到了企业级，建议直接用腾讯原版。
</details>

<details>
<summary><b>一个模型密钥都不配，能用吗？</b></summary>

能。走全降级链路：规则分句抽取事实 → 关键词（BM25 + 词重合）粗排 → 本地 RRF 融合排序 → 注入。效果不如配了向量模型，但**完全可用**。
</details>

<details>
<summary><b>为什么不用 aiosqlite / SQLAlchemy？</b></summary>

为了守住"零硬依赖"。`sqlite3` 是标准库，配合 WAL 模式 + 独立连接 + 纯读检索，已经能满足单机并发需求。少一个依赖，少一类线上问题。
</details>

<details>
<summary><b>方案 A 的"省 80% 账单"是怎么算的？</b></summary>

朴素实现每轮 3 次调用（抽取 + 门禁 + 向量化），5 轮共 15 次。合并批处理后 5 轮只需 3 次（1 抽取 + 1 门禁 + 1 批量向量化）。`(15-3)/15 = 80%`。
</details>

<details>
<summary><b>能用在非 AstrBot 的项目里吗？</b></summary>

核心逻辑（`storage.py` / `summarizer.py` / `retriever.py` / `evaluator.py` / `models_client.py`）都是纯 Python，**只依赖标准库 + 可选 aiohttp**，可以抽出来嵌入任何 Python Agent 框架。当前首发形态是 AstrBot 插件，宿主适配器在路线图里。
</details>

<details>
<summary><b>和腾讯原版是什么关系？</b></summary>

**致敬与借鉴，非分支、非官方。** 我们吸收了它的 L0~L3 分层模型与双路混合检索思想，用纯 Python + SQLite 重新实现了一个轻量版。做企业级记忆中台请用原版；做个人/小团队 Agent，用这个。
</details>

---

<a id="boundary"></a>

## 🚧 已知边界

以下能力**表结构 / 配置项已就绪，运行期链路待接入**，当前不影响 L1/L2/L3 主链路：

| 能力 | 当前状态 |
| :-- | :-- |
| 知识图谱运行期写入（`kg_nodes` / `kg_edges`） | 表结构与迁移映射已就绪，自动构建待方案 C 接入 |
| 专属 WebUI 面板（`webui.*`） | 配置项已预留，HTTP 面板待实现 |
| 检索权重细粒度调节（`retrieval.weight_*`） | 配置项已预留，当前为内置 `0.55` 向量 + `0.45` 关键词 |
| 原生 FTS5 关键词引擎 | ✅ 已落地（方案 B）：倒排索引 + 中文预分词 + 四类同步触发器，无 FTS5 时自动回退 |

---

<a id="license"></a>

## 📄 License

[MIT](LICENSE) © 2026 Zyw052

---

<div align="center">

**如果这个项目帮到了你，点个 ⭐ Star 就是最好的支持。**

*记忆不该是奢侈品，它应该是每个 Agent 的出厂配置。*

</div>
