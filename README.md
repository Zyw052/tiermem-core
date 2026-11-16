<div align="center">

# 🧠 TierMem Core

### 给你的 AI Agent 装一套「不会失忆、也不会自相矛盾」的大脑

**单文件 · 零容器 · 纯 Python + SQLite · 自带可视化面板的分层长期记忆内核**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![SQLite](https://img.shields.io/badge/Storage-SQLite%20WAL-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Containers](https://img.shields.io/badge/Docker%20%2F%20Redis%20%2F%20Milvus-0-2496ED?logo=docker&logoColor=white)](#-为什么你需要它)
[![Tests](https://img.shields.io/badge/Tests-94%20passed%20%C2%B7%209%20suites-2ea44f)](#-测试与质量保障)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

</div>

---

## 目录

| 章节 | 内容 |
| :-- | :-- |
| [为什么你需要它](#-为什么你需要它) | 大多数「有记忆」的机器人，其实是金鱼 |
| [七大核心优势](#-七大核心优势) | 我们到底强在哪 |
| [3 分钟快速上手](#-3-分钟快速上手) | 装上、配好、能跑 |
| [配置速查表](#-配置速查表) | 每个开关干什么 |
| [后端技术实现](#-后端技术实现) | 给想读代码的人 |
| [测试与质量保障](#-测试与质量保障) | 94 个用例锁死的行为契约 |
| [致谢](#-致谢-acknowledgements) | 站在谁的肩膀上 |
| [License](#-license) | MIT |

---

## 💡 为什么你需要它

LLM 天生失忆。于是大家开始给 Agent 加"记忆"，但常见的做法会撞上三堵墙：

1. **基础设施太重** —— 一套记忆系统要 Postgres + Redis + Milvus + Docker Compose，个人小内存 VPS 直接被压垮；
2. **记忆会自相矛盾** —— 用户上个月说"我住北京"，这个月说"我搬到上海了"，两条记忆一起躺在库里，检索时随机命中一条，Agent 就开始胡说；
3. **把系统提示词当用户记忆** —— 后台任务回调、RAG 召回块、系统提醒被当成"用户说过的话"存进长期库，越记越脏。

**TierMem Core** 的定位很简单：**一个 SQLite 文件，一套纯 Python 分层记忆内核，装进 AstrBot 就是一个带可视化面板的长期记忆大脑。** 不需要任何外部数据库、消息队列或向量服务。

> 本项目是 **AstrBot 插件**（插件名 `astrbot_plugin_agent_memory`），仓库名 `tiermem-core`。

---

## 🚀 七大核心优势

### 1. 单文件零容器，轻量部署

不依赖 Docker、Postgres、Redis、Milvus、FAISS。全部状态落在一个 **SQLite 文件**（WAL 模式）里，`pip install` 之外无需任何部署动作。**1 核 1G 的个人 VPS 也能稳跑**，内存占用随数据量线性、可控。

### 2. 开箱即用的插件 WebUI 面板

插件自带可视化记忆大盘（`views/dashboard/index.html`），**零外部 CDN 依赖**，与 AstrBot 插件页面桥接：

| 页面 | 你能做什么 |
| :-- | :-- |
| **概览** | 三层记忆通电自检、L1/L2/L3 分层计数、噪声拦截率、DLQ 积压、检索模式 |
| **浏览** | 分页翻看 L2 记忆原子、看每条记忆的重要度/访问次数/状态 |
| **搜索** | 关键词实时检索，验证"它到底记不记得" |
| **图谱** | L3 实体节点与关系边、置信度与演化状态 |
| **配置** | 服务商绑定、检索权重、衰减参数 |
| **死信** | 失败任务排查 + 一键重放补提 |

服务商配置支持两条路：插件配置里用宿主 LLM 的 `select_provider` 下拉直接选总结/判官模型；**Embedding / Rerank 服务商则可以在 WebUI 面板里直接下拉绑定并热重建**，不用改配置文件、不用重启。

### 3. 冲突自动覆盖与谱系追踪（`superseded`）

这是和"只会追加"的记忆系统最本质的区别。

用户改口时（"我搬到上海了" vs 旧记忆"我住北京"），系统通过**同属性槽不同值**与**否定极性相反**双重规则识别冲突，把旧记忆标记为 `status='superseded'`，写入 `superseded_by` / `superseded_at` 形成**演化链条**，并从检索结果中排除。

- 不再新旧混杂、自相矛盾；
- 旧记忆**不物理删除**，保留完整演化谱系，可追溯"它以前以为你住哪"；
- 关联的 KG 边同步 `status='invalidated'`、`weight=0`，同样保留可追溯性；
- 被**置顶（pinned）**的记忆拥有"免死金牌"，规则层不会把它判成冲突目标。

### 4. 系统噪声双层过滤（`noise_filter.py`）

记忆污染的最大来源不是用户，是**系统自己**。`noise_filter.py` 是一个纯函数、零依赖、零 IO 的双层过滤器，在**写入入口**与**抽取入口**统一拦截：

| 层级 | 处理对象 |
| :-- | :-- |
| **块级（整体丢弃）** | `<RAG-Faiss-Memory>` 召回块、`<system_reminder>`、`[SYSTEM NOTICE]`、```` ```system ```` 代码块、HTML 注释、未闭合的起始标签兜底 |
| **行级（逐行剔除）** | `Background task for subagent '...' finished.`、`【项目大盘·跨代理共享状态】`、各类后台回调与心跳注入 |

同时还有**行内清洗**：正文里夹着的噪声块会被摘掉，有效内容原样保留。正常对话零误伤（中文长句、含尖括号的技术讨论都不受影响），统计与最近拦截样本可在 WebUI「噪声」页直接查看。**任何正则/编码异常都退化为"原文放行"，绝不阻断记忆主链路。**

### 5. 死信队列闭环（DLQ）

提取/向量化失败**不会静默丢失**。失败的异步任务进入 `task_dlq` 表，带 `retry_count` / `max_retries`（默认 3）与 `next_retry_at`，支持**指数退避重试**。WebUI「死信」页可以看失败原因、任务类型、重试次数，并**一键重放补提**。记忆不再"悄悄少了点什么"。

### 6. 时间感知 + 艾宾浩斯衰减 + 知识图谱（KG）

- **时间感知**：事实文本保留时间语境（"曾是 / 现在 / 打算"），KG 边带双时态字段 `valid_from` / `valid_until`，能表达"这条关系在什么时间区间内成立"。
- **艾宾浩斯衰减**：检索打分乘上时间衰减因子 `exp(-λ·Δt)`（默认 λ=1.0，下限 0.05）。**高频访问的记忆被强化、长期不被使用的记忆自然沉降**——模拟人类记忆曲线，而不是无限平权堆积。
- **知识图谱多跳检索**：L3 实体节点 + 关系边，支持种子实体扩展、1-hop 关联边召回、关联加分（默认 +0.15），并把图谱关联解释**回注到 prompt**，让 Agent 说出"因为 A 和 B 有关，所以我推断…"。

### 7. LivingMemory 一键无损迁移（`migrator.py`）

老用户不用重来。`migrator.py` 支持从旧版 `livingmemory.db` 无损迁移：

```
documents      -> l1_docs
memory_atoms   -> l2_atoms
graph_nodes    -> kg_nodes
graph_edges    -> kg_edges
```

迁移前**自动生成带时间戳的 `.bak` 安全快照**，支持 `--dry-run` 只读预览统计，孤儿边自动跳过，全程可回滚。

---

## ⚡ 3 分钟快速上手

### 第 1 步：装进 AstrBot

```bash
cd /你的/AstrBot/data/plugins
git clone https://github.com/Zyw052/tiermem-core.git astrbot_plugin_agent_memory
```

> 目录名必须是 `astrbot_plugin_agent_memory`（AstrBot 按目录名注册插件）。
> 也可以在 AstrBot WebUI 的插件市场/插件管理中安装。

**环境要求**：Python 3.10+、AstrBot 本体。**不需要** Docker / Postgres / Redis / Milvus。

### 第 2 步：重载插件

在 AstrBot WebUI → 插件管理 里重载 `astrbot_plugin_agent_memory`，日志出现下面这行即成功：

```
[agent_memory] 插件 WebUI API 已注册 9 条路由。
```

### 第 3 步：绑定模型（两条路，任选）

**路线 A — 插件配置页（推荐，最快）**

打开插件配置，只需要动这几个：

| 配置项 | 说明 |
| :-- | :-- |
| `enable_memory` | 打开总开关 |
| `llm_provider_id` | **`select_provider` 下拉**，直接选宿主已有的 Chat 服务商（做对话总结/事实抽取） |
| `judge_provider_id` | **`select_provider` 下拉**，选判官模型（做价值门禁与冲突判定） |

保存即生效，3 分钟够了。

**路线 B — WebUI 记忆面板（Embedding / Rerank 走这条）**

进入插件页面 → **配置** 标签：

1. 在 **Embedding 服务商** 下拉里选宿主已配置的 Embedding 服务商；
2. 在 **Rerank 服务商** 下拉里选宿主已配置的 Rerank 服务商；
3. 点「保存并热重建」——绑定写入 `provider_bindings.json`，引擎**热重建**，无需重启。

> 没有 Embedding / Rerank 服务商也能跑：系统会自动降级为 **FTS5 词法 + 向量余弦（无向量时走词法余弦）+ 本地 RRF 融合**，功能不瘫、只是精度下降。检索状态在 WebUI 概览页一眼可见（`provider` / `manual` / `rrf-degraded`）。

### 第 4 步：验证

在 AstrBot 里对它说几句带个人信息的话，然后：

```
/memory status     # 看三层记忆计数与引擎状态
/memory recall 我喜欢喝什么   # 手动召回测试
```

或者直接打开 WebUI 记忆面板，「概览 → 浏览」里就能看到刚存进去的记忆原子。

### 可选：迁移老数据

```bash
# 先预览，不改任何东西
python3 migrator.py --dry-run

# 确认后正式迁移（自动备份 .bak）
python3 migrator.py
```

非标准部署路径可以用环境变量覆盖，不必改代码：

```bash
export ASTRBOT_LIVINGMEMORY_DB=/your/path/livingmemory.db
export ASTRBOT_AGENT_MEMORY_DIR=/your/path/astrbot_plugin_agent_memory
```

---

## 🎛 配置速查表

<details>
<summary><b>点开看完整配置分组</b></summary>

**顶层开关**

| 配置项 | 默认 | 说明 |
| :-- | :-- | :-- |
| `enable_memory` | `true` | 分层记忆总开关 |
| `memory_injection_enable` | `true` | 是否把记忆注入 prompt |
| `summary_turn_interval` | `5` | 每 N 轮触发一次总结打包 |
| `llm_provider_id` | — | `select_provider`：总结/抽取模型 |
| `judge_provider_id` | — | `select_provider`：判官模型 |
| `embedding_provider_id` | — | 向量化服务商 ID |
| `rerank_provider_id` | — | 重排服务商 ID |
| `db_path` | 插件数据目录 | 记忆数据库路径 |

**`retrieval` 检索与注入**

| 配置项 | 默认 | 说明 |
| :-- | :-- | :-- |
| `top_k` | `6` | 单次召回条数 |
| `weight_vector` / `weight_bm25` / `weight_importance` | `0.55` / `0.3` / `0.15` | 三路打分权重 |
| `recall_pool_limit` | `500` | 粗排候选池上限 |
| `retrieve_timeout` | `3.0` | 检索超时（秒），超时静默降级 |
| `strict_session_isolation` | `true` | 严格会话隔离 |
| `prompt_injection_max_chars` | `2000` | 注入字符上限 |
| `inject_position` | `system_head` | 注入位置 |
| `graph_recall_enabled` | `true` | 启用图谱关联召回 |
| `graph_score_bonus` | `0.15` | 图谱关联加分 |
| `decay_enabled` / `decay_lambda` / `decay_floor` | `true` / `1.0` / `0.05` | 艾宾浩斯衰减 |
| `max_pinned_l3` | `15` | 置顶记忆注入上限 |

**`evaluator` 判定门禁**

| 配置项 | 默认 | 说明 |
| :-- | :-- | :-- |
| `enable` | `true` | 价值门禁开关 |
| `min_value_score` | `0.35` | 价值分阈值 |
| `conflict_check` | `true` | 冲突检测（→ `superseded`） |
| `conflict_similarity` | `0.5` | 冲突相似度下限 |
| `fail_open` | `true` | 判定失败时保留事实（不误杀） |

**`graph` / `noise_filter` / `webui` / `migration`**

| 配置项 | 默认 | 说明 |
| :-- | :-- | :-- |
| `graph.enable` | `true` | L3 图谱抽取 |
| `graph.max_triples_per_fact` | `6` | 单事实三元组上限 |
| `noise_filter.enabled` | `true` | 系统噪声过滤 |
| `noise_filter.strip_inline` | `true` | 行内噪声清洗 |
| `webui.register_api` | `true` | 注册页面数据接口 |
| `migration.auto_detect_legacy` | `true` | 自动检测旧资产 |

</details>

---

## 🔧 后端技术实现

> 下面这段是给想看代码/想二次开发的人。所有数字均与当前代码一致。

### 模块划分（10 个 Python 模块）

```
astrbot_plugin_agent_memory/
├── main.py              # 插件入口：配置装配、指令组、钩子注册、引擎通电
├── storage.py           # 存储层：SQLite schema、迁移、FTS5、向量矩阵、DLQ、WebUI 查询
├── summarizer.py        # 异步提取管线：缓冲打包 → 事实抽取 → 判定 → 向量化 → 晋升
├── retriever.py         # 检索层：三路召回 + 融合 + 图谱扩展 + 衰减重排
├── models_client.py     # 模型客户端：Embedding / Rerank / 余弦 / RRF 融合
├── evaluator.py         # 判定门禁：价值评分 + 冲突检测 + 属性槽比对
├── host_providers.py    # ProviderBridge：宿主服务商与独立模型双通道适配
├── noise_filter.py      # 系统噪声双层过滤器（纯函数、零依赖）
├── migrator.py          # LivingMemory 无损迁移引擎
├── page_api.py          # 插件 WebUI 后端 API（9 条路由）
├── _conf_schema.json    # 插件配置 Schema
├── metadata.yaml        # 插件元信息
├── views/dashboard/     # 记忆面板前端（单页，零外部依赖）
└── tests/               # 9 套自执行测试（94 个用例）
```

### 一、分层记忆架构

| 层 | 表 | 职责 |
| :-- | :-- | :-- |
| **L1 短期工作窗口** | `l1_docs` | 会话内原始对话与工作区文档，带 `scope_key` 隔离与版本号，按轮次/静默阈值打包成 batch |
| **L2 情景记忆库** | `l2_atoms` | 抽取后的事实原子：`content` / `topics_json` / `key_facts_json` / `importance` / `access_count` / `embedding_blob`，附 `status`、`superseded_by`、`superseded_at`、`pinned`、`tier` |
| **L3 语义与谱系** | `l2_atoms` 的 `status`/`superseded_*` + `l2_atoms_fts` | 冲突覆盖把旧原子标 `superseded` 并写演化链；FTS5 虚拟表 + 3 个触发器（INSERT/UPDATE/DELETE）保证索引与主表永不脱节 |
| **时序 KG** | `kg_nodes` / `kg_edges` | 实体节点（`weight` / `mention_count`）+ 关系边（`confidence` / `weight` / `valid_from` / `valid_until` / `status`），边随原子失效联动 `invalidated` |
| **异步任务** | `task_dlq` | 死信与重试：`retry_count` / `max_retries` / `next_retry_at` |

写入路径是 L1 → （打包）→ L2 → （三元组抽取）→ KG；读取路径反向渗透，L3 置顶记忆享有独立注入配额。

### 二、混合检索引擎

一次召回走四段流水线：

**① 三路并行粗排**

- **词法路**：SQLite 原生 **FTS5** 倒排（`unicode61` + 应用层 CJK 预分词写入 `fts_text` 缓存列）。**FTS5 不可用时自动回退纯 Python BM25**，零破坏。
- **向量路**：全量活跃向量归一化后装入**二维矩阵缓存**，`matrix @ q` 一次矩阵乘法算出全部余弦相似度（numpy 加速，微秒级）；**无 numpy 时退化为纯 Python 单遍循环**，功能等价只是慢一些。矩阵按 `matrix_cache_sync_interval`（默认 5s）校验增量同步。
- **重要度路**：`importance` + 访问次数 + 置顶状态。

> 说明：本实现**不依赖 `sqlite-vec` 扩展**，向量检索走内存矩阵余弦，因此不存在"宿主 SQLite 未编译扩展就瘫痪"的问题。

**② RRF 倒数排名融合**

当没有可用 Rerank 服务商时，用 **Reciprocal Rank Fusion**（默认 `k=60`）把多路排名融合成统一分数——不依赖各路分数量纲，稳定且抗单路抖动。

**③ Cross-Encoder Rerank**

有 Rerank 服务商时优先走真重排：候选池（默认 500）→ Cross-Encoder 精排 → 取 `top_k`（默认 6）。三级降级链：**宿主 Rerank 服务商 → 手填 API 凭证 → 本地 RRF**，任一级失败自动往下一级掉，绝不空手而归。

**④ 图谱扩展 + 时间/艾宾浩斯加权**

- 种子实体（默认 5）→ 关联边（默认 20）→ 关联加分（默认 0.15），并把图谱关联解释回注 prompt；
- 最终分数乘 `exp(-λ·Δt)` 时间衰减因子（λ 默认 1.0，下限 0.05），`pinned` 记忆免衰减。

### 三、异步提取管线

```
[对话轮次] → ① 噪声过滤 → ② L1 缓冲
                              ↓ (轮次阈值 / 静默阈值)
                       ③ 打包 batch 入队
                              ↓ 独立 Worker 协程消费
            ④ 单次 LLM 事实抽取（含规则分句兜底）
                              ↓
            ⑤ 判定门禁：价值分 + 冲突检测
                              ↓                    ↓
        ⑥ 批量 Embedding              冲突 → 旧原子 superseded + 边 invalidated
                              ↓
            ⑦ 写 L2 / 抽 KG 三元组 / FTS 索引
                              ↓
        ⑧ 任一步失败 → task_dlq（指数退避重试，WebUI 可重放）
```

关键工程约束：

- **批量合并**：一次 batch 只调 **1 次抽取 LLM + 1 次判定 + 1 次批量 Embedding**，把 token 成本压到最低；
- **全链降级**：LLM 不可用 → 规则分句抽取；Embedding 不可用 → 词法余弦；Rerank 不可用 → RRF；FTS5 不可用 → 进程内 BM25；
- **零阻塞**：检索侧只读内存计数，访问次数异步刷盘（可按间隔或阈值触发），优雅停机尾部冲洗；
- **幂等**：`Task` 消费、DLQ 重放、`close()` 均可重复调用。

### 四、`ProviderBridge`：宿主 / 独立双通道适配

`host_providers.py` 解决一个现实问题：**同一个模型能力，可能来自宿主 AstrBot 的 Provider，也可能来自用户手填的 HTTP API。**

- **类型精确识别**：`detect_provider_kind()` 区分 `chat` / `embedding` / `rerank`，**类型不符直接忽略并降级**，绝不把 Chat 服务商错当 Embedding 用（这是最容易踩的坑）；
- **运行期绑定**：`provider_bindings.json` 的优先级高于配置文件，支持运行期改绑 + 解除，改完**热重建**引擎，不重启；
- **真实调用**：后端适配器真调宿主方法（含异步后端），不是空壳 mock；
- **列表分组**：`list_providers()` 按类型分组，直接喂给 WebUI 下拉框。

### 五、WebUI `page_api.py`：9 条路由

注册前缀为 `/{PLUGIN_NAME}/page`（宿主要求第一段必须是插件名，否则 `_match_registered_web_api` 命中不了）。

| # | 路由 | 方法 | 作用 |
| :-- | :-- | :-- | :-- |
| 1 | `/status` | GET | 引擎通电自检与分层统计 |
| 2 | `/memories` | GET | 分页浏览 / 关键词搜索记忆 |
| 3 | `/memory/<atom_id>` | GET | 单条记忆详情与关联三元组 |
| 4 | `/pin` | POST | 设置 / 取消置顶（L3 晋升） |
| 5 | `/noise` | GET | 噪声过滤统计与最近拦截样本 |
| 6 | `/dlq` | GET | 死信队列条目（失败任务排查） |
| 7 | `/providers` | GET | 宿主服务商清单与当前绑定 |
| 8 | `/bindings` | POST | 保存运行期服务商绑定并热重建 |
| 9 | `/recall` | GET | 手动召回测试（观察检索与注入结果） |

**设计原则**：所有处理器**不抛异常**，失败统一返回 `{"status": "error", ...}`；只读为主，唯一两个写操作是「置顶」与「绑定服务商」（都写入插件数据目录）；返回体统一 `{"status": "ok", "data": {...}}`。存储缺失、请求异常等极端入参下依然返回可读错误，**WebUI 不会 500**。

### 六、健壮性与可观测性

- **旧库平滑升级**：所有 schema 变更走 `ALTER TABLE ADD COLUMN` 幂等补齐，历史数据自动 backfill（含 `fts_text` 预分词回填），升级零数据丢失；
- **安全快照**：迁移/危险操作前自动 `.bak`；
- **可观测快照**：`/memory status` 与 WebUI 概览直接暴露 FTS 状态、向量矩阵后端（`numpy` / `python`）、DLQ 积压、检索模式（`provider` / `manual` / `rrf-degraded`）、各阶段统计计数。

---

## 🧪 测试与质量保障

**9 套自执行测试、共 94 个用例，全部通过。** 不依赖 pytest，每套都能单独跑：

```bash
cd /你的/AstrBot/data/plugins/astrbot_plugin_agent_memory
for f in tests/test_stage*.py; do echo "== $f"; python3 "$f" || exit 1; done
```

| 套件 | 文件 | 用例 | 覆盖重点 |
| :-- | :-- | --: | :-- |
| Stage 1 | `test_stage1_buffer.py` | 2 | 缓冲打包/冲洗、Worker 端到端消费 |
| Stage 2 | `test_stage2_hooks.py` | 5 | 双向钩子装配、prompt 注入、轮次记录、异常/超时静默降级 |
| Stage 3 | `test_stage3_commands.py` | 11 | 注入位置路由、`memory.{status,recall,pin,migrate}` 指令、迁移统计 |
| Stage 4 | `test_stage4_e2e.py` | 2 | 全链路端到端、冷重启持久化、优雅停机、资源释放 |
| Stage 5 | `test_stage5_evaluator_gate.py` | 14 | 价值门禁、冲突排除、`superseded` 生效、total 装配 |
| Stage 6 | `test_stage6_hardening.py` | 7 | 止血热修、会话隔离、检索纯读、连接池幂等 |
| Stage 7 | `test_stage7_engine_b.py` | 22 | FTS5 探测与降级、向量矩阵缓存、DLQ 闭环、旧库升级 backfill |
| Stage 8 | `test_stage8_graph_c.py` | 12 | 时序 KG 三跳、衰减重排、软废弃联动、prompt 图谱解释 |
| Stage 9 | `test_stage9_p0_hardening.py` | 19 | 配置 Schema 校验、ProviderBridge 类型判定与运行期绑定、噪声过滤零误伤、WebUI 9 路由与极端入参降级 |

> 用例全部以「真实读写临时 SQLite + 假 Context/假宿主」方式跑，不 mock 存储层；Stage 1–4 已验证**冷重启后记忆不丢**、**二次停机资源完全释放**。

---

## 🙏 致谢 (Acknowledgements)

本项目并非凭空设计，其架构思想深受以下开源项目启发：

### TencentDB Agent Memory

**[TencentDB Agent Memory](https://github.com/TencentCloud/TencentDB-Agent-Memory)** —— 腾讯云开源的 Agent 长期记忆框架。

TierMem Core 的**分层记忆模型（L1/L2/L3）、冲突消解（superseded 演化链）与知识图谱架构**，均以其设计思想为灵感来源。在此谨向原作者与贡献者致以诚挚感谢。

与原项目不同的是，TierMem Core 面向 **AstrBot 个人自部署 / 小内存 VPS** 场景做了一次彻底的**轻量化重构**：把依赖收敛到**纯 Python + SQLite**，去掉容器化中间件，补齐插件级 WebUI 面板与噪声过滤，使"个人机器人也能拥有像样的长期记忆"成为可能。项目遵循与原项目相同的 **MIT 协议**理念，自由使用、修改与分发。

### AstrBot 框架与社区

感谢 **[AstrBot](https://github.com/AstrBotDevs/AstrBot)** 提供插件化的 Agent 框架、Provider 抽象与插件页面桥接机制——没有 `register_web_api` 与 Provider 体系，就不会有本项目的 WebUI 面板和 `ProviderBridge`。也感谢 AstrBot 社区各位插件作者在实践中暴露的真实问题，这些反馈直接塑造了本项目 Stage 6/7/9 的加固重点。

---

## 📄 License

本项目基于 [MIT License](LICENSE) 发布，Copyright (c) 2026 Zyw052。

---

<div align="center">

**如果它让你的机器人少说了一句"我不记得了"，那就值了。**

</div>
