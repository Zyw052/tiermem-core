# astrbot_plugin_agent_memory

**智涌原生分层记忆插件** — 为 AstrBot 提供 L1/L2/L3 三层结构化记忆、多轮缓冲异步总结、两阶段混合检索与三态 Prompt 注入能力。

全部数据落地于本地 SQLite（WAL 模式），**零外部服务依赖即可运行**（未配置模型密钥时自动降级为规则抽取 + 关键词检索 + 本地 RRF 重排）。

---

## 1. 功能特性

### 1.1 三层记忆体系

| 层级 | 存储表 | 内容 | 产生方式 |
| :-- | :-- | :-- | :-- |
| **L1 工作区** | `l1_docs` | 原始对话轮次（`doc_type=dialogue_turn`），保留 User/Assistant 全量文本 | 每轮对话入队后由后台 Worker 写入 |
| **L2 情景原子** | `l2_atoms` | 从对话中提炼的独立事实原子，带 topics / key_facts / 重要度 / 向量 / 访问计数 | 事实抽取 → 向量化 → 余弦去重融合 |
| **L3 核心画像** | `l2_atoms`（`pinned=1, tier=3`） | 偏好、习惯、硬性约束等必须长期遵守的核心记忆，检索时**强力置顶** | 含 L3 关键词或重要度 ≥ 0.85 时自动晋升 |

> **L3 自动晋升关键词**：`喜欢` / `讨厌` / `偏好` / `必须` / `严禁` / `密码` / `习惯`。
> 命中即 `importance=0.90`、`pinned=1`、`tier=3`，无需人工干预。

### 1.2 多轮缓冲 + 异步总结（不阻塞对话）

- 每轮对话**只进内存缓冲池**，零 IO、零阻塞；
- 累积满 `summary_turn_interval`（默认 5 轮）才**整窗打包**为 `kind=batch` 事件推入后台队列，显著降低 LLM 调用次数；
- 不足阈值的尾部轮次由 `flush_session()` / `flush_all()` 强制冲洗，**会话结束/插件卸载不丢记忆**；
- 背包后台 Worker 基于 `asyncio.Queue`，单轮处理失败不阻断整批，异常统一降级写入 `task_dlq` 死信队列。

### 1.3 两阶段混合检索

```
Query ──► ① L3 强力置顶捞取（pinned=1，按 importance 排序，当前会话优先）
      ──► ② 第一阶段粗排：向量余弦召回 Top-N  ⊕  关键词 BM25/字词重合召回 Top-N
                              （去重合并，0.55 向量 + 0.45 关键词融合）
      ──► ③ 第二阶段精排：BGE-Reranker 深度打分  （无密钥时降级为本地 RRF 融合）
      ──► ④ 组装 Markdown 记忆块 + 更新命中项访问计数
```

### 1.4 双向 LLM 钩子与静默降级

| 钩子 | 时机 | 职责 |
| :-- | :-- | :-- |
| `on_llm_request` | LLM 请求前 | 两阶段检索 → 按注入策略写入 `system_prompt` / `prompt` |
| `on_llm_response` | LLM 响应后 | 抽取「用户输入 + 机器人回复」压入多轮缓冲池 |
| `on_decorating_result` | 发送前 | **仅在宿主缺少 `on_llm_response` 时**作为兜底记录点 |

**保障机制**：全量 `try-except`，检索超时（默认 3s）即放弃注入；任何异常都不影响 AstrBot 正常对话流。
**过滤规则**：流式分片（`is_chunk=True`）与工具调用轮（`tools_call_name` 非空）不记录；同会话相同内容 2 秒内只记一次（抵御宿主重复回调）。
**注入幂等**：记忆块携带隐藏标记 `<!-- agent_memory:injected:v1 -->`，同一请求（含工具调用多轮）绝不重复注入。

---

### 1.5 自作判定门禁（Evaluator Gate）

在 L2 原子事实入库前插入一层“廉价模型预判定”，用极小成本挡住噪声、守住一致性：

- **信息价值打分**：`value_score` 过滤寒暄、语气应答与无信息量废话（默认阈值 `0.35`）；
- **事实冲突与推翻检测**：识别“我不喜欢喝绿茶”这类与旧事实矛盾的新事实，把被推翻的旧原子标记为
  `status='superseded'` 软废弃（保留可追溯性，检索侧不再召回）；
- **异常平滑降级**：判官模型超时 / 报错 / 返回非法 JSON 时自动回退本地规则判定，
  主记忆流程**永不阻断**（`fail-open` 默认开启，宁多记不丢记忆）；
- **判官模型**：默认 `jev`，`api_base` / `api_key` 留空时自动复用 `summarizer.*` 凭证，无需重复填写。

### 1.6 方案 A 止血热修（稳定性加固）

针对上线风险清单做了 6 项根治，全部有 Stage 6 专项用例锁定：

| 编号 | 风险 | 修复要点 |
| :-- | :-- | :-- |
| **R-01** | 置顶记忆被误杀 | L3 `pinned=1` 拥有免死金牌：规则层跳过置顶候选，软废弃 SQL 追加 `AND COALESCE(pinned,0)=0` |
| **R-02** | 未满阈值对话蒸发 | `enqueue_dialogue` 即**实时落盘 L1**；析出 L2 后回写 `l2_processed`；新增 idle 超时自动刷盘 |
| **R-03** | N+1 调用账单风暴 | N 轮合并为**一个 Context Window**：1 次事实抽取 + 1 次门禁 + 1 次批量 Embedding |
| **R-04** | 跨群/私聊串台 | `strict_session_isolation` **默认开启**，仅 `scope_key='global'` 允许全局召回 |
| **R-05** | 检索抢 SQLite 写锁 | 检索链路纯读，命中计数内存累加 + 异步定期刷盘（`close()` 兜底刷净） |
| **R-06** | 频繁 TLS 握手 | Embedding / Rerank / 判官 / 总结均复用持久 `aiohttp.ClientSession` 连接池 |

## 2. 架构与数据流

```text
                     ┌──────────────── AstrBot 对话链路 ────────────────┐
                     │                                                  │
   用户消息 ──► on_llm_request ──► [ 检索 + 注入 System Prompt ] ──► LLM ──► on_llm_response
                     │            ▲                                        │
                     │            │                                        ▼
                     │     MemoryRetriever                       多轮缓冲池（内存）
                     │            ▲                                        │ 满 N 轮打包
                     │            │                                        ▼
                     │     l2_atoms (L2 + L3)  ◄──── MemorySummarizer 后台 Worker
                     │            ▲                                  │
                     │            └──── l1_docs (L1 原始轮次)  ◄───────┤
                     │                                               └──► task_dlq（失败死信）
                     └──────────────────────────────────────────────────┘
```

### 模块划分

| 文件 | 职责 |
| :-- | :-- |
| `main.py` | Star 主类：配置规范化、五组件总装、生命周期托管、双向钩子、`/memory` 指令层 |
| `storage.py` | SQLite 存储引擎：DDL、连接管理（WAL）、分层统计、`toggle_atom_pin()` 置顶切换 |
| `summarizer.py` | 多轮缓冲池 + 异步 Worker：事实抽取（LLM / 规则回退）、**自作判定门禁接入**、向量去重融合、L3 晋升、DLQ 降级 |
| `evaluator.py` | **自作判定门禁**：廉价判官模型（默认 `jev`）信息价值打分、事实冲突/推翻检测、超时与异常平滑降级 |
| `retriever.py` | 两阶段混合检索：L3 置顶捞取、粗排召回、精排重排、Markdown 上下文组装 |
| `models_client.py` | BGE-M3 向量客户端与 BGE-Reranker 客户端，含维度对齐校验与 RRF 降级 |
| `migrator.py` | 旧版 `livingmemory` 资产无损迁移（dry-run / 备份快照 / 孤儿边跳过） |
| `tests/` | Stage 1~5 全量测试套件（单元 + 端到端闭环 + 判定门禁） |

---

## 3. 快速开始

将插件目录放入 AstrBot 插件路径后，**最少只需 3 项配置**即可运行：

```jsonc
{
  "enable_memory": true,
  "summary_turn_interval": 5,
  "retrieval": { "inject_position": "system_head" }
}
```

此状态下未配置任何模型密钥，插件将走**全降级链路**：
规则分句抽取事实 → 关键词（BM25 + 词重合）粗排 → 本地 RRF 精排 → 注入。

要实现完整语义能力，再补充向量与重排模型：

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

> 也可填写 `summarizer.embedding_provider_id` / `summarizer.rerank_provider_id` / `summarizer.provider_id`，
> 插件会尝试从 AstrBot 已配置的 Provider 中自动复用 `api_base` / `api_key` / `model`，无需重复填写密钥。

---

## 4. 三态注入配置指南（核心）

记忆块写入位置直接决定模型对记忆的服从度。通过 `retrieval.inject_position` 三态路由：

| 取值 | 写入目标 | 位置 | 推荐模型 | 适用场景 |
| :-- | :-- | :-- | :-- | :-- |
| **`system_head`**（默认） | `req.system_prompt` | 最前端（人格提示词之前） | Claude / GPT-4o | 记忆作为"最高优先级上下文"，指令权重最高 |
| **`system_tail`** | `req.system_prompt` | 最末端（人格提示词之后） | 通用 | 人设优先、记忆作为补充参考，避免稀释人格 |
| **`user_head`** | `req.prompt` | 用户 Query 最前端 | DeepSeek / Qwen | 对即时提问敏感的模型，记忆与当前问题紧邻 |

### 4.1 别名兼容

配置值大小写、分隔符（`-` / 空格）均不敏感，并兼容以下同义写法：

| 标准值 | 兼容别名 |
| :-- | :-- |
| `system_head` | `head`（Stage 2 旧值）、`system`、`system_prompt_head`、`prepend` |
| `system_tail` | `tail`（Stage 2 旧值）、`system_prompt_tail`、`append` |
| `user_head` | `prompt_head`、`user`、`query_head` |

非法 / 空值统一回退 `system_head`，**路由永不失效**。

### 4.2 注入相关配置

```jsonc
{
  "memory_injection_enable": true,        // 注入总开关（可与沉淀独立关闭，只记忆不注入）
  "retrieval": {
    "inject_position": "system_head",     // 三态路由
    "prompt_injection_max_chars": 2000,   // 记忆块字符上限，0 = 不限制
    "retrieve_timeout": 3.0,              // 检索超时(秒)，超时静默放弃本次注入
    "top_k": 6                            // 单次注入的记忆条数上限
  }
}
```

> **调优建议**：若发现人格被记忆稀释，改用 `system_tail`；若发现模型忽略记忆，改用 `system_head`；
> 若记忆内容过长挤占上下文，下调 `prompt_injection_max_chars`。

### 4.3 注入效果示例

```markdown
<!-- agent_memory:injected:v1 -->
## 🧠 长期记忆上下文 (Agent Memory Context)
### 📌 核心记忆与约束 (L3 Pinned)
1. **[核心/preference]** 主人喜欢喝龙井，不加糖。 *(重要度: 0.90)*
2. **[核心/preference]** 主人必须每天早上喝一杯温水。 *(重要度: 0.90)*
### 🔍 相关情景记忆 (L2 Retrieved)
- 上周我们讨论了西湖龙井的采摘季节。 *(相关度: 0.7341)*
```

---

## 5. 管理指令清单

指令组 `/memory`，子指令如下（宿主不支持指令组时自动降级为扁平 `/memory_status` 等）：

| 指令 | 参数 | 功能 | 输出要点 |
| :-- | :-- | :-- | :-- |
| `/memory status` | — | 查看运行状态快照 | L1/L2/L3 体量、待总结轮次与会话数、注入策略、模型健康度、双向钩子统计 |
| `/memory recall <关键词>` | 检索词 | 手动执行两阶段检索 | L3 置顶列表（ID/类型/重要度）+ L2 精排列表（`score=` / ID / 内容） |
| `/memory pin <atom_id>` | 原子 ID | 切换 L3 置顶状态 | `pinned=1,tier=3` 与 `pinned=0,tier=2` 可逆切换，回显内容与当前状态 |
| `/memory migrate` | — | 迁移旧版 `livingmemory` 资产 | 源库路径、dry-run/正式模式、L1/L2/图谱迁移计数、备份快照路径、耗时 |

### 输出示例：`/memory status`

```markdown
## 🧠 分层记忆状态
- **体量**: L1 历史 `7` 条 · L1 待总结 `0` 轮（0 会话） · L2 原子 `7` 条 · L3 置顶 `3` 条 · 图谱 `0` 节点 / `0` 关系
- **策略**: 注入位置 `system_head`（System Prompt 最前端（推荐：Claude / GPT-4o）） · 总结阈值 `5` 轮/批 · 记忆总开关 `开启` · 注入开关 `开启`
- **模型**: Embedding ✅ 可用（BAAI/bge-m3） · Rerank ✅ 可用（BAAI/bge-reranker-v2-m3）
- **钩子**: 注入 成功 `1` / 跳过 `0` / 降级 `0` · 记录 成功 `5` / 跳过 `0` / 降级 `0`
```

> 未配置密钥时，模型行显示 `⚠️ 未配置密钥（零向量降级）` / `⚠️ 未配置密钥（本地 RRF 降级）`，便于快速定位降级原因。

---

## 6. 配置项速查

### 记忆与生命周期

| 配置项 | 类型 | 默认 | 说明 |
| :-- | :-- | :-- | :-- |
| `enable_memory` | bool | `true` | 分层记忆总开关（关闭则不启 Worker、不记录、不注入） |
| `db_path` | string | `''` | 记忆数据库路径。留空→插件数据目录；填目录→目录下 `agent_memory.db`；填文件→直接使用 |
| `summary_turn_interval` | int | `5` | 多轮缓冲打包阈值（轮） |
| `summary_idle_flush_seconds` | int | `180` | 会话静默超时自动刷盘（0 关闭），防止尾部轮次长期滞留内存 |
| `summarizer_queue_maxsize` | int | `1000` | 后台消费队列容量，满则降级写 DLQ |
| `memory_injection_enable` | bool | `true` | 注入开关，独立于总开关 |

### 总结模型（`summarizer.*`）

| 配置项 | 默认 | 说明 |
| :-- | :-- | :-- |
| `provider_id` | `''` | 复用 AstrBot Provider 的 LLM 凭证 |
| `embedding_provider_id` / `rerank_provider_id` | `''` | 复用 Provider 的向量/重排凭证 |
| `api_base` / `api_key` / `model` | `''` | OpenAI 兼容总结端点；缺失则走规则抽取回退 |
| `timeout_seconds` | `45` | 总结请求超时 |
| `max_concurrency` | `2` | 后台总结最大并发 |

### 自作判定门禁（`evaluator.*`）

| 配置项 | 默认 | 说明 |
| :-- | :-- | :-- |
| `enable` | `true` | 是否启用入库前预判定（关闭则原样入库） |
| `model_name` | `jev` | **判官模型名称**；留空回退 `summarizer.model` |
| `api_base` / `api_key` | `''` | 判官端点；留空自动复用 `summarizer.api_base` / `api_key` |
| `timeout_seconds` | `8.0` | 判官请求超时，超时即降级为规则判定 |
| `min_value_score` | `0.35` | 信息价值过滤阈值（低于即丢弃） |
| `fallback_min_value_score` | `0.175` | 降级路径的放宽阈值，避免误伤有效记忆 |
| `conflict_check` | `true` | 启用事实冲突与推翻检测 |
| `conflict_similarity` | `0.5` | 冲突判定的字符二元组相似度下限 |
| `max_candidates` | `8` | 单次判官比对的历史事实上限 |
| `fail_open` | `true` | 判官失败时是否保留事实 |

### 模型（`models.*`）与检索（`retrieval.*`）

| 配置项 | 默认 | 说明 |
| :-- | :-- | :-- |
| `embedding.api_base` / `model` / `dim` | SiliconFlow / `BAAI/bge-m3` / `1024` | 向量模型，维度自动截断/补零对齐 |
| `rerank.api_base` / `model` | SiliconFlow / `BAAI/bge-reranker-v2-m3` | 重排模型，无密钥则本地 RRF |
| `rerank.rrf_k` | `60` | RRF 融合常数 |
| `retrieval.top_k` | `6` | 单次召回/注入条数 |
| `retrieval.inject_position` | `system_head` | **三态注入路由** |
| `retrieval.prompt_injection_max_chars` | `2000` | 注入字符上限 |
| `retrieval.retrieve_timeout` | `3.0` | 前置检索超时（超时即放弃注入） |
| `retrieval.record_dedup_window` | `2.0` | 轮次记录去重时间窗（秒） |
| `retrieval.strict_session_isolation` | `true` | **严格会话隔离**：仅本会话 + `scope_key='global'` 可召回，物理杜绝串台 |
| `retrieval.access_flush_interval` | `60.0` | 访问计数异步刷盘间隔（秒），检索链路保持纯读 |
| `retrieval.access_flush_threshold` | `200` | 访问计数累计达到该条数立即异步刷盘 |

### 迁移（`migration.*`）

| 配置项 | 默认 | 说明 |
| :-- | :-- | :-- |
| `auto_detect_legacy` | `true` | 启动时自动检测旧插件数据目录 |
| `source_db_path` | `''` | 旧版 `livingmemory.db` 路径（留空自动探测） |
| `force` | `false` | 目标库非空仍强制迁移 |
| `dry_run` | `false` | 仅预览统计，不写入 |

---

## 7. 测试与质量保障

四套件分层递进，全部可脱离 AstrBot 宿主独立运行（内置宿主桩包，零外部依赖）：

```bash
python3 tests/test_stage1_buffer.py      # 多轮缓冲打包 / 强制冲洗 / Worker 消费
python3 tests/test_stage2_hooks.py       # 双向钩子装配 / 注入 / 记录 / 降级 / 格式化
python3 tests/test_stage3_commands.py    # 三态注入路由 / 四指令 / 指令组装配
python3 tests/test_stage4_e2e.py         # 全链路端到端闭环 + 冷重启持久化
python3 tests/test_stage5_evaluator_gate.py  # 判定门禁：价值打分 / 冲突废弃 / 超时降级
python3 tests/test_stage6_hardening.py   # 方案A 止血热修：R-01 ~ R-06 六项专项
```

### 覆盖矩阵

| 套件 | 验证重点 |
| :-- | :-- |
| **Stage 1** | 前 N-1 轮留在缓冲、第 N 轮打包 `kind=batch`、`flush_all` 多会话冲洗、空会话返回 0、Worker 消费后 L2 落库 |
| **Stage 2** | 钩子装饰器注册、记忆块前置注入且人设保留、注入幂等、分片/工具轮过滤、重复回调去重、检索异常/超时/缺失全降级、`format_memory_block` 优先/兜底/截断 |
| **Stage 3** | `system_head`/`system_tail`/`user_head` 三态路由与别名归一化、`cmd_status`/`cmd_recall`/`cmd_pin`/`cmd_migrate` 输出与边界零异常、指令组装配 |
| **Stage 4** | ① Worker 启动 → ② 5 轮真实钩子触发 batch → ③ L1 落库 + L2 沉淀 + L3 晋升（DLQ=0）→ ④ 两阶段检索注入 L3+L2 → ⑤ 三指令真实读写联动（`pin` 可逆切换并用 `status` 交叉验证）→ ⑥ `terminate()` 尾部冲洗/Worker 回收/连接关闭/数据无损；外加**冷重启复用同一库仍可召回** |
| **Stage 5** | 判官默认模型 `jev` 与凭证回退、寒暄/纯提问过滤、否定极性翻转与推翻标记识别、判官成功路径（阈值过滤 + 非法 `conflict_id` 过滤）、非法 JSON/连接异常/超时三种降级、`enable=false` 放行、端到端“判官不可用仍入库”、端到端“旧事实软废弃且检索不召回”、融合场景不自我废弃 |
| **Stage 6** | **R-01** 置顶免死（规则层豁免 + SQL 守卫 + 端到端不废弃）、**R-02** L1 实时落盘/`l2_processed` 回写/idle 超时刷盘、**R-03** 5 轮仅 1 次 LLM+1 次门禁+1 次批量 Embedding、**R-04** 默认严格隔离（跨会话不可见、`global` 例外）、**R-05** 检索不写库/异步刷盘/`close()` 刷净、**R-06** Session 复用与幂等 `close()` |

### Stage 4 实测闭环指标

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

---

## 8. 版本演进

| 阶段 | 提交主题 | 交付内容 |
| :-- | :-- | :-- |
| **Stage 1** | `feat(stage1)` | 多轮窗口缓冲打包、异步 Worker 骨架、插件生命周期托管（`initialize` / `terminate`） |
| **Stage 2** | `feat(stage2)` | 双向 LLM 钩子装配（注入 + 记录）、静默降级与幂等保护 |
| **Stage 3** | `feat(stage3)` | 三态动态注入路由、完整 `/memory` 管理指令组（status / recall / pin / migrate） |
| **Stage 4** | `feat(stage4)` | 全链路端到端集成测试、README 文档收口、首个可发布版本 |
| **Stage 5** | `feat(stage5)` | 自作判定门禁：廉价判官 `jev`、价值打分、冲突软废弃与平滑降级 |
| **方案 A** | `fix(hardening)` | 止血热修 R-01 ~ R-06：置顶免死、真批量合并、L1 实时落盘、默认严格隔离、连接池复用、检索纯读 |

---

## 9. 数据安全与隐私

- 全部记忆数据存储于**本地 SQLite**，不上传任何第三方（模型 API 仅发送待提炼的对话文本）；
- 连接启用 `WAL` + `synchronous=NORMAL`，读写并发安全；
- 关闭流程保证 `flush → drain → close` 顺序，**正常停机不丢记忆**；
- 迁移前自动生成安全快照，`dry_run` 可先预览统计再决定是否落库。

---

## 10. 已知边界

以下能力已完成表结构 / 配置项预留，运行期链路将在后续版本接入，**当前不影响 L1/L2/L3 主链路**：

| 能力 | 当前状态 |
| :-- | :-- |
| 知识图谱运行期写入（`kg_nodes` / `kg_edges`） | 表结构与迁移映射已就绪；运行期图谱自动构建待接入（`/memory status` 中图谱计数仅在迁移后非零） |
| 专属 WebUI 面板（`webui.*`） | 配置项已预留；HTTP 面板实现待提供 |
| 检索权重细粒度调节（`retrieval.weight_*`） | 配置项已预留；当前粗排融合为内置 `0.55` 向量 + `0.45` 关键词 |

---

## 11. 许可

随 AstrBot 生态发布。
