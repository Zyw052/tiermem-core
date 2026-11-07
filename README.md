# astrbot_plugin_agent_memory

**智涌原生分层记忆插件**（L1 工作区 / L2 情景原子 / L3 知识图谱与画像 + 独立总结模型 + 旧资产无缝迁移 + 专属 WebUI）

## 1. 架构概览

本插件为 AstrBot 提供结构化、可观测、可迁移的三层记忆体系：

- **L1 工作区记忆 (`l1_docs`)**：维护会话级活跃上下文、主人核心指令、近期焦点摘要，直接注入 System Prompt。
- **L2 情景原子记忆 (`l2_atoms`)**：将对话与事件切分为独立事实原子，支持 FTS5/BM25 关键词检索 + 向量语义检索 + 重要度时间衰减混合排序。
- **L3 实体图谱与画像 (`kg_nodes` / `kg_edges`)**：沉淀长期人物关系、偏好、约定与项目实体拓扑。
- **异步死信队列 (`task_dlq`)**：保障独立总结模型在网络抖动或限流时不丢失记忆提炼任务，支持后台重试。

## 2. 模块划分

```text
astrbot_plugin_agent_memory/
├── metadata.yaml        # 插件元信息声明
├── _conf_schema.json    # 插件配置 Schema（独立总结模型、混合检索权重、WebUI 开关等）
├── main.py              # 插件入口 Star 类、生命周期钩子与指令/工具注册
├── storage.py           # SQLite 分层存储引擎（DDL: l1_docs, l2_atoms, kg_nodes, kg_edges, task_dlq）
└── README.md            # 架构说明与开发文档
```

## 3. 版本演进

- **v1.0.0**：初始化分层存储骨架、配置 Schema 与插件主生命周期。
