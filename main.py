# -*- coding: utf-8 -*-
"""
astrbot_plugin_agent_memory - 插件主入口骨架
负责生命周期管理、配置加载、分层存储初始化与基础状态指令。
"""

import os
from pathlib import Path
from typing import Optional

from astrbot.api.star import Context, Star, register
from astrbot.api import logger

from .storage import MemoryStorage


def _resolve_plugin_data_dir(plugin_name: str = "astrbot_plugin_agent_memory") -> Path:
    """动态解析插件持久化数据目录，避免写死路径"""
    try:
        from astrbot.core.utils.astrbot_path import get_astrbot_data_path
        base_data = Path(get_astrbot_data_path())
        return base_data / "plugin_data" / plugin_name
    except Exception:
        # 回退到基于当前文件位置的相对路径解析
        plugin_dir = Path(__file__).resolve().parent
        data_root = plugin_dir.parent.parent
        return data_root / "plugin_data" / plugin_name


@register(
    "astrbot_plugin_agent_memory",
    "Vivian & Master",
    "智涌原生分层记忆插件 (L1工作区/L2情景/L3画像 + 独立总结模型 + 资产无缝迁移 + 专属WebUI)",
    "1.0.0",
    ""
)
class AgentMemoryPlugin(Star):
    """智涌原生分层记忆插件主类"""

    def __init__(self, context: Context, config: Optional[dict] = None) -> None:
        super().__init__(context)
        self.config = config if isinstance(config, dict) else {}
        self.data_dir: Path = _resolve_plugin_data_dir("astrbot_plugin_agent_memory")
        self.storage: Optional[MemoryStorage] = None

        try:
            self.storage = MemoryStorage(self.data_dir)
            self.storage.init_schema()
            logger.info(
                "[astrbot_plugin_agent_memory] 插件骨架初始化完成，数据目录: %s",
                self.data_dir,
            )
        except Exception as exc:
            logger.error(
                "[astrbot_plugin_agent_memory] 初始化存储引擎失败: %s",
                exc,
                exc_info=True,
            )

    async def initialize(self) -> None:
        """异步生命周期启动钩子"""
        logger.info("[astrbot_plugin_agent_memory] 异步生命周期钩子 initialize() 已执行。")

    async def terminate(self) -> None:
        """插件卸载或停止时的清理钩子"""
        try:
            if self.storage is not None:
                self.storage.close()
            logger.info("[astrbot_plugin_agent_memory] 插件资源已安全释放。")
        except Exception as exc:
            logger.error(
                "[astrbot_plugin_agent_memory] 释放插件资源异常: %s",
                exc,
                exc_info=True,
            )
