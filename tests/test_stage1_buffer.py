# -*- coding: utf-8 -*-
"""
Stage 1 极简自测：验证 summarizer 的多轮缓冲打包与强制冲洗行为。

覆盖点：
1. 连续 4 次 enqueue（阈值 5）不触发打包，轮次留在缓冲池；
2. 第 5 次达到阈值，整窗打包入队（turn_count == 5），缓冲池清空；
3. flush_all() 可冲洗多个会话的剩余轮次并清空缓冲池；
4. flush_session() 对无待总结会话返回 0；
5. 启动 Worker 后，batch 事件能被正常消费并写入 L2 记忆表（端到端）。

运行： python3 tests/test_stage1_buffer.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# 允许以脚本方式直接运行：把插件根目录加入 sys.path
PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from storage import MemoryStorage  # noqa: E402
from summarizer import MemorySummarizer  # noqa: E402


class FakeEmbeddingClient:
    """零向量 Embedding 客户端桩，避免测试依赖外部 API。"""

    dim = 8

    async def get_embedding(self, text: str):
        return [0.0] * self.dim


def _new_summarizer(tmp_dir: Path, interval: int = 5) -> MemorySummarizer:
    storage = MemoryStorage(tmp_dir)
    storage.init_schema()
    return MemorySummarizer(
        storage=storage,
        embedding_client=FakeEmbeddingClient(),
        rerank_client=None,
        config={"summary_turn_interval": interval},
    )


async def test_buffering() -> None:
    with tempfile.TemporaryDirectory(prefix="agent_memory_stage1_") as tmp:
        tmp_dir = Path(tmp)
        sm = _new_summarizer(tmp_dir, interval=5)

        # --- 1. 前 4 轮：只进缓冲池，不打包 ---
        for i in range(1, 5):
            ok = await sm.enqueue_dialogue("s1", f"用户第{i}轮说的话", f"机器人第{i}轮回复")
            assert ok is True, f"第 {i} 轮入缓冲失败"
            assert sm.pending_turns("s1") == i, f"第 {i} 轮后缓冲应为 {i}"
            assert sm._queue.qsize() == 0, f"第 {i} 轮不应触发打包"

        print(f"[OK] 前 4 轮均未触发总结, pending={sm.pending_turns('s1')}, queue={sm._queue.qsize()}")

        # --- 2. 第 5 轮：达到阈值，打包入队 ---
        await sm.enqueue_dialogue("s1", "用户第5轮说的话", "机器人第5轮回复")
        assert sm.pending_turns("s1") == 0, "达到阈值后缓冲池应被清空"
        assert sm._queue.qsize() == 1, "第 5 轮应打包出一个 batch 事件"

        event = sm._queue.get_nowait()
        sm._queue.task_done()
        assert event.get("kind") == "batch", f"事件类型应为 batch，实际 {event.get('kind')}"
        assert event.get("turn_count") == 5, f"batch 应含 5 轮，实际 {event.get('turn_count')}"
        assert len(event.get("turns", [])) == 5
        assert event.get("reason") == "threshold"
        print(f"[OK] 第 5 轮触发打包: kind={event['kind']}, turns={event['turn_count']}, reason={event['reason']}")

        # --- 3. 多会话 flush_all ---
        await sm.enqueue_dialogue("s1", "s1 待冲洗 A", "s1 待冲洗 a")
        await sm.enqueue_dialogue("s1", "s1 待冲洗 B", "s1 待冲洗 b")
        await sm.enqueue_dialogue("s2", "s2 待冲洗 C", "s2 待冲洗 c")
        assert sm.pending_turns() == 3, f"flush 前应有 3 轮待总结，实际 {sm.pending_turns()}"

        flushed = await sm.flush_all()
        assert flushed == {"s1": 2, "s2": 1}, f"flush_all 结果异常: {flushed}"
        assert sm.pending_turns() == 0, "flush_all 后缓冲池应清空"
        assert sm._queue.qsize() == 2, f"flush_all 应产生 2 个 batch，实际 {sm._queue.qsize()}"

        while not sm._queue.empty():
            ev = sm._queue.get_nowait()
            sm._queue.task_done()
            assert ev["reason"] == "flush"
        print(f"[OK] flush_all 清空缓冲: {flushed}, queue 已排空")

        # --- 4. 空会话 flush_session ---
        assert await sm.flush_session("not_exists") == 0
        assert await sm.flush_session("") == 0
        print("[OK] flush_session 对无待总结会话返回 0")

        print(f"[OK] 统计快照: {sm.get_buffer_stats()}")

    print("\n=== 缓冲打包/冲洗 用例全部通过 ===")


async def test_worker_consumes_batch() -> None:
    """端到端：启动 Worker 后 batch 被消费，L2 记忆表产生记录。"""
    with tempfile.TemporaryDirectory(prefix="agent_memory_stage1_worker_") as tmp:
        tmp_dir = Path(tmp)
        sm = _new_summarizer(tmp_dir, interval=5)
        sm.start()
        try:
            for i in range(1, 6):
                await sm.enqueue_dialogue("s_worker", f"我第{i}轮很喜欢喝绿茶{i}", f"记住了第{i}轮")
            assert sm._queue.qsize() == 1
            await asyncio.wait_for(sm._queue.join(), timeout=10.0)

            conn = sm.storage.connect()
            row = conn.execute(
                "SELECT COUNT(1) AS cnt FROM l2_atoms WHERE scope_key = ?", ("s_worker",)
            ).fetchone()
            assert row["cnt"] > 0, "Worker 消费 batch 后 L2 记忆表应产生记录"
            print(f"[OK] Worker 消费 batch 完成, l2_atoms 记录数={row['cnt']}")
        finally:
            await sm.stop(flush=False)
        print("=== Worker 端到端用例通过 ===")


def main() -> int:
    asyncio.run(test_buffering())
    asyncio.run(test_worker_consumes_batch())
    return 0


if __name__ == "__main__":
    sys.exit(main())
