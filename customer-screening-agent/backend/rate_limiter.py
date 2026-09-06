# backend/rate_limiter.py
"""
限流器（对应硬约束 1）。
------------------------------------------------------------
题目特意强调："任意 60 秒窗口" 和 "每分钟一个固定窗口" 不是一回事。

区别举例：
    固定窗口（错误做法）：以整分钟为边界，比如 10:00:00-10:00:59 算一个窗口。
    如果在 10:00:59 发一条，10:01:00 又立刻能发一条——两条消息间隔只有 1 秒，
    但因为跨越了"窗口边界"，固定窗口计数器会天真地认为这是合法的。这就是
    经典的"边界突刺"问题。

    滑动窗口（本文件实现）：不看"消息落在哪个整分钟桶里"，
    只看"过去连续 60 秒内已经发送过几条"。不存在可以被利用的边界。

实现方式：每个客户维护一个时间戳队列（deque），
每次要发消息前，先把队列里"已经超过 60 秒的旧时间戳"全部弹出，
再看剩下的数量是否 >= 上限。

注意：这里限的是"真正发送出去的动作"，不是"LLM 被调用的次数"。
所以调用点在 controller.py 里"决定要不要真的把 reply 发出去"的那一刻，
而不是在"要不要调用 LLM 做分类"的那一刻——分类可以随便调用，
但最终往客户那边发消息这个动作必须过这一关。这也是为什么
即使 LLM 内部重试了 N 次、或者一次消息触发了多次工具调用，
最终能不能真的发出去，都只取决于这里的时间戳队列状态，
和 LLM 决定调用几次完全无关。
"""

import threading
import time
from collections import deque
from typing import Dict, Deque

from .config import settings


class SlidingWindowRateLimiter:
    def __init__(self, window_seconds: int = None, max_messages: int = None):
        self.window_seconds = window_seconds or settings.RATE_LIMIT_WINDOW_SECONDS
        self.max_messages = max_messages or settings.RATE_LIMIT_MAX_MESSAGES
        self._sent_at: Dict[str, Deque[float]] = {}
        # 用一把全局锁保护这个内存结构。系统规模很小（agent场景不涉及题目一
        # 那种多进程 worker 抢任务），单进程内的线程锁就足够保证正确性；
        # 如果未来要多进程/多机部署，这里需要换成 Redis 之类的共享存储，
        # 但当前场景没有这个必要，属于"够用就好，不过度设计"。
        self._lock = threading.Lock()

    def _evict_old(self, q: Deque[float], now: float) -> None:
        while q and now - q[0] > self.window_seconds:
            q.popleft()

    def allow(self, customer_id: str) -> bool:
        """
        检查是否允许发送，如果允许，【原子地】记录这次发送时间戳。
        必须在同一次加锁里完成"检查 + 记录"，否则并发请求下会出现
        两个请求都读到"还没超限"、结果一起发出去，导致真正超限
        （这是经典的 check-then-act 竞态问题，即使这里请求量很小，
        写的时候也应该养成不留这种漏洞的习惯）。
        """
        now = time.monotonic()
        with self._lock:
            q = self._sent_at.setdefault(customer_id, deque())
            self._evict_old(q, now)
            if len(q) >= self.max_messages:
                return False
            q.append(now)
            return True

    def remaining_seconds(self, customer_id: str) -> float:
        """调试/展示用：还要多久窗口才会空出名额。"""
        now = time.monotonic()
        with self._lock:
            q = self._sent_at.get(customer_id)
            if not q:
                return 0.0
            self._evict_old(q, now)
            if len(q) < self.max_messages:
                return 0.0
            return max(0.0, self.window_seconds - (now - q[0]))


rate_limiter = SlidingWindowRateLimiter()
