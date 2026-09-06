# backend/session_store.py
"""
会话状态存储。
------------------------------------------------------------
这是整个安全设计里"状态"的唯一真相来源（single source of truth）。

刻意做成一个简单的内存字典 + 锁，而不是一开始就上数据库，原因写在 README
的"取舍"部分——这是一个 demo 级别的 agent 演示项目，不是题目一那种要求
"任意数量 worker 并发抢占同一条数据库记录"的场景，同一个客户的消息
在真实业务里也是顺序到达、顺序处理的，不需要处理"多个进程同时处理
同一个客户的同一条消息"这种并发场景，所以用进程内锁而不是数据库事务，
是刻意的取舍，不是遗漏。

每个 CustomerSession 只存"决策需要的最小状态"：
    - state：ACTIVE / ESCALATED（约束2的状态机核心）
    - bad_streak：连续异常计数器（约束2）
    - history：对话历史，供 LLM 判断上下文用
"""

import threading
import time
from typing import Dict, List

from .models import ChatMessage, ConversationState


class CustomerSession:
    def __init__(self, customer_id: str):
        self.customer_id = customer_id
        self.state: ConversationState = ConversationState.ACTIVE
        self.bad_streak: int = 0
        self.history: List[ChatMessage] = []
        # 公开属性而不是 _lock：Controller 需要跨多步操作(判断+改状态+发送)
        # 持有同一把锁，做成私有反而逼着外部用 noqa 绕过静态检查，不如大方公开。
        self.lock = threading.Lock()

    def add_message(self, role: str, content: str) -> None:
        self.history.append(ChatMessage(role=role, content=content, timestamp=time.time()))

    def snapshot_history(self) -> List[ChatMessage]:
        return list(self.history)


class SessionStore:
    """
    简单的内存存储 + 全局锁。
    对同一个 customer_id 的读-判断-写全过程加锁，
    避免"人工恢复"和"客户消息处理"这两个操作并发时把状态改乱
    （比如人工正在把 ESCALATED 改回 ACTIVE 的同时，一条旧的客户消息
    也在处理，二者交错写状态）。
    """

    def __init__(self):
        self._sessions: Dict[str, CustomerSession] = {}
        self._store_lock = threading.Lock()

    def get_or_create(self, customer_id: str) -> CustomerSession:
        with self._store_lock:
            if customer_id not in self._sessions:
                self._sessions[customer_id] = CustomerSession(customer_id)
            return self._sessions[customer_id]

    def all_customer_ids(self) -> List[str]:
        with self._store_lock:
            return list(self._sessions.keys())


session_store = SessionStore()
