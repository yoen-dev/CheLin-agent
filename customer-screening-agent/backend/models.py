# backend/models.py
"""
数据模型层。
------------------------------------------------------------
这个文件里定义的枚举（Enum）是整个安全设计的地基之一：

    LLM 只能从这几个"选项"里选一个来分类，不存在"LLM 输出一个
    我们没预料到的动作"这种可能——因为 Action 根本不是 LLM 的输出字段，
    Action 是 Controller 根据 (intent, emotion, state) 用固定规则算出来的。

也就是说：即使攻击者成功"忽悠" LLM 把某句话分类错了（比如把辱骂分类成
interested），最坏结果也只是分类错误，而不可能凭空产生一个不在
ActionType 里的动作。这是回答"约束3怎么在代码层面 100% 强制"的核心论据。
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, constr


class Intent(str, Enum):
    """客户消息的意图分类——LLM 判断结果的合法取值集合，仅这 5 种。"""
    INTERESTED = "interested"           # 有兴趣
    NEED_MORE_INFO = "need_more_info"   # 需要更多信息
    REJECT = "reject"                   # 明确拒绝
    IRRELEVANT = "irrelevant"           # 答非所问
    OTHER = "other"                     # 其他


class ActionType(str, Enum):
    """
    系统允许执行的动作——白名单，一共 4 种，写死在这里。
    Executor（执行层）只认这 4 个值，出现任何其他字符串一律拒绝执行。
    """
    REPLY = "reply"
    SCHEDULE_FOLLOWUP = "schedule_followup"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    MARK_NOT_INTERESTED = "mark_not_interested"


class ConversationState(str, Enum):
    """会话级别的状态机状态，只有两种。"""
    ACTIVE = "active"        # 正常自动运转
    ESCALATED = "escalated"  # 已转人工，全程静默，直到人工重新激活


class LLMJudgement(BaseModel):
    """
    这是 LLM 单次调用被要求返回的结构化输出（对应 judge_agent.py）。
    注意：这里面**没有 action 字段**——这是有意为之。
    LLM 只负责"理解"，不负责"决策"，Controller 才决定 action。

    这正是用户你自己提的思路："写成一个 agent，接收消息后，
    把意图判断和情绪判断这两件事一次性、一起判断出来"——
    比拆成两个独立 Agent 分两次调用更省 token、更省延迟，
    而且两个判断基于同一次上下文推理，逻辑上也更一致
    （不会出现"意图 Agent 和情绪 Agent 各看各的，互相矛盾"的情况）。
    """
    intent: Intent = Field(description="客户消息的主要意图分类")
    emotion_negative: bool = Field(description="客户这条消息是否表现出明显不满情绪")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="模型对分类结果的置信度")
    draft_reply: str = Field(
        default="",
        description="如果后续决定要回复客户，这是建议的回复草稿；"
                    "是否真的会被发送，由 Controller 和限流器决定，不由 LLM 决定。",
    )
    reasoning: str = Field(
        default="",
        description="供内部复盘使用的简短判断依据，不直接展示给客户。",
    )


class ChatMessage(BaseModel):
    """一条对话记录，用于会话历史。"""
    role: str          # "customer" | "agent" | "system_note"
    content: str
    timestamp: float


class ChatRequest(BaseModel):
    """前端发来的一条客户消息，先在 HTTP 边界拒绝空值和异常长输入。"""
    # 限制输入长度可以避免无意义的超长历史污染 LLM 上下文，
    # 也避免客户 ID 变成无限增长的内存字典键。
    customer_id: constr(strip_whitespace=True, min_length=1, max_length=128)
    message: constr(strip_whitespace=True, min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    """返回给前端的结果，方便前端把内部决策过程展示出来（调试/演示用）。"""
    action: ActionType
    state: ConversationState
    reply_text: Optional[str] = None
    intent: Optional[Intent] = None
    emotion_negative: Optional[bool] = None
    bad_streak: int = 0
    rate_limited: bool = False
    votes_count: int = 0        # 本次独立调用了几次 LLM 做判断（多次调用复盘机制）
    used_review: bool = False   # 是否因为N次判断出现分歧，触发了复盘仲裁
    note: str = ""   # 人类可读的说明，比如"因限流被降级为 schedule_followup"


class SessionSnapshot(BaseModel):
    """给前端调试面板用的会话快照。"""
    customer_id: str
    state: ConversationState
    bad_streak: int
    history: List[ChatMessage]
