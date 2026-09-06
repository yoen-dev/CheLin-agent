# backend/controller.py
"""
决策控制器（Decision Controller）。
------------------------------------------------------------
整个 agent 里"谁说了算"的问题，答案就在这个文件：LLM 只产出
(intent, emotion_negative, draft_reply)，从这三个值 + 当前会话状态
算出最终 Action，全部由下面 process_message() 这一个函数里的
【确定性代码】完成，不存在"LLM 自己说要执行哪个动作，我们照着执行"
这种写法。这也是回答"约束2/3具体是靠什么机制保证"这个追问时
要指着讲的核心函数。

处理一条客户消息的完整流程：
    0. 静默门禁：如果已经 ESCALATED，直接返回静默，完全不调用 LLM。
    1. 调用 judge_agent 拿到 (intent, emotion_negative, draft_reply)。
    2. 更新连续异常计数器 bad_streak（约束2的状态机，纯代码判断）。
    3. 如果 bad_streak 达到阈值 -> 强制 action=ESCALATE_TO_HUMAN，
       state=ESCALATED，这一步不看 LLM 的"意愿"，是硬编码规则。
    4. 否则，用一张固定的 intent -> action 映射表决定 action
       （约束3的白名单机制）。
    5. 如果 action 是 REPLY，才需要过限流器这一关：
       通过 -> 真正"发送"，记录时间戳；
       不通过 -> 【降级】成 SCHEDULE_FOLLOWUP，不发送任何消息，
       并在 note 里说明原因（这就是约束1"最终真正发出去的动作
       必须被卡住"的具体实现：不管上游想不想发，这里才是唯一
       真正调用"发送"的地方）。
    6. 发送前对 draft_reply 跑一遍 safety_guard.filter_output
       （约束4的第二层防线）。
"""

import time
from typing import Optional, Tuple

from .judge_agent import judge_with_review
from .llm_client import LLMClient
from .models import ActionType, ChatResponse, ConversationState, Intent
from .rate_limiter import rate_limiter
from .safety_guard import filter_output, is_conversation_silenced
from .session_store import CustomerSession, session_store
from .config import settings

# ----------------------------------------------------------------------
# 意图 -> 动作 的固定映射表。这是"白名单"的具体体现：
# 无论 LLM 的 draft_reply 里写了什么、无论客户消息怎么诱导，
# 能选中的 action 永远只在这张表的取值范围内（REPLY / SCHEDULE_FOLLOWUP /
# MARK_NOT_INTERESTED，外加下面单独硬编码的 ESCALATE_TO_HUMAN）。
# ----------------------------------------------------------------------
INTENT_ACTION_MAP = {
    Intent.INTERESTED: ActionType.REPLY,
    Intent.NEED_MORE_INFO: ActionType.REPLY,
    Intent.REJECT: ActionType.MARK_NOT_INTERESTED,
    Intent.IRRELEVANT: ActionType.REPLY,   # 先礼貌澄清一次，真正连续两次才转人工
    Intent.OTHER: ActionType.REPLY,
}


def _is_bad_signal(intent: Intent, emotion_negative: bool) -> bool:
    """约束2定义的"异常"：答非所问 或 情绪不满，命中任意一个就算。"""
    return intent == Intent.IRRELEVANT or emotion_negative


def process_message(
    customer_id: str,
    message: str,
    llm_client: LLMClient,
    n_calls: Optional[int] = None,
) -> ChatResponse:
    """
    n_calls: 覆盖 config.MULTI_CALL_VOTES 的调用次数。
        - 生产环境不传，走默认的"多次调用+复盘"（见 judge_agent.py）。
        - 单元测试里想单独验证状态机逻辑、不想掺入投票复杂度时，
          可以显式传 n_calls=1，退化成"单次调用直接采信"。
    """
    session: CustomerSession = session_store.get_or_create(customer_id)

    with session.lock:
        # ---------- 0. 静默门禁：ESCALATED 状态下完全不调用 LLM ----------
        if is_conversation_silenced(session):
            return ChatResponse(
                action=ActionType.ESCALATE_TO_HUMAN,
                state=session.state,
                reply_text=None,
                bad_streak=session.bad_streak,
                note="会话已转人工，agent 保持静默，未调用 LLM。需人工在 /admin/reactivate 重新激活。",
            )

        session.add_message("customer", message)

        # ---------- 1. 多次独立调用 + 必要时复盘仲裁（唯一负责"理解"的环节）----------
        judge_result = judge_with_review(llm_client, session.snapshot_history()[:-1], message, n_calls)
        judgement = judge_result.final

        # ---------- 2. 更新连续异常计数器（纯代码，不受 LLM 影响）----------
        if _is_bad_signal(judgement.intent, judgement.emotion_negative):
            session.bad_streak += 1
        else:
            session.bad_streak = 0

        note_parts = []
        if judge_result.used_review:
            vote_summary = ", ".join(v.intent.value for v in judge_result.votes)
            note_parts.append(f"{len(judge_result.votes)}次独立判断出现分歧({vote_summary})，"
                               f"已触发复盘仲裁得到最终结果。")

        # ---------- 3. 达到阈值 -> 强制转人工，覆盖一切 ----------
        if session.bad_streak >= settings.ESCALATE_AFTER_CONSECUTIVE_BAD:
            session.state = ConversationState.ESCALATED
            session.add_message("agent", "[系统] 已转人工，agent 静默")
            return ChatResponse(
                action=ActionType.ESCALATE_TO_HUMAN,
                state=session.state,
                reply_text=None,
                intent=judgement.intent,
                emotion_negative=judgement.emotion_negative,
                bad_streak=session.bad_streak,
                votes_count=len(judge_result.votes),
                used_review=judge_result.used_review,
                note=f"连续 {session.bad_streak} 次异常（答非所问/情绪不满），触发强制转人工。",
            )

        # ---------- 4. 白名单映射表决定 action ----------
        action = INTENT_ACTION_MAP.get(judgement.intent, ActionType.SCHEDULE_FOLLOWUP)

        reply_text = None
        rate_limited = False

        if action == ActionType.REPLY:
            # ---------- 5. 限流器：唯一真正"放行发送"的关口 ----------
            if rate_limiter.allow(customer_id):
                safe_text, was_filtered = filter_output(
                    judgement.draft_reply or "感谢您的消息，我们已收到。"
                )
                reply_text = safe_text
                if was_filtered:
                    note_parts.append("回复内容命中敏感信息过滤，已替换为安全话术。")
                session.add_message("agent", reply_text)
            else:
                # 限流命中：降级为 schedule_followup，绝不因为"急着回"就绕过限流
                action = ActionType.SCHEDULE_FOLLOWUP
                rate_limited = True
                wait = rate_limiter.remaining_seconds(customer_id)
                note_parts.append(f"命中 60 秒滑动窗口限流，降级为 schedule_followup，"
                                   f"约 {wait:.1f} 秒后可再次发送。")

        elif action == ActionType.MARK_NOT_INTERESTED:
            session.add_message("agent", "[系统] 客户标记为不感兴趣，会话结束")

        return ChatResponse(
            action=action,
            state=session.state,
            reply_text=reply_text,
            intent=judgement.intent,
            emotion_negative=judgement.emotion_negative,
            bad_streak=session.bad_streak,
            rate_limited=rate_limited,
            votes_count=len(judge_result.votes),
            used_review=judge_result.used_review,
            note=" ".join(note_parts),
        )


def reactivate(customer_id: str) -> ConversationState:
    """人工重新激活：唯一能把 ESCALATED 改回 ACTIVE 的入口，且只能人工调用。"""
    session = session_store.get_or_create(customer_id)
    with session.lock:
        session.state = ConversationState.ACTIVE
        session.bad_streak = 0
        session.add_message("system_note", "[系统] 人工已重新激活会话")
        return session.state
