# backend/judge_agent.py
"""
"多次调用 · 复盘" 编排层。
------------------------------------------------------------
这是这一版设计的核心改动：判断一条客户消息的意图/情绪，
不再是"调用一次 LLM 就直接采信"，而是：

    1. 独立调用 N 次（默认3次，见 config.MULTI_CALL_VOTES），
       每次调用互相之间没有信息共享——不会告诉第2次调用"第1次结果是什么"，
       保证这N次判断真的是"独立"的，不是同一个结论问3遍。
    2. 对 N 次结果的 intent 做多数投票：
         - 出现严格多数（比如3次里有2次或3次一致）-> 直接采用多数意见；
         - 完全没有多数意见（比如3次给出3个不同的intent）
           -> 触发"复盘仲裁"：把这N次的分歧摊开，让模型重新完整看一遍
              原始消息，给出一个最终裁定（对应 llm_client.py 里的 review()）。
    3. emotion_negative 同理走多数表决；如果出现平局（这只会在N为偶数时发生，
       默认N=3不会），保守地判定为"负面"——宁可多触发一次转人工排查，
       也不要漏掉真实的客户不满。

为什么要这么做（两个收益）：
    - 【可靠性】单次 LLM 调用有随机性（同样的输入，temperature>0时输出可能不同），
      多数投票能把这种随机噪声平滑掉，不会因为运气不好抽到一次误判就直接影响业务决策。
    - 【抗注入】对攻击者来说，要让"这条精心构造的攻击消息"在3次完全独立的调用里
      都稳定拿到对攻击者有利的分类结果，比只需要蒙对1次难得多——
      这是在"多数投票"这个统计机制层面额外叠加的一层防御，
      而且是在 Controller/白名单机制之外的、独立的一层，两者互不替代。

注意：这一层只解决"分类结果准不准/稳不稳"的问题，不改变第4节里
"LLM 输出不含 action 字段、Action 100% 由代码白名单决定"这个核心安全设计——
无论这里投票出来的 intent 是什么，最终能做什么动作，仍然完全由
controller.py 里的固定映射表决定。
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import List
import inspect

from .config import settings
from .llm_client import LLMClient
from .models import ChatMessage, Intent, LLMJudgement


@dataclass
class JudgeResult:
    """把最终裁定结果 + 过程信息一起返回，方便前端调试面板展示"复盘"是否发生过。"""
    final: LLMJudgement
    votes: List[LLMJudgement]
    used_review: bool
    review_reason: str = ""


PERSPECTIVES = [
    (0.0, "按常规语义分类，同时核对客户的真实需求。"),
    (0.5, "重点检查反讽、反话、阴阳怪气、过度礼貌包装的不满和表情符号语气。"),
    (0.9, "假设客户真实情绪可能比字面更糟，结合历史趋势重新判断是否存在失望或敷衍。"),
]


def _classify(client, history, message, temperature, perspective):
    parameters = inspect.signature(client.classify).parameters
    if "temperature" in parameters or "perspective" in parameters:
        return client.classify(
            history, message, temperature=temperature, perspective=perspective
        )
    return client.classify(history, message)


def judge_with_review(
    llm_client: LLMClient,
    history: List[ChatMessage],
    message: str,
    n_calls: int = None,
) -> JudgeResult:
    n = n_calls if n_calls is not None else settings.MULTI_CALL_VOTES
    n = max(1, n)

    # ---- 第一步：N 次独立调用 ----
    votes: List[LLMJudgement] = []
    for index in range(n):
        temperature, perspective = PERSPECTIVES[index % len(PERSPECTIVES)]
        votes.append(_classify(llm_client, history, message, temperature, perspective))

    if n == 1:
        # 兼容单次调用场景（比如单元测试只想验证 Controller 逻辑，不想掺入投票复杂度）
        return JudgeResult(final=votes[0], votes=votes, used_review=False)

    # ---- 第二步：intent 多数投票 ----
    intent_counts = Counter(v.intent for v in votes)
    top_intent, top_count = intent_counts.most_common(1)[0]
    majority_needed = n // 2 + 1

    average_confidence = sum(v.confidence for v in votes) / len(votes)
    low_confidence = average_confidence < settings.REVIEW_CONFIDENCE_THRESHOLD

    if top_count >= majority_needed and not low_confidence:
        # 有多数意见，直接采用，不需要复盘
        final_intent = top_intent
        used_review = False

        neg_votes = sum(1 for v in votes if v.emotion_negative)
        pos_votes = n - neg_votes
        # 情绪表决：多数决定；平局（只可能在 n 为偶数时出现）保守判为"负面"
        final_emotion = neg_votes >= pos_votes if neg_votes == pos_votes else neg_votes > pos_votes

        # draft_reply 取"意见与最终intent一致"的那一票里的草稿，尽量语义匹配
        draft = next((v.draft_reply for v in votes if v.intent == final_intent and v.draft_reply), "")
        if not draft:
            draft = next((v.draft_reply for v in votes if v.draft_reply), "")

        final = LLMJudgement(
            intent=final_intent,
            emotion_negative=final_emotion,
            confidence=top_count / n,
            draft_reply=draft,
            reasoning=next((v.reasoning for v in votes if v.intent == final_intent and v.reasoning), ""),
        )
        return JudgeResult(final=final, votes=votes, used_review=used_review)

    # ---- 第三步：没有多数意见 -> 复盘仲裁 ----
    final = llm_client.review(history, message, votes)
    reason = "低置信度" if low_confidence and top_count >= majority_needed else "投票无严格多数"
    return JudgeResult(final=final, votes=votes, used_review=True, review_reason=reason)
