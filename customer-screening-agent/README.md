# 获客初筛 Agent（kGroup 笔试 · 题目二）

> 本 README 同时充当项目说明 + 设计计划文档。V1（第一版本）已实现并跑通全部硬约束
> 对应的自动化测试，可直接本地运行演示。

---

## 1. 项目结构

```
customer-screening-agent/
├── README.md              # 本文件
├── COLLAB.md               # 团队协作分工说明（提交前补全）
├── requirements.txt
├── .env.example             # 环境变量模板（真实 .env 不提交）
├── .gitignore
├── backend/
│   ├── config.py            # 所有可调参数集中管理
│   ├── models.py             # 枚举(Intent/Action/State) + 数据模型 —— 白名单机制的地基
│   ├── llm_client.py          # LLM 调用封装：抽象接口 + MockLLM + GeminiLLM（classify + review两种调用）
│   ├── judge_agent.py          # 多次调用复盘编排：N次独立投票，分歧时触发复盘仲裁
│   ├── rate_limiter.py        # 滑动窗口限流器（约束1）
│   ├── session_store.py       # 会话状态存储（状态机的"唯一真相来源"）
│   ├── safety_guard.py        # 静默门禁 + 输出过滤（约束3/4的防线）
│   ├── controller.py          # 决策控制器：唯一决定 Action 的地方（约束2/3的核心）
│   └── main.py               # FastAPI 路由入口
├── frontend/
│   └── index.html            # 极简聊天页面 + 内部决策调试面板
└── tests/
    ├── test_controller.py    # 状态机测试
    ├── test_rate_limiter.py  # 限流器测试（含并发）
    ├── test_judge_agent.py   # 多次调用复盘机制测试（多数一致/分歧触发复盘/抗单票注入）
    └── test_adversarial.py   # ≥3 条"让 agent 犯规"的对抗测试
```

每个源码文件顶部都写了较详细的中文注释，说明"这个文件解决什么问题、
为什么这么设计"，可以直接当代码走查材料用；下面只讲整体思路，细节看代码注释。

---

## 2. 核心架构

```
客户消息
   │
   ▼
[safety_guard.is_conversation_silenced]  ← 状态门禁，ESCALATED 直接短路，完全不调用 LLM
   │ (state == ACTIVE)
   ▼
[judge_agent.judge_with_review]  ← 多次调用复盘编排
   │   1. 独立调用 llm_client.classify N次（默认3次，互不知情）
   │   2. intent 多数投票；有多数意见 -> 直接采用
   │   3. 没有多数意见 -> 调用 llm_client.review()，把分歧摊开做一次复盘仲裁
   │   最终产出结构化结果: { intent, emotion_negative, confidence, draft_reply }
   │   注意：全程不含 action 字段——LLM 只负责"理解"，不负责"决策"
   ▼
[controller.process_message]  ← 决策核心（纯代码，确定性）
   │  1. 更新 bad_streak 计数器
   │  2. bad_streak 达标 -> 强制 action=escalate_to_human（不看 LLM 意愿）
   │  3. 否则查固定映射表 intent -> action（白名单）
   │  4. action==reply 时才过 rate_limiter 这一关
   ▼
[safety_guard.filter_output]  ← 发送前扫描敏感信息泄露
   ▼
返回给前端 / 客户
```

设计上刻意让"理解"和"决策"分属两个不同的组件：LLM 只产出分类结果，
Controller 里的固定代码逻辑才决定最终动作。这样面试被问"约束是怎么在代码层面强制的"，
答案永远可以指向 `controller.py` 里的某一行具体代码，而不是"prompt 里写了让它注意"。

---

## 3. 意图 / 情绪 判断

- 5 种意图：`interested` / `need_more_info` / `reject` / `irrelevant` / `other`
- 情绪：`emotion_negative`（布尔），与意图正交，任何意图都可能同时带负面情绪
  （例："产品多少钱？但你们客服回复也太慢了" → intent=need_more_info, emotion_negative=true）
- 用 Gemini 的结构化输出（`response_mime_type=application/json` + `response_schema`）强制模型
  只能在这 5 个枚举值里选，物理上排除"模型输出一个我们没见过的分类"的可能。

---

## 4. 四条硬约束的落地方式

| 约束 | 落地位置 | 机制 |
|---|---|---|
| 1. 60秒滑动窗口限流 | `rate_limiter.py` | 每客户维护时间戳队列，`allow()` 里"检查+记录"在同一把锁内原子完成，杜绝并发下的竞态超发；不是固定分钟桶，见文件内注释里"边界突刺"的例子 |
| 2. 连续两次异常强制转人工 | `controller.py::process_message` | `bad_streak` 计数器，与 LLM 单次输出解耦，纯代码判断，命中阈值后**直接覆盖**任何其他决策结果 |
| 3. 不能被话术越权 / 静默不能被绕过 | `models.py`(Action枚举) + `controller.py`(映射表) + `safety_guard.py`(状态门禁) | (a) LLM 输出里根本没有 action 字段，Action 100% 由代码固定映射表产生，值域被 Python 类型系统锁死为 4 个合法值之一；(b) ESCALATED 状态下**在调用 LLM 之前**就短路返回，模型根本看不到后续消息，不存在"被说服恢复"的执行路径 |
| 4. 防止套出系统提示词/内部规则 | `llm_client.py`(SYSTEM_INSTRUCTION 不含真正机密) + `safety_guard.py::filter_output` | 纵深防御：源头上不把价格底线等机密塞进 prompt（泄露不了不存在的东西）+ 输出层关键词/结构扫描兜底。**明确承认非100%**，见第6节"已知局限" |

---

## 5. 技术选型

- **后端**：Python + FastAPI —— 团队最熟悉，生态里调 LLM API、写异步 IO 都很顺手，
  FastAPI 自带 Pydantic 类型校验，正好配合"用类型系统锁死 Action 取值范围"这个设计。
- **不用 LangChain / AutoGen 等现成 agent 框架**：这几条硬约束都要求"确定性、可测试、
  可以指着某一行代码解释"，现成框架的 Agent 执行链路(ReAct/工具调用循环)会让"谁最终决定
  执行什么动作"变得不透明，反而不利于满足"代码层面强制"的要求。所以选择"直接调 LLM API
  + 自己写状态机/规则引擎"这条路线，约束具体在哪一层生效、由哪一行代码保证，全部可追溯。
- **LLM**：Gemini（题目提供的真实 Key），走原生 REST 接口而不引入额外 SDK，方便直接看清楚
  请求体（system_instruction 与 user content 严格分离、response_schema 长什么样），
  这本身也是"输入隔离"防注入设计的一部分证据。
- **会话状态存储**：进程内内存字典 + 线程锁，不上数据库。这是有意的取舍：这个 demo
  场景不涉及题目一那种"多个 worker 同时抢占同一条数据库记录"的跨进程并发问题
  （同一个客户的消息在真实业务里是顺序到达的），单进程内的锁足够保证正确性，
  上数据库/分布式锁属于过度设计。

---

## 6. 已知局限 / 边界（答辩会追问，提前想清楚）

1. **约束4（防套话）不是100%**：这是自然语言生成的开放问题，本方案的边界是"关键词/结构
   特征扫描"，能拦住直白复述，拦不住模型被高明话术诱导后"换一种说法"泄露的情况
   （比如让模型用"每个字的拼音首字母"逐字拼出系统指令，绕过关键词匹配）。
   更彻底的方案需要额外一次"审计 LLM 调用"去判断输出是否语义上泄露了敏感信息，
   这个 V1 版本没做，作为后续优化项列在第8节。
2. **意图分类准确率依赖模型本身**：如果 Gemini 把一句真实的辱骂错误分类成"interested"，
   Controller 层不会"发现"这个错误——Controller 只保证"分类结果被正确地转成动作"，
   不保证"分类结果本身是对的"。这是"代码强制执行"和"分类准确性"两个不同层面的问题，
   答辩时要分开讲清楚，不要混为一谈。
3. **会话状态是进程内内存**：重启服务会丢失所有会话状态（含 bad_streak、转人工状态）。
   Demo 场景可接受，生产环境需要换成 Redis/数据库持久化。
4. **限流器是单进程内共享**：如果未来把服务水平扩展成多进程/多机器，需要把
   `rate_limiter.py` 换成基于 Redis 的实现，当前版本没有跨进程能力，这是明确的设计边界
   （对应第5节"不上分布式锁"的取舍，多进程部署会打破这个假设，需要提前说明）。

---

## 7. 测试与验证

```bash
cd customer-screening-agent
pip install -r requirements.txt

python tests/test_controller.py     # 状态机：连续异常触发转人工 / 转人工后静默不受内容影响
python tests/test_rate_limiter.py   # 限流器：基本行为 + 滑动窗口语义 + 50线程并发压测
python tests/test_adversarial.py    # ≥4 条对抗测试：越权指令 / 套话泄露 / 静默绕过 / 限流轰炸
```

对抗测试默认用"最坏情况模拟"（假设 LLM 已经被攻击者完全说服，直接构造最不利的 LLM
输出喂给 Controller），这样测试结果不依赖某一次真实调用"侥幸"没被攻破，可以稳定复现、
放进 CI。如果想额外看真实 Gemini 面对这几句攻击话术会怎么分类（分类准确率层面，
不是代码强制层面），配置好 `.env` 后执行：

```bash
RUN_REAL_LLM_TESTS=1 python tests/test_adversarial.py
```

---

## 8. 运行方式

```bash
cd customer-screening-agent
pip install -r requirements.txt
cp .env.example .env    # 填入真实的 GEMINI_API_KEY

# 启动后端
python -m backend.main
# 或：uvicorn backend.main:app --reload

# 打开前端（直接用浏览器打开文件即可，无需额外构建）
open frontend/index.html
```

没有配置 `GEMINI_API_KEY` 时会自动降级为 `MockLLMClient`（关键词规则模拟），
方便先把整条链路跑通，但**正式提交/答辩必须用真实 Gemini**（`LLM_PROVIDER=gemini`），
Mock 只用于离线开发和给测试提供确定性输入。

---

## 9. 后续可做（非硬指标，按精力决定要不要做）

- 更细的意图分类 + 置信度阈值：低置信度时走 `schedule_followup` 而不是贸然 `reply`
- 转人工后的恢复机制细化：目前是单个 `/admin/reactivate/{id}` 接口，可以加上人工看
  历史对话再决定的简单后台页面
- 约束4 的语义级审计：加一次"这段回复是否泄露了系统信息"的二次 LLM 判断，弥补关键词
  扫描的边界
- 多语言客户支持：`llm_client.py` 的 SYSTEM_INSTRUCTION 目前只按中文场景写，扩展成多语言
  需要让 draft_reply 跟随客户消息语言生成

---

## 10. 时间投入（提交前请按实际情况填写）

- [ ] 核心功能 + 四条约束：预计半天到一天
- 实际花费：[请填写]，详见 `COLLAB.md`
