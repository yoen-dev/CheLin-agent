# backend/main.py
"""
FastAPI 应用入口。
------------------------------------------------------------
只做"接口路由"这一件事，不写业务逻辑——业务逻辑全部在 controller.py。
这样答辩时如果被问"这条 HTTP 接口具体做了什么校验"，你可以直接说
"路由层几乎不做校验/决策，都在 controller，这里只是把 HTTP 请求
转成函数调用"，边界很清楚。

提供的接口：
    POST /chat                客户发一条消息，返回 agent 的决策结果
    GET  /session/{id}        查看某个客户当前的会话快照（调试/演示用）
    POST /admin/reactivate/{id}  人工重新激活某个已转人工的会话
    GET  /health               健康检查
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .controller import process_message, reactivate
from .llm_client import build_llm_client
from .models import ChatRequest, ChatResponse, SessionSnapshot
from .session_store import session_store

app = FastAPI(title="获客初筛 Agent - Demo")

# 允许本地简单的 index.html（file:// 或 127.0.0.1 任意端口）直接跨域调用，
# 仅用于本地演示，生产环境应该收紧成具体域名。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局只创建一次 LLM 客户端（内部会根据 config.LLM_PROVIDER 决定用 Gemini 还是 Mock）
llm_client = build_llm_client()


@app.get("/health")
def health():
    return {"status": "ok", "llm_provider": type(llm_client).__name__}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """客户发一条消息，agent 处理并返回决策结果。"""
    return process_message(req.customer_id, req.message, llm_client)


@app.get("/session/{customer_id}", response_model=SessionSnapshot)
def get_session(customer_id: str):
    """查看某个客户当前状态，前端调试面板用。"""
    session = session_store.get_or_create(customer_id)
    return SessionSnapshot(
        customer_id=customer_id,
        state=session.state,
        bad_streak=session.bad_streak,
        history=session.snapshot_history(),
    )


@app.post("/admin/reactivate/{customer_id}")
def admin_reactivate(customer_id: str):
    """
    人工重新激活接口。
    刻意做成一个单独的、明确标注 /admin/ 前缀的接口，而不是让客户对话内容
    有任何机会触发这个效果——这也是约束3"静默不能被绕过"的另一半证据：
    唯一能把 ESCALATED 改回 ACTIVE 的代码路径，就是这里，
    和 /chat 接口（客户消息处理路径）完全隔离，客户端消息永远走不到这里。
    """
    new_state = reactivate(customer_id)
    return {"customer_id": customer_id, "state": new_state}


# 让前端和 API 共用一个来源，远程开发环境中不会把容器地址误当成本机地址。
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    from .config import settings

    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
