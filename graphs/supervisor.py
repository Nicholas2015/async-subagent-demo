import os
from langchain.chat_models import init_chat_model
from deepagents import AsyncSubAgent, create_deep_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

model1 = ChatOpenAI(
    model="qwen3.8:27b",
    base_url=os.getenv("MODELSCOPE_API_URL"),
    api_key=os.getenv("MODELSCOPE_API_KEY")
)

model = init_chat_model(
    base_url=os.getenv("MODELSCOPE_API_URL"),
    api_key=os.getenv("MODELSCOPE_API_KEY"),
    model="qwen3.8:27b",
    temperature=0.5
)

graph = create_deep_agent(
    model=model,
    system_prompt=(
"You are a supervisor agent for an async-subagent demo. "
        "When the user asks for a long-running research task, you must delegate "
        "to the async subagent named researcher immediately. "
        "After calling start_async_task, return the task_id to the user and stop. "
        "Do not call check_async_task unless the user explicitly asks for progress. "
        "If the user asks to revise the background task, call update_async_task."
    ),
    subagents=[
        AsyncSubAgent(
            name="researcher",
            description=(
"Use for any long-running background research or async demo task. "
                "This agent intentionally sleeps before returning so the async "
                "behavior is easy to observe."
            ),
            graph_id="researcher",
        )
    ],
    checkpointer=InMemorySaver()
)
