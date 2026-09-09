import os

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from deepagents import FilesystemPermission
from deepagents.backends import FilesystemBackend


from dotenv import load_dotenv
from langgraph.prebuilt.tool_node import ToolCallRequest

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("MODELSCOPE_MODEL_NAME"),
    base_url=os.getenv("MODELSCOPE_BASE_URL"),
    api_key=os.getenv("MODELSCOPE_API_KEY"),
)

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent

@tool
def delete_file(path: str) -> str:
    """删除指定文件。"""
    resolved = (PROJECT_ROOT / path).resolve()
    return f"已删除 {resolved}"

@tool
def read_file(path: str) -> str:
    """读取文件内容。"""
    resolved = (PROJECT_ROOT / path).resolve()
    try:
        return resolved.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"文件不存在: {resolved}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件。"""
    return f"邮件已发送至 {to}"

# Checkpointer 是 HITL 的必要条件
checkpointer = MemorySaver()

def writes_outside_workspace(request: ToolCallRequest) -> bool:
    """只有写入 workspace 目录之外的路径时才暂停。"""
    path = request.tool_call["args"].get("file_path", "")
    normalized = path.lstrip("./")  # 去掉前导的 ./ 或 /，兼容模型传的多种写法
    return not normalized.startswith("workspace/")

permissions = [
    # 记忆目录：始终允许读写，确保 agent 能自由读写记忆（放最前面优先匹配）
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/workspace/memory/**"],
        mode="allow",
    ),
    # 机密目录：读写全部拒绝（最严格）
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/workspace/secrets/**"],
        mode="deny",
    ),
    # 私有目录：读取需审批，写入直接拒绝
    FilesystemPermission(
        operations=["read"],
        paths=["/workspace/private/**"],
        mode="interrupt",
    ),
    FilesystemPermission(
        operations=["write"],
        paths=["/workspace/private/**"],
        mode="deny",
    ),
    # 配置目录：允许读，写入需审批
    FilesystemPermission(
        operations=["write"],
        paths=["/workspace/config/**"],
        mode="interrupt",
    ),
    # public 目录及其余路径：不写规则即默认 allow
]

agent = create_deep_agent(
    model=model,
    system_prompt=(
        "你维护两个持久化记忆文件，务必分开存放：\n"
        "- /workspace/memory/user.md：关于用户的事实、偏好、个人信息（姓名、邮箱、语言/工作习惯等）。\n"
        "- /workspace/memory/agent.md：你自己学到的经验、知识、可复用的工作模式。\n"
        "当获得值得长期保存的信息时，用 edit_file 工具追加到对应文件："
        "用户相关信息写 user.md，你自己的经验写 agent.md，不要混放，也绝不写入任何密钥/凭证。"
    ),
    memory=["/workspace/memory/agent.md", "/workspace/memory/user.md"],
    backend=FilesystemBackend(
        root_dir=PROJECT_ROOT,
        virtual_mode=True
    ),
    permissions=permissions,
    tools=[delete_file, read_file, send_email],
    interrupt_on={
        "delete_file": {"allowed_decisions": ["approve", "edit", "reject"]},
        "read_file": False,    # 无需中断
        "send_email": {"allowed_decisions": ["approve", "reject"]},  # 只能审批或拒绝，不能修改
        "write_file":{
        "allowed_decisions": ["approve", "edit", "reject"],
            "when": writes_outside_workspace,
        }
    },
    checkpointer=checkpointer,  # 必须配置！
)

import uuid
if __name__ == "__main__":
    myuuid = uuid.uuid4()
    config = {"configurable": {"thread_id": str(myuuid)}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "帮我读取./workspace下的test.txt文件，然后在帮我在文件中写入这是测试内容"}]},
        config=config,
        version="v2"
    )
    print(result)

    if result.interrupts:
        interrupt_value = result.interrupts[0].value
        action_requests = interrupt_value["action_requests"]
        review_configs = interrupt_value["review_configs"]
        config_map = {cfg["action_name"]: cfg for cfg in review_configs}

        # 展示给用户
        for action in action_requests:
            review_config = config_map[action["name"]]
            args = action.get("arguments", action.get("args", {}))
            print(f"工具: {action['name']}")
            print(f"参数: {args}")
            print(f"可选决策: {review_config['allowed_decisions']}")

        # Step 3: 用户做出决策
        decisions = [
            {"type": "approve"}  # 用户批准删除
        ]

        # Step 4: 恢复执行（必须用相同的 config！）
        result = agent.invoke(
            Command(resume={"decisions": decisions}),
            config=config,  # 同一个 thread_id
            version="v2",
        )

    # 获取最终结果
    print(result.value["messages"][-1].content)