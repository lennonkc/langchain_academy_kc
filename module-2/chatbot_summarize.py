# -*- coding: utf-8 -*-
"""
该脚本展示了如何构建一个带有消息摘要功能的聊天机器人。
它使用LangGraph来创建一个可以随着对话进行动态生成摘要的图，
从而在保持长期对话记忆的同时，有效控制token的使用量。
"""

import os
import getpass
from typing import TypedDict, Annotated
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage, RemoveMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display

# --- 环境设置 ---
def _set_env(var: str):
    """如果环境变量未设置，则提示用户输入。"""
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# 设置OpenAI API密钥
_set_env("OPENAI_API_KEY")

# (可选) 设置LangSmith用于追踪，方便调试和监控
# _set_env("LANGSMITH_API_KEY")
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_PROJECT"] = "langchain-academy"


# --- 模型初始化 ---
# 初始化语言模型，这里使用gpt-4o
try:
    model = ChatOpenAI(model="gpt-4o", temperature=0)
except Exception as e:
    print(f"模型初始化失败，请检查API密钥是否正确: {e}")
    exit()

# --- 状态定义 ---
class State(MessagesState):
    """
    定义图的状态。
    它继承自MessagesState，该状态默认包含一个'messages'键来存储消息列表。
    我们额外添加一个'summary'键来存储对话摘要。
    """
    summary: str

# --- 节点定义 ---

def call_model(state: State) -> dict:
    """
    调用LLM生成回复的节点。
    它会检查状态中是否存在摘要，如果存在，就将其作为上下文提供给模型。
    """
    summary = state.get("summary", "")
    messages = state["messages"]
    
    # 如果有摘要，将其格式化为系统消息并添加到消息列表的开头
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        # 将摘要信息和新消息一起传递给模型
        messages_with_summary = [SystemMessage(content=system_message)] + messages
        response = model.invoke(messages_with_summary)
    else:
        response = model.invoke(messages)
        
    # 返回模型的回复，更新messages状态
    return {"messages": [response]}

def summarize_conversation(state: State) -> dict:
    """
    生成或更新对话摘要的节点。
    它会使用LLM来压缩对话历史，并修剪消息列表以节省token。
    """
    summary = state.get("summary", "")
    messages = state["messages"]

    # 构建摘要提示
    if summary:
        # 如果已有摘要，则要求模型在现有摘要的基础上进行扩展
        summary_prompt = (
            f"This is a summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        # 如果没有摘要，则要求模型创建一个新的摘要
        summary_prompt = "Create a summary of the conversation above:"

    # 将摘要提示添加到消息历史中
    summary_messages = messages + [HumanMessage(content=summary_prompt)]
    
    # 调用模型生成新摘要
    new_summary = model.invoke(summary_messages)

    # 为了节省空间，我们从状态中删除除了最近两条之外的所有消息
    # LangGraph通过RemoveMessage的id来精确删除消息
    messages_to_delete = [RemoveMessage(id=m.id) for m in messages[:-2]]

    # 返回更新后的摘要和被删除的消息列表
    return {"summary": new_summary.content, "messages": messages_to_delete}

# --- 条件边 ---

def should_continue(state: State) -> str:
    """
    决定下一跳的条件边。
    如果对话长度超过阈值（这里是6条消息），则触发摘要节点。
    否则，结束当前轮次的图执行。
    """
    messages = state["messages"]
    if len(messages) > 6:
        return "summarize_conversation"
    return END

# --- 图的构建和编译 ---

# 使用MemorySaver作为检查点，实现内存功能
# 这允许图在多次调用之间保持状态，从而实现多轮对话
memory = MemorySaver()

# 定义一个新的状态图
workflow = StateGraph(State)

# 添加节点
workflow.add_node("conversation", call_model)
workflow.add_node("summarize_conversation", summarize_conversation)

# 设置图的入口点
workflow.add_edge(START, "conversation")

# 添加条件边
# 在'conversation'节点之后，根据'should_continue'的返回值决定去向
workflow.add_conditional_edges("conversation", should_continue)

# 从'summarize_conversation'节点到终点
workflow.add_edge("summarize_conversation", END)

# 编译图，并附加检查点
graph = workflow.compile(checkpointer=memory)

# (可选) 可视化图的结构
# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception as e:
#     print(f"无法生成图的可视化表示: {e}")

# --- 主执行逻辑 ---
if __name__ == "__main__":
    print("聊天机器人已启动。输入 'exit' 退出。")
    
    # 为本次会话创建一个唯一的线程ID
    # 这使得我们可以同时处理多个独立的对话
    config = {"configurable": {"thread_id": "user_1"}}
    
    # 示例对话
    # 你可以注释掉这部分，直接进入下面的交互式循环
    initial_messages = [
        ("hi! I'm Lance", "Hello Lance! How can I help you today?"),
        ("what's my name?", "Your name is Lance."),
        ("i like the 49ers!", "The 49ers are a great team! Their defense is particularly strong."),
    ]
    
    # for user_msg, ai_response in initial_messages:
    #     print(f"You: {user_msg}")
    #     output = graph.invoke({"messages": [HumanMessage(content=user_msg)]}, config)
    #     print(f"AI: {output['messages'][-1].content}")

    # 交互式聊天循环
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        # 调用图来处理用户输入并获得回复
        input_message = HumanMessage(content=user_input)
        try:
            output = graph.invoke({"messages": [input_message]}, config)
            # 输出AI的最新回复
            ai_message = output['messages'][-1].content
            print(f"AI: {ai_message}")

            # 检查当前状态和摘要
            current_state = graph.get_state(config)
            summary = current_state.values.get("summary", "")
            # print(f"\n--- DEBUG INFO ---")
            # print(f"Current Summary: {summary}")
            # print(f"Current Messages Count: {len(current_state.values['messages'])}")
            # print("--------------------\n")

        except Exception as e:
            print(f"发生错误: {e}")

    print("聊天结束。")