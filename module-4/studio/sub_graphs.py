# -*- coding: utf-8 -*-

# --- 核心概念: 子图 (Sub-graphs) ---
# 在 LangGraph 中, 一个编译好的图 (Graph) 本身可以被用作另一个图中的一个节点 (Node)。

# -----------
# --- 知识点: typing.List 和 typing.Optional ---
# 在 Python 的类型提示系统中, `List` 和 `Optional` 是两个最常用和最重要的工具, 用于精确地描述数据结构。
#
# --- 1. `List[T]` (泛型列表) ---
#
# Q: 为什么使用 `List[Log]` 而不是普通的 `list`?
# A: 为了明确列表中元素的类型, 从而获得静态类型检查的好处。
#
# - **`list` (无类型提示)**:
#   `my_list: list = [log1, log2, "a string", 123]`
#   - 这是一个普通的 Python 列表, 它可以包含任何类型的元素。
#   - 类型检查器 (如 MyPy) 不知道列表里应该是什么, 因此无法在编程时发现类型错误。
#     比如, 你想对列表中的每个元素调用 `log.get('question')`, 类型检查器不会报错, 但程序在运行时会因为遇到字符串 "a string" 而崩溃。
#
# - **`List[Log]` (有类型提示)**:
#   `my_list: List[Log] = [log1, log2]`
#   - 这明确地告诉开发者和类型检查器: "这个列表**只能**包含 `Log` 类型的对象"。
#   - **优点**:
#     - **代码可读性**: 一眼就能看出这个列表的用途和内容。
#     - **静态检查**: 如果你尝试 `my_list.append("a string")`, VS Code 或 MyPy 会立刻划线报错, 防止 bug 进入运行阶段。
#     - **智能提示**: 编辑器会根据 `Log` 类型为你提供准确的代码补全建议 (如 `log.question`, `log.id` 等)。
#
# `List` 是一个**泛型 (Generic)** 类型, `[Log]` 就是它的类型参数。你可以用任何类型来参数化它, 如 `List[str]`, `List[int]` 等。
#
# --- 2. `Optional[T]` (可选类型) ---
#
# Q: `Optional[List]` 或 `Optional[int]` 是什么意思?
# A: 它表示一个变量的值**可以是指定的类型, 或者 `None`**。
#
# `Optional[T]` 本质上是 `Union[T, None]` 的一个简写形式, 它俩完全等价。
#
# - **使用场景**: 当一个字段、变量或函数返回值可能不存在时, 使用 `Optional` 是最佳实践。
#   在下面的 `Log` 类中, `docs: Optional[List]` 意味着一个日志条目**可能**有关联的文档列表 (`List`), 也**可能**没有 (`None`)。
#
# - **优点**:
#   - **明确性**: 它强制开发者处理 `None` 的情况。当你访问一个 `Optional` 类型的变量时, 类型检查器会提醒你: "这个值可能是 None, 你需要先检查一下"。
#   - **避免 `NoneType` 错误**: 这是 Python 中最常见的运行时错误之一。通过显式使用 `Optional`, 可以在编码阶段就避免它。
#
#   ```python
#   def process_docs(log: Log):
#       if log['docs'] is not None:
#           # 在这个代码块里, 类型检查器知道 log['docs'] 是 List, 可以安全地迭代
#           for doc in log['docs']:
#               print(doc)
#       else:
#           # 如果不加这个 if-else 检查, 直接 for doc in log['docs']
#           # 类型检查器就会发出警告或错误。
#           print("该日志没有相关文档。")
#   ```
#
# 总结: `List` 和 `Optional` 让我们的代码更安全、更清晰、更易于维护, 是编写高质量 Python 代码的基石。
# 这种强大的功能允许我们将复杂的工作流分解成更小、更易于管理和可重用的模块化组件。
#---------
# 把子图想象成一个封装了特定任务的“黑盒子”。父图不需要知道子图内部的复杂逻辑,
# 只需向它提供所需的输入, 然后接收其处理后的输出即可。
#
# 优点:
# 1.  **模块化 (Modularity)**: 将大型、复杂的图分解为功能独立的子图, 使代码更清晰、更易于维护。
# 2.  **可重用性 (Reusability)**: 同一个子图可以在不同的父图中被多次使用, 避免重复编码。
# 3.  **封装 (Encapsulation)**: 子图隐藏了其内部实现细节, 父图只需关注更高层次的逻辑。
# 4.  **并行执行 (Parallelism)**: 多个子图可以像普通节点一样并行运行, 提高效率。

# 导入 'add' 操作符, 我们将用它来合并列表。
from operator import add
# 导入类型提示, 用于定义数据结构。
from typing import List, Optional, Annotated
from typing_extensions import TypedDict
# 从 langgraph.graph 中导入核心组件。
from langgraph.graph import StateGraph, START, END

# --- 知识点: 日志数据结构 ---
# 我们首先定义一个 `TypedDict` 来标准化日志条目的结构。
# 这确保了在整个应用中, 所有部分都以相同的方式理解和处理日志数据。
class Log(TypedDict):
    """定义单个日志条目的数据结构。"""
    id: str  # 日志的唯一标识符
    question: str  # 日志记录的用户问题
    docs: Optional[List]  # 检索到的相关文档, 可能不存在
    answer: str  # 系统生成的回答
    grade: Optional[int]  # 对回答的评分, 可能不存在
    grader: Optional[str]  # 评分者, 可能不存在
    feedback: Optional[str]  # 具体的反馈信息, 可能不存在

# --- 1. 构建第一个子图: 故障分析 (Failure Analysis) ---
# 这个子图的职责是: 从一堆日志中筛选出被标记为“失败”的记录, 并生成一份关于这些失败的摘要。

# --- 知识点: 子图的状态映射 (State Mapping) ---
# 子图通常只关心整个应用状态 (State) 的一部分。为了让子图能够独立工作, 我们需要定义两个关键的状态类:
# 1.  **输入状态 (Input State)**: 定义了子图运行时**需要**从父图接收哪些数据。
# 2.  **输出状态 (Output State)**: 定义了子图执行完毕后, 会**返回**哪些数据给父图。
#
# LangGraph 通过在 `StateGraph` 的构造函数中指定 `input` 和 `output` 类型来实现这种映射。
# `builder = StateGraph(input=InputState, output=OutputState)`
#
# 工作流程:
# - **输入时**: 当父图调用子图时, LangGraph 会自动从父图的完整状态中提取 `InputState` 定义的字段,
#   并将其传递给子图作为其初始状态。
# - **输出时**: 当子图执行完毕 (到达 END 节点), LangGraph 会从子图的最终状态中提取 `OutputState`
#   定义的字段, 并将这些字段的值更新回父图的相应状态字段。

class FailureAnalysisState(TypedDict):
    """故障分析子图的**输入和内部**状态。"""
    cleaned_logs: List[Log]  # 从父图接收的、已清洗的日志列表
    failures: List[Log]      # 内部状态: 筛选出的失败日志
    fa_summary: str          # 内部状态: 生成的故障摘要
    processed_logs: List[str] # 内部状态: 记录处理过的日志ID

class FailureAnalysisOutputState(TypedDict):
    """故障分析子图的**输出**状态。这些字段将被返回给父图。"""
    fa_summary: str
    processed_logs: List[str]

def get_failures(state: FailureAnalysisState) -> dict:
    """
    一个子图节点: 从清洗过的日志中筛选出包含 "grade" 字段的失败日志。
    在我们的设定中, 只要日志被评分过 (`grade` 存在), 就认为它是一个需要分析的案例。
    """
    print("---(子图: FA) 节点: get_failures---")
    cleaned_logs = state["cleaned_logs"]
    # 列表推导式, 高效筛选
    failures = [log for log in cleaned_logs if "grade" in log and log.get("grade") is not None]
    return {"failures": failures}

def generate_fa_summary(state: FailureAnalysisState) -> dict:
    """
    一个子图节点: 根据失败日志生成摘要。
    同时, 创建一个处理记录, 标记哪些日志被这个子图处理了。
    """
    print("---(子图: FA) 节点: generate_fa_summary---")
    failures = state["failures"]
    if not failures:
        return {"fa_summary": "没有检测到失败日志。", "processed_logs": []}
    
    # 在真实场景中, 这里会调用一个 LLM 来进行智能摘要。
    # fa_summary = summarize_with_llm(failures)
    fa_summary = f"检测到 {len(failures)} 个失败案例。主要问题似乎与文档检索的质量不高有关。"
    
    # 返回摘要, 并生成一个处理日志列表, 用于和另一个子图的结果合并。
    processed_logs = [f"failure-analysis-on-log-{failure['id']}" for failure in failures]
    return {"fa_summary": fa_summary, "processed_logs": processed_logs}

# 使用 `input` 和 `output` 参数来定义状态映射
fa_builder = StateGraph(input=FailureAnalysisState, output=FailureAnalysisOutputState)
fa_builder.add_node("get_failures", get_failures)
fa_builder.add_node("generate_fa_summary", generate_fa_summary)
fa_builder.add_edge(START, "get_failures")
fa_builder.add_edge("get_failures", "generate_fa_summary")
fa_builder.add_edge("generate_fa_summary", END)

# --- 2. 构建第二个子图: 问题摘要 (Question Summarization) ---
# 这个子图的职责是: 对所有日志中的问题进行摘要, 并生成一份报告。

class QuestionSummarizationState(TypedDict):
    """问题摘要子图的输入和内部状态。"""
    cleaned_logs: List[Log]
    qs_summary: str
    report: str
    processed_logs: List[str]

class QuestionSummarizationOutputState(TypedDict):
    """问题摘要子图的输出状态。"""
    report: str
    processed_logs: List[str]

def generate_qs_summary(state: QuestionSummarizationState) -> dict:
    """
    一个子图节点: 摘要所有问题。
    """
    print("---(子图: QS) 节点: generate_qs_summary---")
    cleaned_logs = state["cleaned_logs"]
    if not cleaned_logs:
        return {"qs_summary": "没有日志可供摘要。", "processed_logs": []}

    # 真实场景会调用 LLM
    # summary = summarize_questions_with_llm(cleaned_logs)
    summary = f"对 {len(cleaned_logs)} 个问题进行了分析。用户问题主要集中在使用 ChatOllama 和 Chroma 向量存储方面。"
    
    processed_logs = [f"summary-on-log-{log['id']}" for log in cleaned_logs]
    return {"qs_summary": summary, "processed_logs": processed_logs}

def generate_report(state: QuestionSummarizationState) -> dict:
    """
    一个子图节点: 根据摘要生成报告 (例如, 发送到 Slack)。
    """
    print("---(子图: QS) 节点: generate_report---")
    qs_summary = state["qs_summary"]
    # 真实场景会调用报告生成工具或 API
    # report = report_generation_tool(qs_summary)
    report = f"每日报告: {qs_summary} 请相关团队注意。"
    return {"report": report}

qs_builder = StateGraph(input=QuestionSummarizationState, output=QuestionSummarizationOutputState)
qs_builder.add_node("generate_qs_summary", generate_qs_summary)
qs_builder.add_node("generate_report", generate_report)
qs_builder.add_edge(START, "generate_qs_summary")
qs_builder.add_edge("generate_qs_summary", "generate_report")
qs_builder.add_edge("generate_report", END)

# --- 3. 构建主图 (Entry Graph) ---
# 这个主图将编排整个工作流程:
# 1. 清洗原始日志。
# 2. 并行启动 "故障分析" 和 "问题摘要" 两个子图。
# 3. 收集并合并两个子图的输出。

class EntryGraphState(TypedDict):
    """主图的全局状态。"""
    raw_logs: List[Log]
    cleaned_logs: List[Log]
    # 'fa_summary' 只会在 FA 子图中生成, 然后被映射回这里。
    fa_summary: str
    # 'report' 只会在 QS 子图中生成, 然后被映射回这里。
    report: str
    # 'processed_logs' 会在两个子图中同时生成。
    # 我们使用 `Annotated` 和 `add` 操作符告诉 LangGraph,
    # 当两个子图都返回这个字段时, 应该将它们的列表合并, 而不是相互覆盖。
    processed_logs: Annotated[List[str], add]

def clean_logs(state: EntryGraphState) -> dict:
    """
    主图的第一个节点: 清洗日志。
    (在这个示例中, 我们只是简单地传递数据, 但真实场景可以包含复杂的清洗逻辑)
    """
    print("---(主图) 节点: clean_logs---")
    raw_logs = state["raw_logs"]
    # 假设的清洗逻辑:
    # cleaned_logs = [log for log in raw_logs if is_valid(log)]
    cleaned_logs = raw_logs
    return {"cleaned_logs": cleaned_logs}

# 编译子图, 得到可执行的图对象。
# 这两个对象现在可以像普通函数一样被用作节点。
failure_analysis_subgraph = fa_builder.compile()
question_summarization_subgraph = qs_builder.compile()

# 构建主图
entry_builder = StateGraph(EntryGraphState)
entry_builder.add_node("clean_logs", clean_logs)

# --- 知识点: 将子图作为节点添加 ---
# 我们使用 `add_node` 将编译好的子图添加到主图中。
# LangGraph 会自动处理状态的传入和传出映射。
entry_builder.add_node("failure_analysis", failure_analysis_subgraph)
entry_builder.add_node("question_summarization", question_summarization_subgraph)

# 定义主图的流程
entry_builder.add_edge(START, "clean_logs")
# 从 "clean_logs" 节点分叉, 并行执行两个子图
entry_builder.add_edge("clean_logs", "failure_analysis")
entry_builder.add_edge("clean_logs", "question_summarization")
# 两个子图都执行完毕后, 流程结束。
# LangGraph 会自动等待两个并行分支都完成后再汇集到 END。
entry_builder.add_edge("failure_analysis", END)
entry_builder.add_edge("question_summarization", END)

# 编译主图
graph = entry_builder.compile()

# --- 启动测试案例 ---
if __name__ == "__main__":
    
    # 准备一些模拟的原始日志数据
    mock_logs = [
        Log(id="1", question="如何使用 ChromaDB?", answer="...", grade=None),
        Log(id="2", question="Ollama 的 API 怎么用?", answer="...", grade=None),
        Log(id="3", question="ChromaDB 的文档在哪里?", answer="错误的链接", grade=1, grader="bot", feedback="检索到了不相关的文档"),
        Log(id="4", question="LangGraph 如何处理并行?", answer="...", grade=None),
        Log(id="5", question="如何安装 Ollama?", answer="安装指南...", grade=None),
        Log(id="6", question="ChromaDB get 失败", answer="你可能需要更新版本", grade=2, grader="human", feedback="回答不完整"),
    ]

    # 定义初始状态, 只需提供主图入口所需的 `raw_logs`
    initial_state = {"raw_logs": mock_logs}

    print("--- 开始执行图 ---")
    # 使用 `invoke` 方法执行图
    final_state = graph.invoke(initial_state)

    print("\n--- 图执行完毕, 查看最终状态 ---")
    
    print("\n[故障分析子图的输出]:")
    print(f"  摘要: {final_state.get('fa_summary')}")
    
    print("\n[问题摘要子图的输出]:")
    print(f"  报告: {final_state.get('report')}")
    
    print("\n[合并后的处理记录]:")
    # 这个列表是两个子图 `processed_logs` 列表合并后的结果
    # 注意它的顺序可能是不确定的, 因为子图是并行执行的
    for log_entry in final_state.get('processed_logs', []):
        print(f"  - {log_entry}")

    # 我们可以通过 `graph.get_graph().draw_mermaid()` 来查看图的结构
    print("\n--- 图结构 (Mermaid 语法) ---")
    print(graph.get_graph().draw_mermaid())