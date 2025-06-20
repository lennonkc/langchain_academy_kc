# 导入 'operator' 模块, 它提供了一套与Python内部操作符对应的函数。
# 在这里, 我们将使用 operator.add 来合并列表, 这是 LangGraph 中实现状态更新的一种常见模式。
import operator
# 导入 'Annotated' 类型, 它允许我们向类型提示中添加上下文相关的元数据。
# 在 LangGraph 中, 它与 operator.add 结合使用, 用于指定状态字段的更新方式。
# 导入 'TypedDict'，它用于创建具有固定键和特定值类型的字典，可以提供静态类型检查。
from typing import Annotated
from typing_extensions import TypedDict

# 从 langchain_core 中导入 'Document' 类, 用于表示从数据源加载的文档。
from langchain_core.documents import Document
# 从 langchain_core 中导入消息类型 'HumanMessage' 和 'SystemMessage', 用于与语言模型进行交互。
from langchain_core.messages import HumanMessage, SystemMessage

# 从 langchain_community 中导入 'WikipediaLoader', 这是一个用于从维基百科加载文档的工具。
from langchain_community.document_loaders import WikipediaLoader
# 从 langchain_community 中导入 'TavilySearchResults', 这是一个进行网络搜索的工具。
from langchain_community.tools import TavilySearchResults

# 从 langchain_openai 中导入 'ChatOpenAI', 这是与 OpenAI 的聊天模型 (如 GPT-4o) 进行交互的接口。
from langchain_openai import ChatOpenAI

# 从 langgraph.graph 中导入 'StateGraph', 'START', 'END'。
# 'StateGraph' 是构建状态图的核心类。
# 'START' 和 'END' 是特殊的节点, 分别代表图的入口和出口。
from langgraph.graph import StateGraph, START, END

# --- 知识点: Class 语法和面向对象编程 ---
# Python 中的 `class` 关键字用于定义一个类, 它是创建对象的蓝图。
# 类可以包含属性 (变量) 和方法 (函数)。
# 在下面的代码中, 我们定义了一个名为 `State` 的类, 它继承自 `TypedDict`。
# 继承是一种面向对象编程的概念, 子类可以获取父类的所有功能。
# `State` 类本身没有定义新的方法, 而是利用 `TypedDict` 的特性来定义一个结构化的数据类型。

# --- 知识点: TypedDict 深入解析 ---
# `TypedDict` (类型化字典) 是一个强大的工具, 用于为字典提供类型提示, 从而增强代码的可读性和健壮性。
#
# 核心特性:
# 1.  **静态类型检查 (在编程阶段)**:
#     "静态分析"阶段就是指**编程阶段**，而不是程序（例如 LangGraph 应用）部署后的**运行阶段**。
#     当您在 VS Code (借助 Pylance/Pyright) 或使用 `mypy` 等工具编写和检查代码时，`TypedDict` 会帮助检查：
#     - 字典是否包含了所有必需的键。
#     - 每个键对应的值是否是正确的类型。
#     - 是否有拼写错误的键名。
#     这可以在代码运行前就发现潜在的 bug。
#
# 2.  **运行时等效性 (在运行阶段)**:
#     一旦程序开始执行 (`python your_script.py`), `TypedDict` 定义的类型在功能上就和一个普通的 `dict` 完全相同。
#     Python 解释器在运行时不会检查键是否存在或类型是否正确，除非您显式地编写了检查代码。
#     例如:
#       >>> Point2D = TypedDict('Point2D', {'x': int, 'y': int})
#       >>> Point2D(x=1, y=2) == dict(x=1, y=2)  # 这将返回 True
#
# 3.  **定义方式对比**:
#     - **类语法 (推荐)**:
#       ```python
#       class MyState(TypedDict):
#           key: str
#           other: int
#       ```
#       - **优点**: 更易读，支持继承，可以添加文档字符串和方法，类型检查工具支持更好。
#       - **适用**: Python 3.8 及以上版本。
#
#     - **函数式语法**:
#       ```python
#       MyState = TypedDict('MyState', {'key': str, 'other': int})
#       ```
#       - **优点**: 语法简洁，适用于需要动态创建类型的场景。
#       - **缺点**: 扩展性差，不支持继承和文档字符串。
#
# 4.  **键的强制性 (Totality)**:
#     - **默认 (`total=True`)**: 所有在类中定义的键都必须存在。
#     - **可选 (`total=False`)**: `class MyState(TypedDict, total=False): ...` 这会使所有键都变为可选。
#     - **精细控制**: 使用 `typing.Required` 和 `typing.NotRequired` 可以分别标记单个键为必需或可选 (在 Python 3.11+ 中可用)。
#
# 在 LangGraph 中, `TypedDict` 被用作定义图状态 (State) 的标准方式。这提供了一个清晰、类型安全的“契约”，
# 规定了在整个图的生命周期中数据应该如何组织。每个节点都可以安全地访问和修改这个状态字典, 因为其结构是预先定义好的。

# 初始化语言模型
# model="gpt-4o": 指定使用的模型。
# temperature=0: 设置温度为0, 表示模型的输出将更具确定性, 减少随机性。
llm = ChatOpenAI(model="gpt-4o", temperature=0)

class State(TypedDict):
    """
    定义图的状态。这个 TypedDict 描述了在图的执行过程中需要跟踪的所有数据。
    每个键代表状态的一个字段。
    """
    # 'question': 用户的原始问题, 类型为字符串。
    question: str
    # 'answer': 由 LLM 生成的最终答案, 类型为字符串。
    answer: str
    # 'context': 从各个来源 (网络搜索、维基百科) 收集到的上下文信息。
    # --- 知识点: Annotated - 为何不能只用 list? ---
    #
    # Q: 为什么这里用 `Annotated[list, operator.add]` 而不是直接用 `list`?
    # A: 为了解决并行节点更新同一个状态字段时的数据覆盖问题。
    #
    # 1. 如果只用 `context: list`:
    #    当 `search_web` 和 `search_wikipedia` 并行运行时, 它们都会尝试更新 `context`。
    #    如果 `search_web` 先完成, 它返回 `{'context': ['web的结果']}`。此时状态 `context` 变为 `['web的结果']`。
    #    然后 `search_wikipedia` 完成, 它返回 `{'context': ['wiki的结果']}`。此时状态 `context` 会被**覆盖**成 `['wiki的结果']`。
    #    最终, `search_web` 的结果就丢失了。
    #
    # 2. `Annotated` 的作用:
    #    `Annotated` 允许我们给一个类型 (如 `list`) 附加额外的元数据 (metadata)。
    #    这些元数据对普通的 Python 代码和类型检查器是透明的, 但特定的框架 (如 LangGraph) 可以读取并利用它们。
    #
    # 3. `Annotated[list, operator.add]` 的含义:
    #    - `list`: 这是基础类型, 明确 `context` 是一个列表。
    #    - `operator.add`: 这是附加的元数据, 在 LangGraph 中它被解释为一个 "reducer" (聚合器) 函数。
    #
    #    这行代码等于在告诉 LangGraph:
    #    " `context` 是一个列表。当有多个节点同时要更新它时, 不要相互覆盖, 而是使用 `operator.add` 函数 (即列表的 `+` 操作)
    #    将所有返回的列表**合并**在一起。"
    #
    #    因此, `['web的结果']` 和 `['wiki的结果']` 会被正确地合并为 `['web的结果', 'wiki的结果']`。
    #
    # 总结: `Annotated` 在这里就像一个特殊的指令标签, 它在不改变 `context` 是一个列表这个事实的前提下,
    # 为 LangGraph 框架提供了如何安全地处理并发更新的关键信息。
    context: Annotated[list, operator.add]

# --- 知识点: LangGraph 节点 (Node) 深度解析 ---
#
# --- 1. Python 函数签名语法 `search_web(state: State) -> dict:` ---
# 这是 Python 的类型提示 (Type Hinting) 语法, 用于增强代码的可读性和可维护性。
# - `state: State`: 冒号 `:` 用于注解参数类型。它告诉开发者和静态分析工具, `state` 参数期望接收一个 `State` 类型的对象。
# - `-> dict`: 箭头 `->` 用于注解函数返回值的类型。它表明 `search_web` 函数预计将返回一个字典 (`dict`)。
# 注意: 这些只是“提示”，Python 解释器在运行时并不会强制执行它们，但它们对于开发工具 (如代码补全) 和代码质量检查至关重要。
#
# --- 2. LangGraph 节点的输入与输出 ---
# 在 LangGraph 中, 节点是图的基本计算单元。每个节点都遵循严格的输入/输出模式:
# - **输入 (Input)**: 节点的第一个参数总是图的当前状态 (state)。这个 state 对象包含了图到目前为止的所有信息, 节点可以从中读取所需数据。
# - **输出 (Output)**: 节点必须返回一个字典。这个字典的键必须是 State `TypedDict` 中定义的键。
#   LangGraph 会用这个返回的字典来更新对应的状态字段。如果返回了 State 中不存在的键, 会导致错误。
#   例如, `return {"context": [formatted_search_docs]}` 就是告诉 LangGraph: "请将 `[formatted_search_docs]` 这个列表更新到状态的 `context` 字段中"。
#
# --- 3. 节点的高级功能 (除了输入输出) ---
# 一个 LangGraph 节点不仅仅是简单的“输入->处理->输出”，它还可以拥有更复杂的能力，以下是一些示例：
#
# - **运行时配置 (Runtime Configuration)**:
#   节点可以接受第二个可选参数 `config`，从而在运行时动态调整行为。
#   ```python
#   # from langchain_core.runnables import RunnableConfig
#   def my_node(state: State, config: RunnableConfig):
#       # 从 config 中获取 user_id
#       user_id = config.get("configurable", {}).get("user_id", "default_user")
#       print(f"Current user is: {user_id}")
#       # ... 节点逻辑 ...
#       return {"answer": f"Processed for {user_id}"}
#
#   # 调用时传入 config
#   # graph.invoke(..., config={"configurable": {"user_id": "user_123"}})
#   ```
#
# - **流式输出 (Streaming)**:
#   节点可以从内部实时地“流式”返回中间数据，而无需等到整个节点执行完毕。
#   ```python
#   # from langgraph.config import get_stream_writer
#   def streaming_node(state: State):
#       writer = get_stream_writer()
#       writer.write({"progress": "25%"})
#       # ...长时间运行的任务...
#       writer.write({"progress": "75%"})
#       return {"answer": "done"}
#
#   # 调用时需要指定 stream_mode
#   # for chunk in graph.stream(..., stream_mode="values" | "updates" | "states" | "custom"):
#   #     print(chunk)
#   ```
#   **参数**: `stream_mode` 可选值为 `"values"`(默认), `"updates"`, `"states"`, `"custom"`。
#
# - **错误处理 (Error Handling)**:
#   使用预构建的 `ToolNode` 时，可以精细控制其错误处理行为。
#   ```python
#   # from langgraph.prebuilt import ToolNode
#   # tools = [...]
#   # node = ToolNode(tools, handle_tool_errors=True)
#   ```
#   **参数**: `handle_tool_errors` 可选值为:
#   - `True` (默认): 捕获异常并作为错误信息返回。
#   - `False`: 不处理异常，让其直接抛出。
#   - `"custom error message"`: 返回一个自定义的错误字符串。
#   - `callable_function`: 使用一个自定义函数来处理异常。
#
# - **中断与人工介入 (Human-in-the-loop)**:
#   可以在图的任何节点前后设置断点，暂停图的执行。
#   ```python
#   # 编译时设置断点
#   # graph = builder.compile(interrupt_before=["generate_answer"])
#
#   # 运行时设置断点
#   # graph.invoke(..., interrupt_after=["search_web"])
#   ```
#   **参数**: `interrupt_before` 和 `interrupt_after` 都接受一个包含节点名称字符串的列表。
#
# - **与持久化存储交互**:
#   当配置了 Checkpointer (检查点/持久化后端) 时，节点可以访问更多上下文信息。
#   ```python
#   # from langgraph.checkpoint.base import BaseCheckpointSaver
#   def memory_node(state: State, store: BaseCheckpointSaver, previous: State):
#       # store: 访问检查点存储对象
#       # previous: 获取上一个检查点的状态
#       print("Previous answer was:", previous.get('answer'))
#       return {}
#   ```
#   **参数**: LangGraph 会根据 Checkpointer 的配置自动向节点注入 `store` 和 `previous` 等参数。

def search_web(state: State) -> dict:
    """
    一个图节点: 从网络上检索文档。
    
    参数:
        state (State): 当前图的状态, 包含 'question' 字段。
        
    返回:
        dict: 一个字典, 其中 'context' 键包含从网络搜索中找到的格式化文档。
    """
    print("---节点: search_web---")
    # 初始化 Tavily 搜索工具, max_results=3 表示最多返回3个搜索结果。
    tavily_search = TavilySearchResults(max_results=3)
    # 使用状态中的 'question' 来调用搜索工具。
    search_docs = tavily_search.invoke(state['question'])
    
    # 将搜索结果格式化为字符串。
    # 每个文档都被包裹在 <Document> 标签中, 并包含其来源 URL。
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )
    
    # 返回一个字典来更新状态。'context' 字段将被添加上新的搜索结果。
    return {"context": [formatted_search_docs]}

def search_wikipedia(state: State) -> dict:
    """
    一个图节点: 从维基百科检索文档。
    
    参数:
        state (State): 当前图的状态, 包含 'question' 字段。
        
    返回:
        dict: 一个字典, 其中 'context' 键包含从维基百科中找到的格式化文档。
    """
    print("---节点: search_wikipedia---")
    # 初始化维基百科加载器。
    # query=state['question']: 使用问题作为查询词。
    # load_max_docs=2: 最多加载2篇维基百科文章。
    search_docs = WikipediaLoader(query=state['question'], 
                                  load_max_docs=2).load()
    
    # 将维基百科文档格式化为字符串。
    # 每个文档都被包裹在 <Document> 标签中, 并包含其来源 (source) 和页码 (page)。
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
    
    # 返回一个字典来更新状态。
    return {"context": [formatted_search_docs]}

def generate_answer(state: State) -> dict:
    """
    一个图节点: 根据收集到的上下文生成答案。
    
    参数:
        state (State): 当前图的状态, 包含 'question' 和 'context' 字段。
        
    返回:
        dict: 一个字典, 其中 'answer' 键包含由 LLM 生成的答案对象。
    """
    print("---节点: generate_answer---")
    # 从状态中获取问题和上下文。
    context = state["context"]
    question = state["question"]
    
    # 创建一个提示模板, 指示 LLM 如何使用上下文来回答问题。
    answer_template = """使用以下上下文来回答问题: {question}
上下文: 
{context}
"""
    answer_instructions = answer_template.format(question=question, 
                                                 context=context)    
    
    # 调用 LLM 来生成答案。
    # 我们传递一个系统消息 (包含指令) 和一个人类消息 (提示开始回答)。
    answer = llm.invoke([SystemMessage(content=answer_instructions),
                         HumanMessage(content=f"请回答这个问题。")])
      
    # 返回一个字典来更新状态中的 'answer' 字段。
    return {"answer": answer}

# --- 图的构建 ---

# 1. 初始化 StateGraph
# 我们传入 `State` 类, 这样 StateGraph 就知道了图的状态结构。
builder = StateGraph(State)

# 2. 添加节点 (Nodes)
# 每个节点都有一个唯一的名称 (字符串) 和一个与之关联的函数。
builder.add_node("search_web", search_web)
builder.add_node("search_wikipedia", search_wikipedia)
builder.add_node("generate_answer", generate_answer)

# 3. 添加边 (Edges)
# 边定义了节点之间的执行流程。
# 从 START 开始, 并行执行 "search_wikipedia" 和 "search_web"。
# 这就是所谓的“扇出”(fan-out), 因为执行流程从一个点分叉到多个并行的分支。
builder.add_edge(START, "search_wikipedia")
builder.add_edge(START, "search_web")

# "search_wikipedia" 和 "search_web" 两个节点都执行完毕后, 将它们的结果汇集到 "generate_answer" 节点。
# 这就是所谓的“扇入”(fan-in), 因为多个并行分支的流程汇集到一个点。
# LangGraph 会自动等待这两个并行节点都完成后, 再执行 "generate_answer"。
builder.add_edge("search_wikipedia", "generate_answer")
builder.add_edge("search_web", "generate_answer")

# "generate_answer" 节点执行完毕后, 流程结束 (END)。
builder.add_edge("generate_answer", END)

# 4. 编译图
# `compile()` 方法将我们定义的节点和边组合成一个可执行的图对象。
graph = builder.compile()

# --- 启动测试案例 ---
# `if __name__ == "__main__":` 是 Python 的一个常用模式。
# 这段代码块里的内容只有在直接运行这个 .py 文件时才会执行,
# 如果这个文件被其他文件作为模块导入, 则不会执行。
# 这使得我们可以方便地为模块添加测试或演示代码。

if __name__ == "__main__":
    # 我们可以使用 `graph.get_graph().draw_mermaid_png()` 来可视化图的结构。
    # 需要安装 `pygraphviz` 和 `graphviz` 才能使用。
    # try:
    #     from IPython.display import Image, display
    #     display(Image(graph.get_graph().draw_mermaid_png()))
    # except ImportError:
    #     print("无法生成图的可视化, 请安装 pygraphviz 和 graphviz。")
    #     print("Mermaid 语法表示的图结构:")
    #     print(graph.get_graph().draw_mermaid())

    # 定义一个问题
    question = "谷歌2025年第二季度的财报表现如何?"
    
    print(f"问题: {question}\n")
    
    # 使用 `invoke` 方法来执行图。
    # 我们需要提供一个符合 `State` 结构的初始状态字典。
    # 这里我们只提供了 'question', 其他字段 ('answer', 'context') 将在图的执行过程中被填充。
    result = graph.invoke({"question": question})
    
    print("\n---最终结果---")
    # 打印最终状态中的 'answer' 字段。
    # `result` 是最终的状态字典。
    # `result['answer']` 是一个 AIMessage 对象, 我们需要访问它的 `.content` 属性来获取字符串形式的答案。
    print(f"答案: {result['answer'].content}")
    
    print("\n---收集的上下文---")
    # 打印收集到的所有上下文信息
    for i, ctx in enumerate(result['context']):
        print(f"上下文 {i+1}:\n{ctx}\n")
