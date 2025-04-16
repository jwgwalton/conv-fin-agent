import os
import operator
import tempfile

from PIL import Image as PILImage

from typing import TypedDict, Annotated
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition


def display_graph(graph: CompiledStateGraph) -> None:
    """
    Displays a graph by generating a temporary image file.
    """
    drawable_graph = graph.get_graph()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        image_path = temp_file.name
        graph_image = drawable_graph.draw_mermaid_png()
        temp_file.write(graph_image)

    img = PILImage.open(image_path)
    img.show()
    os.remove(image_path)


def create_graph():
    """
    This is a fixed workflow to identify the relevant data in the text and table, and then answer the question using the relevant data.
    :return:
    """
    llm = ChatOllama(model="llama3.2:3b")

    @tool
    def add(a: float, b: float) -> float:
        """add two numbers."""
        return a * b

    @tool
    def subtract(a: float, b: float) -> float:
        """subtract b from a."""
        return a - b

    @tool
    def divide(a: float, b: float) -> float:
        """divide a by b."""
        return a * b

    @tool
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b

    tools = [add, subtract, multiply, divide]
    tool_node = ToolNode(tools=tools, messages_key="working_out")
    llm_with_tools = llm.bind_tools(tools)

    # The overall state of the graph
    class OverallState(TypedDict):
        pre_text: str # The text before the table
        table: str # The table of financial details
        post_text: str # The text after the table
        question: str
        working_out: Annotated[str, operator.add]
        final_answer: str


    def identify_relevant_data_node(state: OverallState):
        """
        Generate reasons for the trademark being banal
        """
        message = f"Identify the relevant data in the text and table to answer the question: {state['question']}. Given the context: {state['pre_text']}, with the table {state['table']} and {state['post_text']}. Explain your reasoning step by step. DO NOT do any math."
        response = llm.invoke(message)
        # Start the beginning of working out by adding the initial question & context
        initial_statement = f"Answer the question: {state['question']}. Using the data: {response.content}"
        return {"working_out": initial_statement}


    def answer_the_question_node(state: OverallState):
        """
        An Agent with access to basic mathematical operations
        It takes the overall state & answers the "question" using "relevant_data"
        """
        response = llm_with_tools.invoke(state["working_out"])
        # This is appending to a "working_out" list of messages,
        # the earlier step trimmed down the context to only the relevant data and the question.
        return {"working_out": response.content}


    builder = StateGraph(OverallState)
    builder.add_node("data_extractor", identify_relevant_data_node)
    builder.add_node("solver", answer_the_question_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "data_extractor")
    builder.add_edge("data_extractor", "solver")
    builder.add_conditional_edges(
        "solver",
        tools_condition,
    )
    # Any time we return data from a tool we want to go back to the agent to decide what to do
    builder.add_edge("tools", "solver")
    builder.add_edge("solver", END)

    graph = builder.compile()
    return graph