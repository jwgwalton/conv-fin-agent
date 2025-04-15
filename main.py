from typing import Literal
from typing_extensions import TypedDict


from langchain_ollama.chat_models import ChatOllama
from langgraph.types import Command

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import create_react_agent



def create_graph():
    """Create a graph for the conv-fin-agent.

    This is a multi-agent graph that uses a supervisor to route tasks to the different members
    data_extractor and maths_solver. The data_extractor is responsible for extracting the relevant data from the user's input
    """

    members = ["data_extractor", "maths_solver"]

    # Our team supervisor is an LLM node. It just picks the next agent to do work
    # and decides when the work is completed
    options = members + ["FINISH"]

    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    # Only certain models on Ollama support tool use
    llm = ChatOllama(model="llama3.2:3b")

    class State(MessagesState):
        next: str

    def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
        messages = [
                       {"role": "system", "content": system_prompt},
                   ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})


    data_extraction_agent = create_react_agent(
        llm, tools = [], prompt="You are a data extraction expert for financial services data. You identify the relevant information to answer the question. DO NOT do any math."
    )


    def data_extractor_node(state: State) -> Command[Literal["supervisor"]]:
        result = data_extraction_agent.invoke(state)
        return Command(
            update={
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name="data_extractor")
                ]
            },
            goto="supervisor",
        )

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


    maths_agent = create_react_agent(llm, tools=[add, subtract, multiply, divide])


    def maths_node(state: State) -> Command[Literal["supervisor"]]:
        result = maths_agent.invoke(state)
        return Command(
            update={
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name="maths_solver")
                ]
            },
            goto="supervisor",
        )


    builder = StateGraph(State)
    builder.add_edge(START, "supervisor")
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("data_extractor", data_extractor_node)
    builder.add_node("maths_solver", maths_node)
    graph = builder.compile()
    return graph