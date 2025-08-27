from typing import TypedDict, Annotated, Optional
from langgraph.graph import add_messages, StateGraph, END
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from uuid import uuid4
import os
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from tavily import TavilyClient

load_dotenv()

# --- Tool Definition ---
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

@tool
def search(query: str) -> str:
    """Searches the web for the user's query using the Tavily search engine."""
    try:
        results = tavily.search(query=query, max_results=4)
        return "\n\n".join(
            [f"Source URL: {res['url']}\nContent: {res['content']}" for res in results["results"]]
        )
    except Exception as e:
        return f"Error running search: {e}"

# --- Model and Tool Schema ---
# MODIFICATION: Corrected typo in model name
llm = ChatGroq(model="openai/gpt-oss-20b")

tools = [search]
tool_schemas = [convert_to_openai_tool(t) for t in tools]


# --- State and Memory ---
memory = MemorySaver()
class State(TypedDict):
    messages: Annotated[list, add_messages]


# --- Graph Nodes ---
async def model(state: State):
    """
    This node manually calls the LLM with the tools schema.
    """
    response = await llm.ainvoke(
        state["messages"],
        tools=tool_schemas,
        tool_choice="auto"
    )
    return {"messages": [response]}

async def tool_node(state: State):
    """
    Executes the search tool call requested by the model.
    """
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = []
    for tool_call in tool_calls:
        if tool_call["name"] == "search":
            search_results = await search.ainvoke(tool_call["args"])
            tool_messages.append(ToolMessage(
                content=str(search_results),
                tool_call_id=tool_call["id"],
            ))
    return {"messages": tool_messages}

def tools_router(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_node"
    else:
        return END

# --- Graph Definition ---
graph_builder = StateGraph(State)
graph_builder.add_node("model", model)
graph_builder.add_node("tool_node", tool_node)
graph_builder.set_entry_point("model")
graph_builder.add_conditional_edges("model", tools_router)
graph_builder.add_edge("tool_node", "model")
graph = graph_builder.compile(checkpointer=memory)


# --- FastAPI App ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

async def generate_chat_responses(message: str, checkpoint_id: Optional[str] = None):
    is_new_conversation = not checkpoint_id
    if is_new_conversation:
        checkpoint_id = str(uuid4())
        yield f"data: {json.dumps({'type': 'checkpoint', 'checkpoint_id': checkpoint_id})}\n\n"

    config = {"configurable": {"thread_id": checkpoint_id}}
    
    async for chunk in graph.astream(
        {"messages": [HumanMessage(content=message)]},
        config=config
    ):
        if "model" in chunk:
            last_message = chunk["model"]["messages"][-1]
            if last_message.tool_calls:
                tool_name = last_message.tool_calls[0]['name']
                tool_query = last_message.tool_calls[0]['args'].get('query', '')
                
                if tool_name == 'search':
                    # MODIFICATION: This is the direct fix for the SyntaxError.
                    # We construct the content string first to avoid backslashes in the f-string expression.
                    content = f'Searching for: "{tool_query}"...'
                    payload = {"type": "tool_call", "content": content}
                    yield f"data: {json.dumps(payload)}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'content', 'content': last_message.content})}\n\n"

    yield f"data: {json.dumps({'type': 'end'})}\n\n"


@app.get("/chat_stream/{message}")
async def chat_stream(message: str, checkpoint_id: Optional[str] = Query(None)):
    return StreamingResponse(
        generate_chat_responses(message, checkpoint_id), 
        media_type="text/event-stream"
    )