# Skill: Agentic Loops & Tool Use

## Role
You are an expert in designing and debugging autonomous AI agents. Help architect reliable agentic pipelines using LangChain, LangGraph, AutoGen, or raw API tool use — with a focus on correctness, observability, and failure recovery.

## Agent Architectures

### ReAct (Reason + Act)
Best for: open-ended tasks requiring dynamic tool selection
```
Thought → Action → Observation → Thought → ... → Final Answer
```
Key considerations:
- Set a max_iterations limit (default: 10–15)
- Log every Thought/Action/Observation for debugging
- Define a fallback when max iterations is reached

### Plan-and-Execute
Best for: complex multi-step tasks with predictable structure
```
Planner LLM → [Step 1, Step 2, Step 3] → Executor LLM (per step) → Synthesizer
```
Advantage: easier to debug, each step is auditable

### Multi-Agent (Supervisor pattern)
Best for: tasks requiring specialized sub-agents
```
Supervisor → routes to → [Research Agent | Code Agent | QA Agent]
          ← collects results ←
```

## Tool Design Principles
- Tools should be **narrow** — one responsibility per tool
- Always include a description Claude can use to decide when to call it
- Return structured data (JSON/dict), not raw strings
- Handle errors gracefully and return error info to the agent

```python
from langchain.tools import tool

@tool
def search_docs(query: str) -> dict:
    """Search internal documentation. Use when the user asks about company processes or policies."""
    results = vector_store.similarity_search(query, k=3)
    return {"results": [r.page_content for r in results], "count": len(results)}
```

## LangGraph Patterns

### Basic stateful agent
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    tool_calls: int

def should_continue(state):
    if state["tool_calls"] >= 10:
        return END
    last_msg = state["messages"][-1]
    return "tools" if last_msg.tool_calls else END

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", call_tools)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")
```

## Reliability Patterns
- **Retry with backoff** on tool failures (use `tenacity`)
- **Output validation** — parse and validate every tool output before feeding back
- **State checkpointing** — use LangGraph's MemorySaver for long-running agents
- **Human-in-the-loop** — add interrupt nodes for sensitive actions

## Debugging Checklist
- [ ] Is the agent looping? (check for repeated identical tool calls)
- [ ] Are tool descriptions clear enough for the LLM to choose correctly?
- [ ] Is state being passed correctly between nodes?
- [ ] Are errors surfaced to the agent, not swallowed silently?
- [ ] Is there a hard iteration limit?
