# Section 12: LangGraph 🕸️

LangGraph enables building stateful, multi-actor applications with cycles, branching, and complex control flow using graph structures.

---

## 🎯 What is LangGraph?

**LangGraph** = Build LLM applications as graphs with nodes, edges, and state

**Why LangGraph?**

Traditional chains and agents are linear or have limited branching:
```python
# Chain: A → B → C (linear)
# Agent: Loop until done (limited control)
```

LangGraph enables:
```python
# Graphs: Complex workflows
#   A → B → C
#   ↑   ↓   ↓
#   D ← E → F
# - Cycles (loops)
# - Conditional branching
# - Human-in-the-loop
# - Parallel execution
# - State persistence
```

---

## 🏗️ Core Concepts

### **1. StateGraph**
The graph that manages state and execution flow

### **2. Nodes**
Functions that process state (can be LLM calls, tool calls, or any logic)

### **3. Edges**
Connections between nodes (normal or conditional)

### **4. State**
Shared data structure passed between nodes

### **5. Checkpoints**
Save and resume state at any point

---

## 📦 Installation

```python
# Install LangGraph
# pip install langgraph

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
```

---

## 💻 Example 1: Simple Linear Graph

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Define state
class GraphState(TypedDict):
    input: str
    output: str
    steps: list

# Define nodes
def node_a(state: GraphState) -> GraphState:
    """First node - process input."""
    print("Node A: Processing input")
    return {
        **state,
        "steps": state.get("steps", []) + ["Node A"],
        "output": f"Processed: {state['input']}"
    }

def node_b(state: GraphState) -> GraphState:
    """Second node - enhance output."""
    print("Node B: Enhancing output")
    return {
        **state,
        "steps": state.get("steps", []) + ["Node B"],
        "output": state["output"].upper()
    }

def node_c(state: GraphState) -> GraphState:
    """Third node - finalize."""
    print("Node C: Finalizing")
    return {
        **state,
        "steps": state.get("steps", []) + ["Node C"],
        "output": f"Final: {state['output']}"
    }

# Build graph
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("node_a", node_a)
workflow.add_node("node_b", node_b)
workflow.add_node("node_c", node_c)

# Add edges
workflow.set_entry_point("node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", "node_c")
workflow.add_edge("node_c", END)

# Compile
app = workflow.compile()

# Run
result = app.invoke({"input": "hello world", "steps": []})

print("\n" + "="*50)
print("Final Result:")
print(f"Output: {result['output']}")
print(f"Steps: {result['steps']}")

# Output:
# Node A: Processing input
# Node B: Enhancing output
# Node C: Finalizing
# Final Result:
# Output: Final: PROCESSED: HELLO WORLD
# Steps: ['Node A', 'Node B', 'Node C']
```

---

## 🔀 Example 2: Conditional Branching

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

# State
class GraphState(TypedDict):
    messages: list
    sentiment: str
    response: str

# Nodes
def analyze_sentiment(state: GraphState) -> GraphState:
    """Analyze sentiment of user message."""
    message = state["messages"][-1]
    
    # Simple sentiment analysis
    if any(word in message.lower() for word in ["happy", "great", "love", "excellent"]):
        sentiment = "positive"
    elif any(word in message.lower() for word in ["sad", "bad", "hate", "terrible"]):
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    print(f"Sentiment: {sentiment}")
    return {**state, "sentiment": sentiment}

def handle_positive(state: GraphState) -> GraphState:
    """Handle positive sentiment."""
    print("Handling positive sentiment")
    return {**state, "response": "That's wonderful! I'm glad to hear that!"}

def handle_negative(state: GraphState) -> GraphState:
    """Handle negative sentiment."""
    print("Handling negative sentiment")
    return {**state, "response": "I'm sorry to hear that. How can I help?"}

def handle_neutral(state: GraphState) -> GraphState:
    """Handle neutral sentiment."""
    print("Handling neutral sentiment")
    return {**state, "response": "I understand. What would you like to know?"}

# Conditional routing function
def route_by_sentiment(state: GraphState) -> Literal["positive", "negative", "neutral"]:
    """Route to appropriate handler based on sentiment."""
    return state["sentiment"]

# Build graph
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("analyze", analyze_sentiment)
workflow.add_node("positive", handle_positive)
workflow.add_node("negative", handle_negative)
workflow.add_node("neutral", handle_neutral)

# Add edges
workflow.set_entry_point("analyze")

# Conditional edge - routes based on sentiment
workflow.add_conditional_edges(
    "analyze",  # From this node
    route_by_sentiment,  # Use this function to decide
    {
        "positive": "positive",  # If returns "positive", go to "positive" node
        "negative": "negative",
        "neutral": "neutral"
    }
)

# All handlers end the graph
workflow.add_edge("positive", END)
workflow.add_edge("negative", END)
workflow.add_edge("neutral", END)

# Compile
app = workflow.compile()

# Test different sentiments
test_messages = [
    "I love this product!",
    "This is terrible",
    "Can you help me?"
]

for msg in test_messages:
    print(f"\n{'='*50}")
    print(f"Input: {msg}")
    print("="*50)
    
    result = app.invoke({"messages": [msg]})
    print(f"Response: {result['response']}")
```

---

## 🔄 Example 3: Graph with Cycles (Loop)

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

# State
class GraphState(TypedDict):
    count: int
    numbers: list
    sum: int

# Nodes
def add_number(state: GraphState) -> GraphState:
    """Add current count to list."""
    count = state["count"]
    numbers = state.get("numbers", [])
    
    print(f"Adding number: {count}")
    
    return {
        **state,
        "numbers": numbers + [count],
        "count": count + 1
    }

def calculate_sum(state: GraphState) -> GraphState:
    """Calculate sum of all numbers."""
    total = sum(state["numbers"])
    print(f"Sum: {total}")
    return {**state, "sum": total}

# Routing function
def should_continue(state: GraphState) -> Literal["add_number", "calculate_sum"]:
    """Decide whether to continue adding or calculate sum."""
    if state["count"] <= 5:
        return "add_number"  # Loop back
    else:
        return "calculate_sum"  # Exit loop

# Build graph
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("add_number", add_number)
workflow.add_node("calculate_sum", calculate_sum)

# Add edges
workflow.set_entry_point("add_number")

# Conditional edge - creates a loop
workflow.add_conditional_edges(
    "add_number",
    should_continue,
    {
        "add_number": "add_number",  # Loop back to itself
        "calculate_sum": "calculate_sum"  # Exit to sum
    }
)

workflow.add_edge("calculate_sum", END)

# Compile
app = workflow.compile()

# Run
result = app.invoke({"count": 1, "numbers": []})

print("\n" + "="*50)
print("Final Result:")
print(f"Numbers: {result['numbers']}")
print(f"Sum: {result['sum']}")

# Output:
# Adding number: 1
# Adding number: 2
# Adding number: 3
# Adding number: 4
# Adding number: 5
# Sum: 15
```

---

## 🤖 Example 4: Agent with LangGraph

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
import operator

# Define tools
@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

tools = [multiply, add]

# State with message list
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # Append messages

# LLM with tools
llm = ChatOpenAI(model="gpt-4", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# Agent node
def call_model(state: AgentState) -> AgentState:
    """Call the LLM."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    print(f"LLM response: {response.content}")
    if response.tool_calls:
        print(f"Tool calls: {response.tool_calls}")
    return {"messages": [response]}

# Tool execution node
def execute_tools(state: AgentState) -> AgentState:
    """Execute tool calls."""
    last_message = state["messages"][-1]
    tool_results = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Find and execute tool
        tool_map = {tool.name: tool for tool in tools}
        tool = tool_map[tool_name]
        result = tool.invoke(tool_args)
        
        print(f"Tool {tool_name}({tool_args}) = {result}")
        
        tool_results.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )
    
    return {"messages": tool_results}

# Routing function
def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Check if there are tool calls to execute."""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    else:
        return "end"

# Build graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tools)

# Add edges
workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

# After tools, go back to agent
workflow.add_edge("tools", "agent")

# Compile
app = workflow.compile()

# Run
inputs = {"messages": [HumanMessage(content="What is (5 + 3) * 2?")]}
result = app.invoke(inputs)

print("\n" + "="*50)
print("Final Answer:")
print(result["messages"][-1].content)

# The agent will:
# 1. Decide to call add(5, 3)
# 2. Execute tool → 8
# 3. Decide to call multiply(8, 2)
# 4. Execute tool → 16
# 5. Generate final answer → 16
```

---

## 👤 Example 5: Human-in-the-Loop

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Literal

# State
class GraphState(TypedDict):
    query: str
    draft_response: str
    final_response: str
    approved: bool

# Nodes
def generate_draft(state: GraphState) -> GraphState:
    """Generate draft response."""
    query = state["query"]
    draft = f"Draft response to '{query}': This is an automated draft."
    print(f"Generated draft: {draft}")
    return {**state, "draft_response": draft}

def await_approval(state: GraphState) -> GraphState:
    """Wait for human approval."""
    print("\n⏸️  Waiting for human approval...")
    print(f"Draft: {state['draft_response']}")
    # In real app, this would be interrupted and wait for external input
    return state

def finalize_response(state: GraphState) -> GraphState:
    """Finalize after approval."""
    print("✅ Approved! Finalizing...")
    return {
        **state,
        "final_response": state["draft_response"],
        "approved": True
    }

def revise_response(state: GraphState) -> GraphState:
    """Revise if not approved."""
    print("❌ Not approved. Revising...")
    return {
        **state,
        "draft_response": f"REVISED: {state['draft_response']}"
    }

# Routing - in real app, this checks actual human input
def check_approval(state: GraphState) -> Literal["finalize", "revise"]:
    """Check if human approved."""
    # Simulate approval decision
    # In real app: check state["approved"] from external input
    import random
    approved = random.choice([True, False])
    print(f"Approval decision: {approved}")
    return "finalize" if approved else "revise"

# Build graph
workflow = StateGraph(GraphState)

workflow.add_node("generate_draft", generate_draft)
workflow.add_node("await_approval", await_approval)
workflow.add_node("finalize", finalize_response)
workflow.add_node("revise", revise_response)

workflow.set_entry_point("generate_draft")
workflow.add_edge("generate_draft", "await_approval")

workflow.add_conditional_edges(
    "await_approval",
    check_approval,
    {
        "finalize": "finalize",
        "revise": "revise"
    }
)

workflow.add_edge("revise", "await_approval")  # Loop back for re-approval
workflow.add_edge("finalize", END)

# Compile with checkpointer for state persistence
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Run
config = {"configurable": {"thread_id": "1"}}
result = app.invoke(
    {"query": "Explain machine learning"},
    config=config
)

print("\n" + "="*50)
print("Final Result:")
print(f"Approved: {result.get('approved', False)}")
print(f"Response: {result.get('final_response', 'N/A')}")
```

---

## 💾 Example 6: Persistent State (Checkpoints)

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict

# State
class GraphState(TypedDict):
    count: int
    messages: list

# Nodes
def increment(state: GraphState) -> GraphState:
    """Increment counter."""
    new_count = state.get("count", 0) + 1
    messages = state.get("messages", [])
    
    print(f"Count: {new_count}")
    
    return {
        "count": new_count,
        "messages": messages + [f"Incremented to {new_count}"]
    }

# Build graph
workflow = StateGraph(GraphState)
workflow.add_node("increment", increment)
workflow.set_entry_point("increment")
workflow.add_edge("increment", END)

# Compile with memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Thread ID identifies the conversation/session
thread_config = {"configurable": {"thread_id": "user_123"}}

# First invocation
print("First invocation:")
result1 = app.invoke({"count": 0}, config=thread_config)
print(f"Result: {result1}\n")

# Second invocation - continues from saved state
print("Second invocation (same thread):")
result2 = app.invoke({"count": result1["count"]}, config=thread_config)
print(f"Result: {result2}\n")

# Different thread - starts fresh
print("Third invocation (different thread):")
result3 = app.invoke({"count": 0}, config={"configurable": {"thread_id": "user_456"}})
print(f"Result: {result3}")

# State persists per thread!
```

---

## 🔧 Example 7: Parallel Execution

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict
import time

# State
class GraphState(TypedDict):
    input: str
    result_a: str
    result_b: str
    result_c: str
    final: str

# Parallel nodes
def task_a(state: GraphState) -> GraphState:
    """Task A - slow operation."""
    print("Task A started")
    time.sleep(1)
    print("Task A completed")
    return {**state, "result_a": f"A processed: {state['input']}"}

def task_b(state: GraphState) -> GraphState:
    """Task B - medium operation."""
    print("Task B started")
    time.sleep(0.5)
    print("Task B completed")
    return {**state, "result_b": f"B processed: {state['input']}"}

def task_c(state: GraphState) -> GraphState:
    """Task C - fast operation."""
    print("Task C started")
    time.sleep(0.2)
    print("Task C completed")
    return {**state, "result_c": f"C processed: {state['input']}"}

def combine_results(state: GraphState) -> GraphState:
    """Combine results from parallel tasks."""
    print("Combining results")
    final = f"Combined: {state['result_a']}, {state['result_b']}, {state['result_c']}"
    return {**state, "final": final}

# Build graph
workflow = StateGraph(GraphState)

workflow.add_node("task_a", task_a)
workflow.add_node("task_b", task_b)
workflow.add_node("task_c", task_c)
workflow.add_node("combine", combine_results)

# Set entry point to all three tasks (they run in parallel)
workflow.set_entry_point("task_a")
workflow.set_entry_point("task_b")
workflow.set_entry_point("task_c")

# All tasks converge to combine
workflow.add_edge("task_a", "combine")
workflow.add_edge("task_b", "combine")
workflow.add_edge("task_c", "combine")

workflow.add_edge("combine", END)

# Compile
app = workflow.compile()

# Run
start_time = time.time()
result = app.invoke({"input": "test data"})
elapsed = time.time() - start_time

print("\n" + "="*50)
print(f"Final result: {result['final']}")
print(f"Total time: {elapsed:.2f}s")
print("(Would be 1.7s if sequential, but parallel execution is faster)")
```

---

## 🎨 Example 8: Multi-Agent Collaboration

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# State
class CollaborationState(TypedDict):
    task: str
    research_findings: str
    draft_content: str
    reviewed_content: str
    final_output: str

# Agents (simulated with prompts)
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

def researcher_agent(state: CollaborationState) -> CollaborationState:
    """Research agent gathers information."""
    print("🔍 Researcher Agent working...")
    
    prompt = f"Research the following topic and provide key findings: {state['task']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {**state, "research_findings": response.content}

def writer_agent(state: CollaborationState) -> CollaborationState:
    """Writer agent creates content."""
    print("✍️ Writer Agent working...")
    
    prompt = f"""Based on these research findings, write a brief article:
    
    Research: {state['research_findings']}
    Topic: {state['task']}
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {**state, "draft_content": response.content}

def reviewer_agent(state: CollaborationState) -> CollaborationState:
    """Reviewer agent reviews and improves content."""
    print("👀 Reviewer Agent working...")
    
    prompt = f"""Review and improve this article:
    
    {state['draft_content']}
    
    Provide the improved version.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {**state, "reviewed_content": response.content}

def finalize(state: CollaborationState) -> CollaborationState:
    """Finalize the output."""
    print("✅ Finalizing...")
    return {**state, "final_output": state["reviewed_content"]}

# Build graph
workflow = StateGraph(CollaborationState)

workflow.add_node("researcher", researcher_agent)
workflow.add_node("writer", writer_agent)
workflow.add_node("reviewer", reviewer_agent)
workflow.add_node("finalize", finalize)

# Linear flow: research → write → review → finalize
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "reviewer")
workflow.add_edge("reviewer", "finalize")
workflow.add_edge("finalize", END)

# Compile
app = workflow.compile()

# Run
result = app.invoke({"task": "The impact of AI on healthcare"})

print("\n" + "="*50)
print("Final Output:")
print(result["final_output"])
```

---

## 🔄 Example 9: Streaming Execution

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

# State
class GraphState(TypedDict):
    messages: list
    count: int

# Nodes
def step_1(state: GraphState) -> GraphState:
    print("Step 1 executing...")
    return {
        **state,
        "messages": state.get("messages", []) + ["Step 1 complete"],
        "count": 1
    }

def step_2(state: GraphState) -> GraphState:
    print("Step 2 executing...")
    return {
        **state,
        "messages": state.get("messages", []) + ["Step 2 complete"],
        "count": 2
    }

def step_3(state: GraphState) -> GraphState:
    print("Step 3 executing...")
    return {
        **state,
        "messages": state.get("messages", []) + ["Step 3 complete"],
        "count": 3
    }

# Build graph
workflow = StateGraph(GraphState)

workflow.add_node("step_1", step_1)
workflow.add_node("step_2", step_2)
workflow.add_node("step_3", step_3)

workflow.set_entry_point("step_1")
workflow.add_edge("step_1", "step_2")
workflow.add_edge("step_2", "step_3")
workflow.add_edge("step_3", END)

# Compile
app = workflow.compile()

# Stream execution - see state after each node
print("Streaming execution:")
print("="*50)

for state in app.stream({"messages": [], "count": 0}):
    print(f"\nState update: {state}")

print("\n" + "="*50)
print("Execution complete")
```

---

## 📊 Example 10: Complex RAG with LangGraph

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Create knowledge base
documents = [
    Document(page_content="Python is a high-level programming language."),
    Document(page_content="Machine learning uses algorithms to learn from data."),
    Document(page_content="Neural networks are inspired by biological brains."),
]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

llm = ChatOpenAI(model="gpt-4", temperature=0)

# State
class RAGState(TypedDict):
    question: str
    retrieved_docs: list
    relevance_score: float
    answer: str
    needs_web_search: bool

# Nodes
def retrieve(state: RAGState) -> RAGState:
    """Retrieve documents."""
    print("📚 Retrieving documents...")
    docs = vectorstore.similarity_search(state["question"], k=3)
    return {**state, "retrieved_docs": docs}

def grade_documents(state: RAGState) -> RAGState:
    """Grade document relevance."""
    print("📊 Grading relevance...")
    
    # Simple grading (in production, use LLM)
    question = state["question"].lower()
    docs = state["retrieved_docs"]
    
    relevant_count = sum(
        1 for doc in docs 
        if any(word in doc.page_content.lower() for word in question.split())
    )
    
    score = relevant_count / len(docs) if docs else 0
    print(f"Relevance score: {score}")
    
    return {**state, "relevance_score": score}

def generate_answer(state: RAGState) -> RAGState:
    """Generate answer from documents."""
    print("💡 Generating answer...")
    
    context = "\n".join([doc.page_content for doc in state["retrieved_docs"]])
    
    prompt = ChatPromptTemplate.from_template("""
    Context: {context}
    Question: {question}
    Answer:
    """)
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "question": state["question"]
    })
    
    return {**state, "answer": answer}

def web_search(state: RAGState) -> RAGState:
    """Fallback to web search."""
    print("🌐 Performing web search...")
    # Simulated
    return {
        **state,
        "answer": f"Web search result for: {state['question']}"
    }

# Routing
def should_use_web_search(state: RAGState) -> Literal["generate", "web_search"]:
    """Decide if web search is needed."""
    if state["relevance_score"] < 0.3:
        print("⚠️ Low relevance - using web search")
        return "web_search"
    else:
        print("✅ Good relevance - generating answer")
        return "generate"

# Build graph
workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade_documents)
workflow.add_node("generate", generate_answer)
workflow.add_node("web_search", web_search)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")

workflow.add_conditional_edges(
    "grade",
    should_use_web_search,
    {
        "generate": "generate",
        "web_search": "web_search"
    }
)

workflow.add_edge("generate", END)
workflow.add_edge("web_search", END)

# Compile
app = workflow.compile()

# Test
questions = [
    "What is Python?",
    "What is quantum computing?"  # Not in knowledge base
]

for q in questions:
    print("\n" + "="*50)
    print(f"Question: {q}")
    print("="*50)
    
    result = app.invoke({"question": q})
    print(f"\nAnswer: {result['answer']}")
```

---

## 🔥 Best Practices

### **1. Keep State Minimal**
```python
# ❌ Bad - too much state
class State(TypedDict):
    data1: list
    data2: list
    data3: dict
    temp1: str
    temp2: str
    # ... 20 more fields

# ✅ Good - only what's needed
class State(TypedDict):
    input: str
    output: str
    metadata: dict
```

### **2. Use Typed State**
```python
# ✅ Always use TypedDict
from typing import TypedDict

class GraphState(TypedDict):
    field1: str
    field2: int
```

### **3. Keep Nodes Focused**
```python
# ✅ Each node does one thing
def retrieve_docs(state): ...
def grade_docs(state): ...
def generate_answer(state): ...

# ❌ Don't combine everything
def do_everything(state): ...
```

### **4. Use Checkpoints for Long Workflows**
```python
# ✅ Enable persistence
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

### **5. Handle Errors in Nodes**
```python
def my_node(state):
    try:
        # Node logic
        return new_state
    except Exception as e:
        return {**state, "error": str(e)}
```

---

## 📊 LangGraph vs Agents

| Feature | Traditional Agent | LangGraph |
|---------|------------------|-----------|
| Control Flow | Limited | Full control |
| Cycles | Basic loops | Complex cycles |
| Branching | Limited | Conditional routing |
| State Management | Limited | Full state control |
| Human-in-Loop | Difficult | Built-in |
| Persistence | Limited | Checkpoints |
| Complexity | Simple | Can be complex |
| Best For | Simple tasks | Complex workflows |

---

## 🎯 Practical Exercise

```python
"""
Exercise: Build a Research Assistant Graph

Create a multi-step research workflow:

1. Query Analysis Node
   - Classify query complexity
   - Extract key entities

2. Research Planning Node
   - Decide which sources to search
   - Generate search queries

3. Parallel Search Nodes
   - Web search
   - Knowledge base search
   - (Optionally) Academic papers

4. Synthesis Node
   - Combine results
   - Remove duplicates

5. Drafting Node
   - Generate draft answer

6. Review Node (Conditional)
   - Check quality
   - If poor, loop back to research
   - If good, proceed

7. Finalization Node
   - Format output
   - Add citations

Requirements:
- Use conditional edges for quality check
- Implement cycle for re-research if needed
- Use parallel execution for searches
- Add checkpointing
- Include human approval step

Test with: "What are the latest developments in quantum computing?"
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Literal

class ResearchState(TypedDict):
    query: str
    complexity: str
    search_queries: list
    web_results: str
    kb_results: str
    synthesis: str
    draft: str
    quality_score: float
    final_output: str
    iteration: int

# TODO: Implement all nodes
# TODO: Build graph with conditional edges and cycles
# TODO: Add checkpointing
# TODO: Test with sample query
```

---

## ✅ Key Takeaways

1. **LangGraph = Graphs for LLM apps** - nodes, edges, state
2. **StateGraph manages state** - passed between nodes
3. **Nodes are functions** - process and return state
4. **Conditional edges enable branching** - dynamic routing
5. **Cycles create loops** - for iterative workflows
6. **Checkpoints enable persistence** - save and resume
7. **Human-in-the-loop** - pause for approval
8. **Parallel execution** - run nodes concurrently
9. **Better than chains** - for complex workflows
10. **More control than agents** - explicit flow definition

---

## 📝 Understanding Check

1. What's the difference between a normal edge and a conditional edge?
2. How do you create a cycle in LangGraph?
3. What are checkpoints used for?
4. When would you use LangGraph over a traditional agent?

**Ready for Section 13 on LangSmith?** This is essential for monitoring, debugging, and improving your LLM applications in production! 📊

Or would you like to:
- See the exercise solution?
- Practice more with LangGraph?
- Deep dive into specific patterns?