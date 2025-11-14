# Section 10: Agents 🤖

Agents are LLM-powered systems that can reason, decide which tools to use, and take actions autonomously to accomplish goals.

---

## 🎯 What are Agents?

**Agent** = LLM + Tools + Reasoning Loop

**Without Agent (Manual Tool Calling):**
```python
# You manually orchestrate everything
response = llm.invoke("What's 5+3 and weather in NYC?")
# Execute tool 1
# Send result back
# Execute tool 2
# Send result back
# Get final answer
# 😰 Complex and error-prone!
```

**With Agent:**
```python
# Agent handles everything automatically
agent.invoke("What's 5+3 and weather in NYC?")
# ✅ Agent:
#   1. Calls calculator tool
#   2. Calls weather tool
#   3. Combines results
#   4. Returns answer
```

---

## 🔄 Agent Architecture

```
User Input
    ↓
Agent (LLM with reasoning)
    ↓
Thought: "I need to use the calculator tool"
    ↓
Action: calculator(5+3)
    ↓
Observation: 8
    ↓
Thought: "Now I need weather data"
    ↓
Action: weather_tool("NYC")
    ↓
Observation: "75°F, Sunny"
    ↓
Thought: "I have all information"
    ↓
Final Answer: "5+3=8. Weather in NYC is 75°F and sunny."
```

This is called the **ReAct** pattern: **Reasoning + Acting**

---

## 📋 Types of Agents

| Agent Type | Description | Best For |
|-----------|-------------|----------|
| **create_react_agent** | ReAct pattern (modern) | General purpose, recommended |
| **create_tool_calling_agent** | Uses native function calling | OpenAI/Anthropic models |
| **create_structured_chat_agent** | Handles complex inputs | Multi-modal, structured data |
| **create_openai_functions_agent** | Legacy OpenAI functions | Older systems (deprecated) |
| **create_json_agent** | Works with JSON | API interactions |

---

## 💻 Example 1: Basic ReAct Agent

```python
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# Define tools
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

# Create LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Get ReAct prompt from hub
prompt = hub.pull("hwchase17/react")

# Create agent
tools = [get_word_length, multiply]
agent = create_react_agent(llm, tools, prompt)

# Create agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # Shows reasoning steps
    handle_parsing_errors=True
)

# Run agent
result = agent_executor.invoke({
    "input": "What is the length of the word 'intelligence' multiplied by 3?"
})

print("\n" + "="*50)
print("Final Answer:", result["output"])

# Output shows:
# Thought: I need to find the length of 'intelligence'
# Action: get_word_length
# Action Input: intelligence
# Observation: 12
# Thought: Now I need to multiply by 3
# Action: multiply
# Action Input: 12, 3
# Observation: 36
# Thought: I now know the final answer
# Final Answer: 36
```

---

## 🔧 Example 2: Tool Calling Agent (Modern, Recommended)

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# Define tools
@tool
def search_database(query: str) -> str:
    """Search the product database."""
    products = {
        "laptop": "MacBook Pro - $1999",
        "phone": "iPhone 15 - $999",
        "tablet": "iPad Air - $599"
    }
    return products.get(query.lower(), "Product not found")

@tool
def calculate_tax(amount: float, rate: float = 0.08) -> float:
    """Calculate tax on an amount."""
    return round(amount * rate, 2)

@tool
def check_inventory(product: str) -> str:
    """Check if product is in stock."""
    inventory = {
        "laptop": "In stock (5 units)",
        "phone": "In stock (12 units)",
        "tablet": "Out of stock"
    }
    return inventory.get(product.lower(), "Product not found")

# Create LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful shopping assistant. Use the available tools to help customers."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),  # For agent reasoning
])

# Create agent
tools = [search_database, calculate_tax, check_inventory]
agent = create_tool_calling_agent(llm, tools, prompt)

# Create executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

# Test
result = agent_executor.invoke({
    "input": "I want to buy a laptop. Is it in stock and how much is the tax?"
})

print("\n" + "="*50)
print("Answer:", result["output"])

# Agent will:
# 1. Call search_database("laptop") → $1999
# 2. Call check_inventory("laptop") → In stock
# 3. Call calculate_tax(1999) → $159.92
# 4. Combine results into final answer
```

---

## 🎨 Example 3: Agent with Memory (Conversational Agent)

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.memory import ConversationBufferMemory

# Tools
@tool
def add_to_cart(product: str, quantity: int = 1) -> str:
    """Add a product to shopping cart."""
    return f"Added {quantity} x {product} to cart"

@tool
def view_cart() -> str:
    """View current shopping cart."""
    return "Cart: 2x Laptop, 1x Mouse"  # Simplified

@tool
def get_price(product: str) -> str:
    """Get product price."""
    prices = {"laptop": "$999", "mouse": "$29", "keyboard": "$79"}
    return prices.get(product.lower(), "Not found")

# Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Prompt with memory
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful shopping assistant. Remember the conversation context."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Agent
tools = [add_to_cart, view_cart, get_price]
agent = create_tool_calling_agent(llm, tools, prompt)

# Executor with memory
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# Conversation
print("Conversation 1:")
result1 = agent_executor.invoke({"input": "Add 2 laptops to my cart"})
print("Response:", result1["output"])

print("\n" + "="*50)
print("Conversation 2:")
result2 = agent_executor.invoke({"input": "How much would that cost?"})
print("Response:", result2["output"])

print("\n" + "="*50)
print("Conversation 3:")
result3 = agent_executor.invoke({"input": "What's in my cart?"})
print("Response:", result3["output"])

# Agent remembers:
# - "that" refers to the 2 laptops
# - Previous cart additions
```

---

## 🔍 Example 4: Agent with Web Search

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

# Web search tool
search = DuckDuckGoSearchRun()

# Custom tool
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# LLM and prompt
llm = ChatOpenAI(model="gpt-4", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful research assistant. Use search for current information and calculator for math."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Agent
tools = [search, calculator]
agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

# Test with current information
result = agent_executor.invoke({
    "input": "What is the current population of Tokyo and what's 10% of that?"
})

print("\n" + "="*50)
print("Answer:", result["output"])

# Agent will:
# 1. Search for Tokyo population
# 2. Extract the number
# 3. Calculate 10% using calculator
# 4. Return the result
```

---

## 📊 Example 5: Structured Chat Agent (Complex Inputs)

```python
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Tool with complex input schema
class AnalysisInput(BaseModel):
    data: str = Field(description="The data to analyze")
    metric: str = Field(description="The metric to calculate: mean, median, or sum")
    filter_value: float = Field(default=0, description="Filter values greater than this")

@tool(args_schema=AnalysisInput)
def analyze_data(data: str, metric: str, filter_value: float = 0) -> str:
    """Analyze numerical data with specified metric and optional filtering."""
    try:
        # Parse data
        numbers = [float(x.strip()) for x in data.split(",")]
        
        # Filter
        if filter_value > 0:
            numbers = [n for n in numbers if n > filter_value]
        
        # Calculate metric
        if metric == "mean":
            result = sum(numbers) / len(numbers)
        elif metric == "median":
            sorted_nums = sorted(numbers)
            n = len(sorted_nums)
            result = sorted_nums[n//2] if n % 2 == 1 else (sorted_nums[n//2-1] + sorted_nums[n//2]) / 2
        elif metric == "sum":
            result = sum(numbers)
        else:
            return f"Unknown metric: {metric}"
        
        return f"{metric.title()}: {result}"
    
    except Exception as e:
        return f"Error: {str(e)}"

# LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a data analysis assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Agent
tools = [analyze_data]
agent = create_structured_chat_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# Test
result = agent_executor.invoke({
    "input": "What's the mean of these numbers: 10, 20, 5, 30, 15, 8, but only include values greater than 10?"
})

print("\n" + "="*50)
print("Answer:", result["output"])
```

---

## 🔄 Example 6: Agent with Early Stopping

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
import time

@tool
def slow_operation(task: str) -> str:
    """A slow operation that takes time."""
    time.sleep(2)  # Simulate slow operation
    return f"Completed: {task}"

@tool
def quick_operation(task: str) -> str:
    """A quick operation."""
    return f"Quickly done: {task}"

llm = ChatOpenAI(model="gpt-4", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a task manager."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

tools = [slow_operation, quick_operation]
agent = create_tool_calling_agent(llm, tools, prompt)

# Agent executor with limits
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=3,  # Stop after 3 iterations
    max_execution_time=5,  # Stop after 5 seconds
    early_stopping_method="generate"  # or "force" - how to stop
)

# Test
try:
    result = agent_executor.invoke({
        "input": "Perform 5 slow operations"
    })
    print("Result:", result["output"])
except Exception as e:
    print(f"Agent stopped: {e}")
```

---

## 🎯 Example 7: Agent with Custom Callback

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Dict

# Custom callback to log agent actions
class LoggingCallback(BaseCallbackHandler):
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when tool starts."""
        print(f"\n🔧 Tool Started: {serialized.get('name', 'Unknown')}")
        print(f"   Input: {input_str}")
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when tool ends."""
        print(f"   Output: {output}")
    
    def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Called when tool errors."""
        print(f"   ❌ Error: {str(error)}")
    
    def on_agent_action(self, action, **kwargs) -> None:
        """Called when agent takes an action."""
        print(f"\n🤖 Agent Action: {action.tool}")
    
    def on_agent_finish(self, finish, **kwargs) -> None:
        """Called when agent finishes."""
        print(f"\n✅ Agent Finished")

# Tools
@tool
def fetch_data(source: str) -> str:
    """Fetch data from a source."""
    return f"Data from {source}: [1, 2, 3, 4, 5]"

@tool
def process_data(data: str) -> str:
    """Process the data."""
    return f"Processed: {data}"

# Setup agent
llm = ChatOpenAI(model="gpt-4", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a data processing assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

tools = [fetch_data, process_data]
agent = create_tool_calling_agent(llm, tools, prompt)

# Create callback
callback = LoggingCallback()

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,  # Our callback handles logging
    callbacks=[callback]
)

# Run
result = agent_executor.invoke({
    "input": "Fetch data from 'database' and process it"
})

print("\n" + "="*50)
print("Final Result:", result["output"])
```

---

## 🗃️ Example 8: Agent with RAG (Retrieval Tools)

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Create knowledge base
documents = [
    Document(page_content="Python was created by Guido van Rossum in 1991."),
    Document(page_content="Python is widely used for data science and machine learning."),
    Document(page_content="Popular Python web frameworks include Django and Flask."),
    Document(page_content="Python uses indentation for code blocks instead of braces."),
]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# Create retrieval tool
@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information about Python."""
    docs = vectorstore.similarity_search(query, k=2)
    return "\n".join([doc.page_content for doc in docs])

# Calculator tool
@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"

# Agent setup
llm = ChatOpenAI(model="gpt-4", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the knowledge base for Python questions and calculator for math."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

tools = [search_knowledge_base, calculate]
agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# Test
questions = [
    "Who created Python?",
    "What is Python used for?",
    "What's 25 * 37?",
    "Tell me about Python's syntax and calculate 100 / 4"
]

for question in questions:
    print(f"\nQ: {question}")
    result = agent_executor.invoke({"input": question})
    print(f"A: {result['output']}")
    print("-" * 50)
```

---

## 🔧 Example 9: Multi-Step Agent (Complex Reasoning)

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# Tools for multi-step task
@tool
def get_user_preferences(user_id: int) -> str:
    """Get user preferences."""
    prefs = {
        1: {"cuisine": "Italian", "budget": "medium", "dietary": "vegetarian"},
        2: {"cuisine": "Japanese", "budget": "high", "dietary": "none"}
    }
    return str(prefs.get(user_id, {}))

@tool
def search_restaurants(cuisine: str, budget: str) -> str:
    """Search restaurants by cuisine and budget."""
    restaurants = {
        ("Italian", "medium"): ["Pasta Paradise", "Luigi's Kitchen"],
        ("Japanese", "high"): ["Sushi Master", "Tokyo Garden"],
        ("Italian", "high"): ["Il Ristorante"]
    }
    results = restaurants.get((cuisine, budget), [])
    return ", ".join(results) if results else "No restaurants found"

@tool
def filter_by_dietary(restaurants: str, restriction: str) -> str:
    """Filter restaurants by dietary restrictions."""
    # Simplified filtering
    if restriction == "vegetarian":
        return f"Vegetarian-friendly options in: {restaurants}"
    return restaurants

@tool
def get_reviews(restaurant: str) -> str:
    """Get reviews for a restaurant."""
    reviews = {
        "Pasta Paradise": "4.5 stars - Great vegetarian options",
        "Luigi's Kitchen": "4.2 stars - Authentic Italian",
        "Sushi Master": "4.8 stars - Best sushi in town"
    }
    return reviews.get(restaurant, "No reviews available")

# Agent
llm = ChatOpenAI(model="gpt-4", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a restaurant recommendation assistant. 
    To recommend a restaurant:
    1. Get user preferences
    2. Search restaurants matching preferences
    3. Filter by dietary restrictions if needed
    4. Get reviews for top options
    5. Make a recommendation"""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

tools = [get_user_preferences, search_restaurants, filter_by_dietary, get_reviews]
agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10
)

# Test
result = agent_executor.invoke({
    "input": "Recommend a restaurant for user 1"
})

print("\n" + "="*50)
print("Recommendation:", result["output"])

# Agent will:
# 1. Get user 1's preferences → Italian, medium, vegetarian
# 2. Search Italian + medium restaurants
# 3. Filter for vegetarian options
# 4. Get reviews
# 5. Make recommendation
```

---

## 🎨 Example 10: Agent with Return Intermediate Steps

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# Tools
@tool
def get_temperature(city: str) -> str:
    """Get temperature for a city."""
    temps = {"NYC": "72°F", "LA": "85°F", "Chicago": "68°F"}
    return temps.get(city, "Unknown")

@tool
def convert_to_celsius(fahrenheit: str) -> str:
    """Convert Fahrenheit to Celsius."""
    try:
        f = float(fahrenheit.replace("°F", ""))
        c = (f - 32) * 5/9
        return f"{c:.1f}°C"
    except:
        return "Conversion error"

# Agent
llm = ChatOpenAI(model="gpt-4", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a weather assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

tools = [get_temperature, convert_to_celsius]
agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True  # Return all steps!
)

# Run
result = agent_executor.invoke({
    "input": "What's the temperature in NYC in Celsius?"
})

print("\n" + "="*50)
print("Final Answer:", result["output"])
print("\n" + "="*50)
print("Intermediate Steps:")
for step in result["intermediate_steps"]:
    action, observation = step
    print(f"\nAction: {action.tool}")
    print(f"Input: {action.tool_input}")
    print(f"Output: {observation}")
```

---

## 🔥 Best Practices

### **1. Choose the Right Agent Type**
```python
# ✅ General purpose - use create_tool_calling_agent
agent = create_tool_calling_agent(llm, tools, prompt)

# ✅ Need reasoning traces - use create_react_agent
agent = create_react_agent(llm, tools, prompt)

# ✅ Complex structured inputs - use create_structured_chat_agent
agent = create_structured_chat_agent(llm, tools, prompt)
```

### **2. Set Reasonable Limits**
```python
# ✅ Prevent infinite loops
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,  # Max reasoning steps
    max_execution_time=60,  # Max seconds
    early_stopping_method="generate"
)
```

### **3. Use verbose=True for Debugging**
```python
# ✅ During development
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ✅ In production
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
```

### **4. Handle Errors Gracefully**
```python
# ✅ Handle parsing errors
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True  # Don't crash on malformed output
)
```

### **5. Provide Clear System Prompts**
```python
# ✅ Guide agent behavior
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Rules:
    1. Always search knowledge base before answering
    2. Use calculator for all math
    3. Be concise in responses"""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
```

### **6. Add Memory for Conversations**
```python
# ✅ For conversational agents
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory
)
```

---

## 📊 Agent vs Manual Tool Calling

| Aspect | Manual Tool Calling | Agent |
|--------|-------------------|-------|
| Complexity | High | Low |
| Reasoning | Manual | Automatic |
| Multi-step | Hard to implement | Built-in |
| Error handling | Manual | Automatic |
| Iteration limit | Manual | Built-in |
| Best for | Simple, predictable tasks | Complex, dynamic tasks |

---

## 🎯 Practical Exercise

```python
"""
Exercise: Build a Research Assistant Agent

Create an agent that can:
1. Search the web for information
2. Save findings to a knowledge base
3. Query the knowledge base
4. Summarize multiple sources
5. Generate a research report

Tools needed:
- Web search tool
- Knowledge base storage tool
- Knowledge base query tool
- Summarization tool
- Report generation tool

Test with:
"Research the top 3 trends in AI for 2024 and create a summary report"

The agent should:
1. Search for "AI trends 2024"
2. Save findings to knowledge base
3. Query knowledge base for details
4. Summarize the information
5. Generate a formatted report
"""

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# Knowledge base (in-memory for this example)
knowledge_base = []

# TODO: Implement web search tool
search_tool = DuckDuckGoSearchRun()

# TODO: Implement save to knowledge base tool
@tool
def save_to_knowledge_base(content: str, source: str) -> str:
    """Save content to the knowledge base with source attribution."""
    pass

# TODO: Implement query knowledge base tool
@tool
def query_knowledge_base(query: str) -> str:
    """Query the knowledge base for relevant information."""
    pass

# TODO: Implement summarization tool
@tool
def summarize_content(content: str, max_length: int = 200) -> str:
    """Summarize content to specified length."""
    pass

# TODO: Implement report generation tool
@tool
def generate_report(title: str, sections: str) -> str:
    """Generate a formatted report with title and sections."""
    pass

# TODO: Create agent with all tools
# TODO: Test with research task

# Expected behavior:
# 1. Agent searches for AI trends
# 2. Saves results to knowledge base
# 3. Queries KB for specific trends
# 4. Summarizes each trend
# 5. Generates final report
```

---

## ✅ Key Takeaways

1. **Agents automate tool calling** - no manual orchestration needed
2. **ReAct pattern** - Reasoning + Acting in a loop
3. **create_tool_calling_agent** - modern, recommended approach
4. **AgentExecutor runs the agent** - handles iteration, errors
5. **Set max_iterations** - prevent infinite loops
6. **Use verbose=True** - see agent reasoning during development
7. **Add memory for conversations** - maintain context
8. **Agents can use multiple tools** - orchestrates them intelligently
9. **Handle errors gracefully** - set handle_parsing_errors=True
10. **Return intermediate steps** - for debugging and transparency

---

## 📝 Understanding Check

1. What's the difference between an agent and manual tool calling?
2. What is the ReAct pattern?
3. Why set max_iterations on an agent?
4. How do you add memory to an agent?

**Ready for Section 11 on Advanced RAG Techniques?** We'll explore query rewriting, reranking, HyDE, and more advanced patterns! 🚀

Or would you like to:
- See the exercise solution?
- Practice more with agents?
- Ask questions about specific agent types?