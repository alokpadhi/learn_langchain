# Section 9: Tools and Function Calling 🔧

Tools enable LLMs to interact with external systems, APIs, databases, and perform actions beyond text generation.

---

## 🎯 What are Tools?

**Tools** = Functions that LLMs can call to perform specific tasks

**Without Tools:**
```python
llm.invoke("What's the weather in San Francisco?")
# ❌ "I don't have access to real-time weather data"
```

**With Tools:**
```python
# LLM can call a weather tool
llm_with_tools.invoke("What's the weather in San Francisco?")
# ✅ Calls weather_tool("San Francisco") → Returns "72°F, Sunny"
# ✅ "The weather in San Francisco is 72°F and sunny"
```

---

## 🔧 How Tools Work

```
User: "What's 25 * 37?"
    ↓
LLM: "I need to use the calculator tool"
    ↓
Tool Call: calculator(25 * 37)
    ↓
Tool Result: 925
    ↓
LLM: "25 * 37 equals 925"
    ↓
User gets answer
```

---

## 📋 Types of Tools

### **Built-in Tools:**
- DuckDuckGo Search
- Wikipedia
- Arxiv (research papers)
- Python REPL
- File system operations
- Shell commands

### **Custom Tools:**
- API calls
- Database queries
- Internal business logic
- External service integrations

---

## 💻 Example 1: Creating a Simple Tool

```python
from langchain_core.tools import tool

# Method 1: Using @tool decorator (simplest)
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

# Test the tool directly
result = multiply.invoke({"a": 5, "b": 7})
print(f"Tool result: {result}")  # 35

# Tool attributes
print(f"Tool name: {multiply.name}")  # multiply
print(f"Tool description: {multiply.description}")  # Multiply two numbers together.
print(f"Tool args: {multiply.args}")  # Schema of arguments

# Output:
# Tool result: 35
# Tool name: multiply
# Tool description: Multiply two numbers together.
```

---

## 🎨 Example 2: Tool with Type Hints and Descriptions

```python
from langchain_core.tools import tool
from typing import Optional

@tool
def search_user(
    user_id: int,
    include_details: Optional[bool] = False
) -> str:
    """
    Search for a user by their ID.
    
    Args:
        user_id: The unique identifier for the user
        include_details: Whether to include additional user details
    """
    # Simulate database lookup
    user = {
        "id": user_id,
        "name": "John Doe",
        "email": "john@example.com",
        "role": "developer"
    }
    
    if include_details:
        return f"User {user['name']} ({user['email']}) - Role: {user['role']}"
    else:
        return f"User {user['name']}"

# Test
result = search_user.invoke({"user_id": 123, "include_details": True})
print(result)

# The docstring becomes the tool description
# Type hints define the expected argument types
```

---

## 🔗 Example 3: Using Tools with LLMs (Function Calling)

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# Define tools
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Simulate weather API
    weather_data = {
        "San Francisco": "72°F, Sunny",
        "New York": "65°F, Cloudy",
        "London": "55°F, Rainy"
    }
    return weather_data.get(city, "Weather data not available")

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Bind tools to LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)
llm_with_tools = llm.bind_tools([get_weather, calculate])

# Invoke with a question that requires tool use
response = llm_with_tools.invoke("What's the weather in San Francisco?")

print("Response type:", type(response))
print("Content:", response.content)
print("Tool calls:", response.tool_calls)

# Output:
# Response type: <class 'langchain_core.messages.ai.AIMessage'>
# Content: 
# Tool calls: [
#     {
#         'name': 'get_weather',
#         'args': {'city': 'San Francisco'},
#         'id': 'call_abc123'
#     }
# ]

# The LLM decided to call the get_weather tool!
```

---

## ⚙️ Example 4: Executing Tool Calls

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

llm = ChatOpenAI(model="gpt-4", temperature=0)
llm_with_tools = llm.bind_tools([get_word_length])

# Step 1: User asks question
messages = [HumanMessage(content="How long is the word 'artificial'?")]

# Step 2: LLM decides to use tool
ai_message = llm_with_tools.invoke(messages)
print("AI wants to call:", ai_message.tool_calls)

# Step 3: Execute the tool
tool_call = ai_message.tool_calls[0]
tool_output = get_word_length.invoke(tool_call["args"])
print(f"Tool result: {tool_output}")

# Step 4: Send tool result back to LLM
messages.append(ai_message)
messages.append(ToolMessage(
    content=str(tool_output),
    tool_call_id=tool_call["id"]
))

# Step 5: LLM generates final response
final_response = llm_with_tools.invoke(messages)
print(f"\nFinal answer: {final_response.content}")

# Output:
# AI wants to call: [{'name': 'get_word_length', 'args': {'word': 'artificial'}, 'id': 'call_123'}]
# Tool result: 10
# Final answer: The word 'artificial' is 10 letters long.
```

---

## 🔄 Example 5: Tool Calling in a Loop (Manual Agent)

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Define tools
@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

# Setup
llm = ChatOpenAI(model="gpt-4", temperature=0)
tools = [add, multiply]
llm_with_tools = llm.bind_tools(tools)

# Create tool execution map
tools_map = {tool.name: tool for tool in tools}

# Agent loop
def run_agent(user_input: str, max_iterations: int = 5):
    messages = [HumanMessage(content=user_input)]
    
    for i in range(max_iterations):
        print(f"\n--- Iteration {i+1} ---")
        
        # Get LLM response
        ai_message = llm_with_tools.invoke(messages)
        messages.append(ai_message)
        
        # Check if LLM wants to call tools
        if not ai_message.tool_calls:
            print("No more tool calls needed")
            print(f"Final answer: {ai_message.content}")
            return ai_message.content
        
        # Execute each tool call
        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"Calling tool: {tool_name}({tool_args})")
            
            # Execute tool
            tool = tools_map[tool_name]
            tool_output = tool.invoke(tool_args)
            
            print(f"Tool result: {tool_output}")
            
            # Add tool result to messages
            messages.append(ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_call["id"]
            ))
    
    return "Max iterations reached"

# Test
result = run_agent("What is (5 + 3) * 2?")

# Output:
# --- Iteration 1 ---
# Calling tool: add({'a': 5, 'b': 3})
# Tool result: 8
#
# --- Iteration 2 ---
# Calling tool: multiply({'a': 8, 'b': 2})
# Tool result: 16
#
# --- Iteration 3 ---
# No more tool calls needed
# Final answer: (5 + 3) * 2 equals 16
```

---

## 📚 Example 6: StructuredTool (More Control)

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# Define input schema with Pydantic
class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    num_results: int = Field(default=5, description="Number of results to return")
    language: str = Field(default="en", description="Language for results")

# Define the function
def search_documents(query: str, num_results: int = 5, language: str = "en") -> str:
    """Search through documents and return relevant results."""
    # Simulate search
    results = [f"Result {i}: {query}" for i in range(1, num_results + 1)]
    return f"Found {num_results} results in {language}: " + ", ".join(results)

# Create structured tool
search_tool = StructuredTool.from_function(
    func=search_documents,
    name="search_documents",
    description="Search through the document database",
    args_schema=SearchInput,
    return_direct=False  # If True, returns tool output directly without LLM processing
)

# Test
result = search_tool.invoke({
    "query": "machine learning",
    "num_results": 3,
    "language": "en"
})
print(result)

# Inspect tool
print(f"\nTool name: {search_tool.name}")
print(f"Tool description: {search_tool.description}")
print(f"Tool args schema: {search_tool.args_schema.schema()}")
```

---

## 🌐 Example 7: Built-in Tools - DuckDuckGo Search

```python
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI

# Create search tool
search = DuckDuckGoSearchRun()

# Test tool directly
result = search.invoke("LangChain framework")
print("Search result:")
print(result[:300])  # First 300 chars

# Use with LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)
llm_with_search = llm.bind_tools([search])

# Ask a question requiring real-time info
response = llm_with_search.invoke("What are the latest developments in AI?")
print("\nLLM response with tool calls:")
print(response.tool_calls)

# Note: You need to execute the tool and send results back
# (We'll see automatic execution with Agents in the next section)
```

---

## 🐍 Example 8: Python REPL Tool

```python
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI

# Create Python REPL tool (executes Python code)
python_repl = PythonREPLTool()

# Test directly
result = python_repl.invoke("print([x**2 for x in range(10)])")
print("Direct execution:")
print(result)

# Use with LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)
llm_with_tools = llm.bind_tools([python_repl])

# Ask LLM to write and execute code
response = llm_with_tools.invoke(
    "Write Python code to calculate the factorial of 5"
)

if response.tool_calls:
    tool_call = response.tool_calls[0]
    print("\nLLM generated code:")
    print(tool_call["args"]["query"])
    
    # Execute the code
    code_result = python_repl.invoke(tool_call["args"])
    print(f"\nExecution result: {code_result}")

# ⚠️ WARNING: PythonREPLTool executes arbitrary code - use with caution!
```

---

## 🔧 Example 9: Multiple Tools Working Together

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# Define multiple related tools
@tool
def get_user_data(user_id: int) -> dict:
    """Get user information by ID."""
    users = {
        1: {"name": "Alice", "department": "Engineering", "manager_id": 3},
        2: {"name": "Bob", "department": "Sales", "manager_id": 3},
        3: {"name": "Charlie", "department": "Management", "manager_id": None}
    }
    return users.get(user_id, {})

@tool
def get_department_budget(department: str) -> float:
    """Get the budget for a department."""
    budgets = {
        "Engineering": 500000,
        "Sales": 300000,
        "Management": 200000
    }
    return budgets.get(department, 0)

@tool
def calculate_percentage(part: float, whole: float) -> float:
    """Calculate what percentage 'part' is of 'whole'."""
    if whole == 0:
        return 0
    return (part / whole) * 100

# Setup
llm = ChatOpenAI(model="gpt-4", temperature=0)
tools = [get_user_data, get_department_budget, calculate_percentage]
llm_with_tools = llm.bind_tools(tools)
tools_map = {tool.name: tool for tool in tools}

# Complex query requiring multiple tools
def run_multi_tool_query(question: str):
    messages = [HumanMessage(content=question)]
    
    for iteration in range(10):  # Max 10 iterations
        ai_message = llm_with_tools.invoke(messages)
        messages.append(ai_message)
        
        if not ai_message.tool_calls:
            return ai_message.content
        
        # Execute all tool calls
        for tool_call in ai_message.tool_calls:
            tool = tools_map[tool_call["name"]]
            result = tool.invoke(tool_call["args"])
            
            print(f"Tool: {tool_call['name']}")
            print(f"Args: {tool_call['args']}")
            print(f"Result: {result}\n")
            
            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            ))
    
    return "Max iterations reached"

# Test
answer = run_multi_tool_query(
    "What percentage of the total budget does Alice's department represent?"
)
print(f"Final Answer: {answer}")

# The LLM will:
# 1. Call get_user_data(1) to find Alice's department
# 2. Call get_department_budget("Engineering") to get budget
# 3. Maybe call get_department_budget for other departments to get total
# 4. Call calculate_percentage to compute the answer
```

---

## 🎯 Example 10: Tool with Error Handling

```python
from langchain_core.tools import tool
from typing import Optional

@tool
def divide_numbers(a: float, b: float) -> str:
    """
    Divide two numbers.
    
    Args:
        a: The numerator
        b: The denominator
    
    Returns:
        The result of a/b or an error message
    """
    try:
        if b == 0:
            return "Error: Cannot divide by zero"
        
        result = a / b
        return f"Result: {result}"
    
    except Exception as e:
        return f"Error: {str(e)}"

# Test error handling
print(divide_numbers.invoke({"a": 10, "b": 2}))   # Result: 5.0
print(divide_numbers.invoke({"a": 10, "b": 0}))   # Error: Cannot divide by zero
```

---

## 🔐 Example 11: Tool with API Integration

```python
from langchain_core.tools import tool
import requests
from typing import Optional

@tool
def get_github_user(username: str) -> str:
    """
    Get information about a GitHub user.
    
    Args:
        username: The GitHub username
    """
    try:
        response = requests.get(f"https://api.github.com/users/{username}")
        
        if response.status_code == 404:
            return f"User '{username}' not found"
        
        response.raise_for_status()
        data = response.json()
        
        return f"""
        Username: {data['login']}
        Name: {data.get('name', 'N/A')}
        Bio: {data.get('bio', 'N/A')}
        Public Repos: {data['public_repos']}
        Followers: {data['followers']}
        """
    
    except Exception as e:
        return f"Error fetching user data: {str(e)}"

# Test
result = get_github_user.invoke({"username": "langchain-ai"})
print(result)
```

---

## 🗄️ Example 12: Database Query Tool

```python
from langchain_core.tools import tool
import sqlite3
from typing import List

# Setup: Create sample database
def setup_database():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT,
            department TEXT,
            salary INTEGER
        )
    """)
    
    employees = [
        (1, "Alice", "Engineering", 120000),
        (2, "Bob", "Sales", 80000),
        (3, "Charlie", "Engineering", 110000),
        (4, "Diana", "Marketing", 90000)
    ]
    
    cursor.executemany("INSERT INTO employees VALUES (?, ?, ?, ?)", employees)
    conn.commit()
    return conn

# Create database
db_conn = setup_database()

@tool
def query_employees(department: str = None) -> str:
    """
    Query employee information.
    
    Args:
        department: Filter by department (optional)
    """
    cursor = db_conn.cursor()
    
    if department:
        cursor.execute(
            "SELECT name, department, salary FROM employees WHERE department = ?",
            (department,)
        )
    else:
        cursor.execute("SELECT name, department, salary FROM employees")
    
    results = cursor.fetchall()
    
    if not results:
        return f"No employees found" + (f" in {department}" if department else "")
    
    output = []
    for name, dept, salary in results:
        output.append(f"{name} ({dept}): ${salary:,}")
    
    return "\n".join(output)

@tool
def get_average_salary(department: str = None) -> str:
    """
    Get average salary, optionally filtered by department.
    
    Args:
        department: Department to filter by (optional)
    """
    cursor = db_conn.cursor()
    
    if department:
        cursor.execute(
            "SELECT AVG(salary) FROM employees WHERE department = ?",
            (department,)
        )
    else:
        cursor.execute("SELECT AVG(salary) FROM employees")
    
    avg = cursor.fetchone()[0]
    
    if avg is None:
        return "No data available"
    
    dept_str = f" in {department}" if department else ""
    return f"Average salary{dept_str}: ${avg:,.2f}"

# Test tools
print(query_employees.invoke({"department": "Engineering"}))
print("\n" + get_average_salary.invoke({"department": "Engineering"}))
```

---

## 📊 Example 13: Tool Calling with Streaming

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price for a symbol."""
    # Simulate stock data
    prices = {
        "AAPL": 175.43,
        "GOOGL": 142.56,
        "MSFT": 378.91
    }
    price = prices.get(symbol.upper(), 0)
    return f"${price}"

llm = ChatOpenAI(model="gpt-4", temperature=0, streaming=True)
llm_with_tools = llm.bind_tools([get_stock_price])

# Stream the response
print("Streaming response:")
for chunk in llm_with_tools.stream("What's the price of AAPL stock?"):
    # Each chunk can contain content or tool calls
    if chunk.content:
        print(chunk.content, end="", flush=True)
    if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
        print(f"\n[Tool call: {chunk.tool_calls}]")

print()
```

---

## 🔥 Best Practices

### **1. Clear Tool Descriptions**
```python
# ❌ Bad - vague description
@tool
def process_data(data: str) -> str:
    """Process data."""
    pass

# ✅ Good - specific description
@tool
def process_data(data: str) -> str:
    """
    Clean and normalize user input data by removing special characters 
    and converting to lowercase.
    """
    pass
```

### **2. Use Type Hints**
```python
# ✅ Always use type hints
@tool
def calculate_tax(amount: float, rate: float) -> float:
    """Calculate tax on an amount."""
    return amount * rate
```

### **3. Handle Errors Gracefully**
```python
# ✅ Return error messages, don't raise exceptions
@tool
def api_call(endpoint: str) -> str:
    """Call an API endpoint."""
    try:
        # API logic
        pass
    except Exception as e:
        return f"Error: {str(e)}"  # Return error as string
```

### **4. Keep Tools Focused**
```python
# ❌ Bad - does too much
@tool
def manage_user(action: str, user_id: int, **kwargs) -> str:
    """Create, update, delete, or query users."""
    pass

# ✅ Good - separate tools
@tool
def get_user(user_id: int) -> str:
    """Get user information."""
    pass

@tool
def update_user(user_id: int, name: str) -> str:
    """Update user name."""
    pass
```

### **5. Use return_direct for Simple Tools**
```python
@tool
def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%I:%M %p")

# If this doesn't need LLM processing, set return_direct=True
time_tool = StructuredTool.from_function(
    func=get_current_time,
    return_direct=True  # Skip LLM, return directly to user
)
```

---

## 📋 Tool Execution Flow Summary

```python
# Complete flow:
# 1. User asks question
# 2. LLM decides which tool(s) to call
# 3. Tool(s) executed
# 4. Results sent back to LLM
# 5. LLM formulates final answer

# This is manual in tools, but automated in Agents (next section!)
```

---

## 🎯 Practical Exercise

```python
"""
Exercise: Build a Personal Assistant Toolkit

Create a set of tools for a personal assistant:
1. Calendar tool - get/add events
2. Weather tool - get weather for location
3. Calculator tool - perform calculations
4. Email tool - draft/send emails (simulate)
5. Task tool - manage todo list

Then create a system that:
- Accepts natural language requests
- Calls appropriate tools
- Returns formatted responses

Test with queries like:
- "What's the weather in NYC?"
- "Add meeting tomorrow at 2pm"
- "Calculate 15% tip on $45.50"
- "Draft email to John about project update"
"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from datetime import datetime, timedelta
from typing import Optional

# TODO: Implement calendar tool
@tool
def calendar_get_events(date: str) -> str:
    """Get calendar events for a specific date."""
    pass

@tool
def calendar_add_event(title: str, date: str, time: str) -> str:
    """Add an event to calendar."""
    pass

# TODO: Implement weather tool
@tool
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    pass

# TODO: Implement calculator tool
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    pass

# TODO: Implement email tool
@tool
def draft_email(recipient: str, subject: str, body: str) -> str:
    """Draft an email."""
    pass

# TODO: Implement task tool
@tool
def add_task(task: str, priority: str = "medium") -> str:
    """Add a task to todo list."""
    pass

@tool
def get_tasks() -> str:
    """Get all pending tasks."""
    pass

# TODO: Create assistant that uses these tools
# TODO: Test with various queries
```

---

## ✅ Key Takeaways

1. **Tools extend LLM capabilities** - interact with external systems
2. **@tool decorator** - simplest way to create tools
3. **Clear descriptions are critical** - LLM uses them to decide when to call
4. **Type hints define arguments** - ensure proper inputs
5. **StructuredTool for complex schemas** - use Pydantic models
6. **Tools return strings** - LLM processes the output
7. **Error handling is important** - return errors as strings
8. **bind_tools attaches tools to LLM** - enables function calling
9. **Tool execution requires manual loop** - or use Agents (next section!)
10. **Multiple tools can work together** - LLM orchestrates them

---

## 📝 Understanding Check

1. What's the difference between a tool and a regular function?
2. Why are tool descriptions so important?
3. How does bind_tools work?
4. What's the manual process for executing tool calls?

**Ready for Section 10 on Agents?** Agents automate the tool calling loop and can reason about which tools to use! This is where LangChain becomes truly powerful! 🤖

Or would you like to:
- See the exercise solution?
- Practice more with tools?
- Ask questions about function calling?