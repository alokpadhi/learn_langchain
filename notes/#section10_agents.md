# Section 10: Agents 🤖 (LangChain 1.0+ Pure)
---

## 🎯 What are Agents?

**Agent** = LLM + Tools + Reasoning Loop

In LangChain 1.0+, agents are built explicitly by:
1. **Binding tools to LLM** using `.bind_tools()`
2. **Creating a tool calling loop** manually
3. **Managing state** through message history
4. **Handling tool execution** explicitly

**The Modern Philosophy**: Explicit is better than implicit. You control every step.

---

## 🔄 Modern Agent Architecture

```
User Input (HumanMessage)
    ↓
LLM with bound tools (llm.bind_tools())
    ↓
Response with tool_calls?
    ↓
Yes: Execute tools → Create ToolMessage → Loop back to LLM
    ↓
No: Return final answer
```

---

## 💻 Example 1: Basic Tool Calling (Foundation) ⭐

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Define tools
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

# Create LLM and bind tools
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [get_word_length, multiply]
llm_with_tools = llm.bind_tools(tools)

# Single turn tool calling
query = "What is the length of 'intelligence'?"
messages = [HumanMessage(content=query)]

# Get LLM response
response = llm_with_tools.invoke(messages)

print("Response:", response)
print("\nTool calls:", response.tool_calls)

# Execute tool if called
if response.tool_calls:
    for tool_call in response.tool_calls:
        # Find the tool
        tool_map = {t.name: t for t in tools}
        selected_tool = tool_map[tool_call["name"]]
        
        # Execute
        tool_output = selected_tool.invoke(tool_call["args"])
        print(f"\nTool: {tool_call['name']}")
        print(f"Input: {tool_call['args']}")
        print(f"Output: {tool_output}")
```

---

## 🔄 Example 2: Multi-Turn Agent Loop (Core Pattern) ⭐⭐⭐

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# Define tools
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Examples: '2+2', '10*5', '100/4'"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())

# Setup
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [calculator, get_word_count]
llm_with_tools = llm.bind_tools(tools)

# Create tool map for execution
tool_map = {t.name: t for t in tools}

def run_agent(query: str, max_iterations: int = 10):
    """Run agent loop manually."""
    messages = [HumanMessage(content=query)]
    
    for iteration in range(max_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}")
        print(f"{'='*60}")
        
        # Get LLM response
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        print(f"LLM Response: {response.content}")
        
        # Check if done
        if not response.tool_calls:
            print("\n✅ Agent finished!")
            return response.content
        
        # Execute tool calls
        print(f"\n🔧 Executing {len(response.tool_calls)} tool(s)...")
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"\n  Tool: {tool_name}")
            print(f"  Args: {tool_args}")
            
            # Execute
            selected_tool = tool_map[tool_name]
            tool_output = selected_tool.invoke(tool_args)
            
            print(f"  Output: {tool_output}")
            
            # Add tool result to messages
            messages.append(ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_call["id"]
            ))
    
    return "Max iterations reached"

# Test
result = run_agent(
    "How many words are in 'Hello World' and what is 15 times 3?"
)

print(f"\n{'='*60}")
print(f"Final Answer: {result}")
```

---

## 🎨 Example 3: Agent with Memory (Conversation State) ⭐

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Tools
@tool
def add_to_cart(product: str, quantity: int = 1) -> str:
    """Add a product to shopping cart."""
    return f"Added {quantity}x {product} to cart"

@tool
def get_price(product: str) -> str:
    """Get product price."""
    prices = {"laptop": "$999", "mouse": "$29", "keyboard": "$79"}
    return prices.get(product.lower(), "Not found")

@tool  
def calculate_total(items: str) -> str:
    """Calculate total price. Items should be comma-separated product names."""
    prices = {"laptop": 999, "mouse": 29, "keyboard": 79}
    total = sum(prices.get(item.strip().lower(), 0) for item in items.split(","))
    return f"${total}"

# Setup
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [add_to_cart, get_price, calculate_total]
llm_with_tools = llm.bind_tools(tools)
tool_map = {t.name: t for t in tools}

class ConversationalAgent:
    """Agent that maintains conversation history."""
    
    def __init__(self, llm_with_tools, tools):
        self.llm_with_tools = llm_with_tools
        self.tool_map = {t.name: t for t in tools}
        self.messages = []  # Conversation memory
    
    def run(self, user_input: str, max_iterations: int = 5):
        """Run agent with conversation context."""
        # Add user message
        self.messages.append(HumanMessage(content=user_input))
        
        for _ in range(max_iterations):
            # Get response with full history
            response = self.llm_with_tools.invoke(self.messages)
            self.messages.append(response)
            
            # If no tool calls, we're done
            if not response.tool_calls:
                return response.content
            
            # Execute tools
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                selected_tool = self.tool_map[tool_name]
                tool_output = selected_tool.invoke(tool_args)
                
                self.messages.append(ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_call["id"]
                ))
        
        return "Max iterations reached"
    
    def get_history(self):
        """Get conversation history."""
        return self.messages

# Create agent
agent = ConversationalAgent(llm_with_tools, tools)

# Multi-turn conversation
print("Turn 1:")
response1 = agent.run("Add 2 laptops and 1 mouse to my cart")
print(f"Response: {response1}\n")

print("Turn 2:")
response2 = agent.run("What's the total cost?")  # Agent remembers context!
print(f"Response: {response2}\n")

print("Turn 3:")
response3 = agent.run("Actually, make it 3 laptops")
print(f"Response: {response3}")
```

---

## 🔍 Example 4: Agent with System Prompt ⭐

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# Tools
@tool
def search_products(category: str) -> str:
    """Search products by category."""
    products = {
        "electronics": "Laptop, Phone, Tablet",
        "accessories": "Mouse, Keyboard, Headphones",
        "furniture": "Desk, Chair, Lamp"
    }
    return products.get(category.lower(), "Category not found")

@tool
def check_stock(product: str) -> str:
    """Check if product is in stock."""
    inventory = {
        "laptop": "In stock (5 units)",
        "phone": "Low stock (2 units)",
        "tablet": "Out of stock"
    }
    return inventory.get(product.lower(), "Product not found")

# Setup with system prompt
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [search_products, check_stock]
llm_with_tools = llm.bind_tools(tools)
tool_map = {t.name: t for t in tools}

# System prompt to guide behavior
SYSTEM_PROMPT = """You are a helpful shopping assistant. Follow these rules:
1. Always check stock availability before recommending products
2. If a product is out of stock, suggest alternatives
3. Be concise and friendly
4. Use tools to get accurate information"""

def run_agent_with_system_prompt(query: str):
    """Run agent with system prompt."""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query)
    ]
    
    for iteration in range(5):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        if not response.tool_calls:
            return response.content
        
        # Execute tools
        for tool_call in response.tool_calls:
            selected_tool = tool_map[tool_call["name"]]
            tool_output = selected_tool.invoke(tool_call["args"])
            
            messages.append(ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_call["id"]
            ))
    
    return "Max iterations reached"

# Test
result = run_agent_with_system_prompt(
    "I'm looking for electronics. What's available and in stock?"
)
print(f"Response: {result}")
```

---

## 📊 Example 5: Parallel Tool Execution ⭐

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# Tools that can run in parallel
@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    weather = {"NYC": "72°F, Sunny", "LA": "85°F, Clear", "Chicago": "65°F, Cloudy"}
    return weather.get(city, "Unknown")

@tool
def get_population(city: str) -> str:
    """Get population of a city."""
    pop = {"NYC": "8.3M", "LA": "3.9M", "Chicago": "2.7M"}
    return pop.get(city, "Unknown")

@tool
def get_timezone(city: str) -> str:
    """Get timezone of a city."""
    tz = {"NYC": "EST", "LA": "PST", "Chicago": "CST"}
    return tz.get(city, "Unknown")

# Setup
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [get_weather, get_population, get_timezone]
llm_with_tools = llm.bind_tools(tools)
tool_map = {t.name: t for t in tools}

def run_agent_parallel(query: str):
    """Agent that can execute multiple tools in parallel."""
    messages = [HumanMessage(content=query)]
    
    for iteration in range(5):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        if not response.tool_calls:
            return response.content
        
        # Execute ALL tool calls (they can run in parallel)
        print(f"\n🔧 Executing {len(response.tool_calls)} tools in parallel...")
        
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"  - {tool_name}({tool_args})")
            
            selected_tool = tool_map[tool_name]
            tool_output = selected_tool.invoke(tool_args)
            
            messages.append(ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_call["id"]
            ))
    
    return "Max iterations reached"

# Test - LLM will call multiple tools at once
result = run_agent_parallel(
    "Tell me about NYC: weather, population, and timezone"
)

print(f"\nFinal Answer: {result}")
```

---

## 🎯 Example 6: Streaming Agent Responses ⭐

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# Tools
@tool
def search_docs(query: str) -> str:
    """Search documentation."""
    docs = {
        "python": "Python is a high-level programming language.",
        "langchain": "LangChain is a framework for LLM applications.",
        "agents": "Agents combine LLMs with tools for autonomous action."
    }
    return docs.get(query.lower(), "Not found")

# Setup
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [search_docs]
llm_with_tools = llm.bind_tools(tools)
tool_map = {t.name: t for t in tools}

def stream_agent(query: str):
    """Stream agent responses as they're generated."""
    messages = [HumanMessage(content=query)]
    
    for iteration in range(5):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}")
        print(f"{'='*60}\n")
        
        # Stream the response
        print("LLM Response: ", end="", flush=True)
        
        full_response = None
        for chunk in llm_with_tools.stream(messages):
            # Print content as it arrives
            if chunk.content:
                print(chunk.content, end="", flush=True)
            
            # Store the complete response
            if not full_response:
                full_response = chunk
            else:
                full_response = full_response + chunk
        
        print()  # Newline after streaming
        messages.append(full_response)
        
        # Check if done
        if not full_response.tool_calls:
            return full_response.content
        
        # Execute tools
        print(f"\n🔧 Executing tools...")
        for tool_call in full_response.tool_calls:
            print(f"  Tool: {tool_call['name']}({tool_call['args']})")
            
            selected_tool = tool_map[tool_call["name"]]
            tool_output = selected_tool.invoke(tool_call["args"])
            
            print(f"  Output: {tool_output}")
            
            messages.append(ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_call["id"]
            ))
    
    return "Max iterations reached"

# Test
result = stream_agent("What is Python? Then tell me about agents.")
```

---

## 🗃️ Example 7: Agent with RAG (Vector Store Tool) ⭐

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, ToolMessage

# Create vector store
documents = [
    Document(page_content="Python was created by Guido van Rossum in 1991."),
    Document(page_content="Python is widely used for data science and ML."),
    Document(page_content="Popular frameworks: Django, Flask, FastAPI."),
    Document(page_content="Python uses indentation for code blocks."),
]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# Create RAG tool
@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information about Python.
    Use this for factual questions about Python."""
    docs = vectorstore.similarity_search(query, k=2)
    return "\n".join([doc.page_content for doc in docs])

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression.
    Examples: '2+2', '10*5', '100/4'"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Setup
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [search_knowledge_base, calculate]
llm_with_tools = llm.bind_tools(tools)
tool_map = {t.name: t for t in tools}

def run_rag_agent(query: str):
    """Agent with RAG capability."""
    messages = [HumanMessage(content=query)]
    
    for _ in range(5):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        if not response.tool_calls:
            return response.content
        
        for tool_call in response.tool_calls:
            selected_tool = tool_map[tool_call["name"]]
            tool_output = selected_tool.invoke(tool_call["args"])
            
            messages.append(ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_call["id"]
            ))
    
    return "Max iterations reached"

# Test
questions = [
    "Who created Python?",
    "What is Python used for?",
    "What's 25 * 37?",
    "Tell me about Python frameworks and calculate 100 / 4"
]

for q in questions:
    print(f"\nQ: {q}")
    result = run_rag_agent(q)
    print(f"A: {result}")
    print("-" * 60)
```

---

## 🔧 Example 8: Structured Output with Tools ⭐

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field

# Define structured input for tool
class AnalysisInput(BaseModel):
    """Input for data analysis tool."""
    data: str = Field(description="Comma-separated numbers")
    metric: str = Field(description="Metric to calculate: mean, median, or sum")
    filter_threshold: float = Field(default=0, description="Filter values above this")

@tool(args_schema=AnalysisInput)
def analyze_data(data: str, metric: str, filter_threshold: float = 0) -> str:
    """Analyze numerical data with specified metric and optional filtering."""
    try:
        # Parse data
        numbers = [float(x.strip()) for x in data.split(",")]
        
        # Filter
        if filter_threshold > 0:
            numbers = [n for n in numbers if n > filter_threshold]
            
        if not numbers:
            return "No data after filtering"
        
        # Calculate
        if metric == "mean":
            result = sum(numbers) / len(numbers)
        elif metric == "median":
            sorted_nums = sorted(numbers)
            n = len(sorted_nums)
            result = sorted_nums[n//2] if n % 2 else (sorted_nums[n//2-1] + sorted_nums[n//2]) / 2
        elif metric == "sum":
            result = sum(numbers)
        else:
            return f"Unknown metric: {metric}"
        
        return f"{metric.title()}: {result:.2f} (from {len(numbers)} values)"
    
    except Exception as e:
        return f"Error: {str(e)}"

# Setup
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [analyze_data]
llm_with_tools = llm.bind_tools(tools)
tool_map = {t.name: t for t in tools}

def run_structured_agent(query: str):
    """Agent with structured tool inputs."""
    messages = [HumanMessage(content=query)]
    
    for iteration in range(3):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        print(f"\nIteration {iteration + 1}:")
        print(f"Response: {response.content}")
        
        if not response.tool_calls:
            return response.content
        
        for tool_call in response.tool_calls:
            print(f"\nTool Call:")
            print(f"  Name: {tool_call['name']}")
            print(f"  Args: {tool_call['args']}")
            
            selected_tool = tool_map[tool_call["name"]]
            tool_output = selected_tool.invoke(tool_call["args"])
            
            print(f"  Output: {tool_output}")
            
            messages.append(ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_call["id"]
            ))
    
    return "Max iterations reached"

# Test
result = run_structured_agent(
    "Calculate the mean of these numbers: 10, 20, 5, 30, 15, 8, but only values above 10"
)
print(f"\n{'='*60}")
print(f"Final: {result}")
```

---

## 🎨 Example 9: Error Handling and Retries ⭐

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# Tools with potential failures
@tool
def fetch_data(source: str) -> str:
    """Fetch data from a source. Valid sources: api, database, cache."""
    valid_sources = ["api", "database", "cache"]
    if source.lower() not in valid_sources:
        return f"Error: Invalid source '{source}'. Valid: {', '.join(valid_sources)}"
    return f"Data from {source}: [1, 2, 3, 4, 5]"

@tool
def process_data(data: str) -> str:
    """Process data. Data must start with 'Data from'."""
    if not data.startswith("Data from"):
        return "Error: Invalid data format. Please fetch data first."
    return f"Processed: {data}"

# Setup
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [fetch_data, process_data]
llm_with_tools = llm.bind_tools(tools)
tool_map = {t.name: t for t in tools}

def run_agent_with_error_handling(query: str):
    """Agent that handles errors gracefully."""
    messages = [HumanMessage(content=query)]
    
    for iteration in range(10):  # More iterations for retries
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}")
        
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        print(f"LLM: {response.content}")
        
        if not response.tool_calls:
            return response.content
        
        # Execute tools and handle errors
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"\n🔧 Tool: {tool_name}")
            print(f"   Args: {tool_args}")
            
            try:
                selected_tool = tool_map[tool_name]
                tool_output = selected_tool.invoke(tool_args)
                
                # Check if output indicates error
                if "Error:" in str(tool_output):
                    print(f"   ⚠️  {tool_output}")
                else:
                    print(f"   ✅ {tool_output}")
                
                messages.append(ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_call["id"]
                ))
                
            except Exception as e:
                error_msg = f"Error executing {tool_name}: {str(e)}"
                print(f"   ❌ {error_msg}")
                
                messages.append(ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_call["id"]
                ))
    
    return "Max iterations reached"

# Test with invalid input (agent should recover)
result = run_agent_with_error_handling(
    "Fetch data from 'invalid_source' and process it"
)

print(f"\n{'='*60}")
print(f"Final: {result}")
```

---

## 🔥 Example 10: Multi-Step Complex Agent ⭐

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# Multi-step planning tools
@tool
def get_user_profile(user_id: int) -> str:
    """Get user profile information."""
    profiles = {
        1: "User: Alice, Preferences: Italian food, Budget: $$, Vegetarian: Yes",
        2: "User: Bob, Preferences: Japanese food, Budget: $$$, Vegetarian: No"
    }
    return profiles.get(user_id, "User not found")

@tool
def search_restaurants(cuisine: str, budget: str) -> str:
    """Search restaurants by cuisine and budget level ($, $$, $$$)."""
    restaurants = {
        ("italian", "$$"): "Pasta Paradise, Luigi's Kitchen",
        ("japanese", "$$$"): "Sushi Master, Tokyo Garden",
        ("italian", "$$$"): "Il Ristorante"
    }
    key = (cuisine.lower(), budget)
    return restaurants.get(key, "No restaurants found")

@tool
def filter_dietary(restaurants: str, restriction: str) -> str:
    """Filter restaurants by dietary restriction."""
    if "vegetarian" in restriction.lower():
        # Simplified: assume first restaurant is veggie-friendly
        first = restaurants.split(",")[0].strip()
        return f"{first} (Vegetarian-friendly)"
    return restaurants

@tool
def get_reviews(restaurant: str) -> str:
    """Get reviews for a specific restaurant."""
    reviews = {
        "pasta paradise": "⭐ 4.5/5 - Excellent vegetarian options",
        "luigi's kitchen": "⭐ 4.2/5 - Authentic Italian",
        "sushi master": "⭐ 4.8/5 - Best sushi in town"
    }
    return reviews.get(restaurant.lower(), "No reviews available")

# Setup
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [get_user_profile, search_restaurants, filter_dietary, get_reviews]
llm_with_tools = llm.bind_tools(tools)
tool_map = {t.name: t for t in tools}

SYSTEM_PROMPT = """You are a restaurant recommendation assistant.

To recommend a restaurant, follow these steps:
1. Get user profile to understand preferences
2. Search restaurants matching cuisine and budget
3. Filter by dietary restrictions if needed
4. Get reviews for the top options
5. Make a final recommendation

Be thorough and use all necessary tools."""

def run_complex_agent(query: str):
    """Multi-step agent with planning."""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query)
    ]
    
    for iteration in range(10):
        print(f"\n{'='*60}")
        print(f"Step {iteration + 1}")
        print(f"{'='*60}")
        
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        if response.content:
            print(f"Thinking: {response.content}")
        
        if not response.tool_calls:
            print(f"\n✅ Final Recommendation: {response.content}")
            return response.content
        
        # Execute tools
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"\n🔧 Using tool: {tool_name}")
            print(f"   Input: {tool_args}")
            
            selected_tool = tool_map[tool_name]
            tool_output = selected_tool.invoke(tool_args)
            
            print(f"   Result: {tool_output}")
            
            messages.append(ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_call["id"]
            ))
    
    return "Max iterations reached"

# Test
result = run_complex_agent("Recommend a restaurant for user ID 1")
```

---

## 🎯 Example 11: Forced Tool Calling ⭐

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# Tools
@tool
def get_current_weather(location: str) -> str:
    """Get the current weather for a location."""
    weather_data = {
        "new york": "72°F and sunny",
        "london": "55°F and rainy",
        "tokyo": "68°F and cloudy"
    }
    return weather_data.get(location.lower(), "Weather data not available")

@tool
def get_forecast(location: str, days: int = 3) -> str:
    """Get weather forecast for a location."""
    return f"3-day forecast for {location}: Mostly sunny, temps 70-75°F"

# Setup
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [get_current_weather, get_forecast]

# Force the model to ALWAYS call a specific tool
llm_force_weather = llm.bind_tools(
    tools,
    tool_choice="get_current_weather"  # Force this specific tool
)

# Or force ANY tool (not no tool)
llm_force_any = llm.bind_tools(
    tools,
    tool_choice="any"  # Must use a tool, but can choose which one
)

tool_map = {t.name: t for t in tools}

def run_with_forced_tool(query: str):
    """Demonstrate forced tool calling."""
    messages = [HumanMessage(content=query)]
    
    print("Using forced tool calling...")
    response = llm_force_weather.invoke(messages)
    
    print(f"Tool called: {response.tool_calls[0]['name']}")
    print(f"Arguments: {response.tool_calls[0]['args']}")
    
    # Execute the forced tool call
    tool_call = response.tool_calls[0]
    selected_tool = tool_map[tool_call["name"]]
    result = selected_tool.invoke(tool_call["args"])
    
    return result

# Test
result = run_with_forced_tool("Tell me about New York")  # Will force weather check
print(f"\nResult: {result}")
```

---

## 🔥 Best Practices (LangChain 1.0+)

### **1. Always Use .bind_tools() for Tool Calling**
```python
# ✅ Modern way
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

# ❌ Don't do this (legacy)
# from langchain.agents import AgentExecutor
```

### **2. Build Explicit Tool Calling Loops**
```python
# ✅ Explicit and controllable
def run_agent(query):
    messages = [HumanMessage(content=query)]
    
    for iteration in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        if not response.tool_calls:
            return response.content
        
        for tool_call in response.tool_calls:
            tool_output = execute_tool(tool_call)
            messages.append(ToolMessage(...))
    
    return "Done"
```

### **3. Maintain Conversation State with Messages**
```python
# ✅ Message history is your memory
class Agent:
    def __init__(self):
        self.messages = []  # Persistent conversation state
    
    def run(self, user_input):
        self.messages.append(HumanMessage(content=user_input))
        # ... agent loop
```

### **4. Handle Tool Errors Gracefully**
```python
# ✅ Return error messages from tools
@tool
def risky_operation(input: str) -> str:
    """Perform operation that might fail."""
    try:
        result = do_something(input)
        return f"Success: {result}"
    except Exception as e:
        return f"Error: {str(e)}. Try a different approach."
```

### **5. Use SystemMessage for Behavior Control**
```python
# ✅ Guide agent with system prompts
messages = [
    SystemMessage(content="You are a helpful assistant. Always..."),
    HumanMessage(content=user_query)
]
```

### **6. Set Reasonable Iteration Limits**
```python
# ✅ Prevent infinite loops
MAX_ITERATIONS = 10

for iteration in range(MAX_ITERATIONS):
    # agent loop
    pass
```

### **7. Stream for Better UX**
```python
# ✅ Stream responses as they're generated
for chunk in llm_with_tools.stream(messages):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

### **8. Use Descriptive Tool Docstrings**
```python
# ✅ Clear tool descriptions help the LLM
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Examples:
    - '2 + 2' returns '4'
    - '10 * 5' returns '50'
    - '100 / 4' returns '25.0'
    
    Only supports basic arithmetic operators: +, -, *, /
    """
    pass
```

### **9. Separate Tool Map Creation**
```python
# ✅ Easy tool lookup
tools = [tool1, tool2, tool3]
tool_map = {t.name: t for t in tools}

# Then use in loop
selected_tool = tool_map[tool_name]
```

### **10. Use Pydantic for Structured Inputs**
```python
# ✅ Type-safe tool inputs
class SearchInput(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, description="Max results")

@tool(args_schema=SearchInput)
def search(query: str, max_results: int = 5) -> str:
    """Search with structured input."""
    pass
```

---

## 📊 LangChain 1.0+ Philosophy

| Aspect | LangChain 1.0+ Approach |
|--------|------------------------|
| **Control** | Explicit loops, full visibility |
| **Flexibility** | Build your own patterns |
| **Debugging** | See every step, easy to trace |
| **State** | Message history (simple list) |
| **Tools** | bind_tools() + manual execution |
| **Memory** | Message list persistence |
| **Streaming** | Built into LLM streaming |
| **Best For** | Production apps, custom workflows |

---

## 🎯 Practical Exercise

```python
"""
Exercise: Build a Research Assistant (Pure LangChain 1.0+)

Create an agent that can:
1. Search for information (mock)
2. Save findings to knowledge base (in-memory list)
3. Query knowledge base
4. Generate summary report

Requirements:
- Use .bind_tools() pattern
- Build explicit agent loop
- Maintain state in messages
- No LangGraph, no AgentExecutor
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# In-memory knowledge base
knowledge_base = []

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    # Mock implementation
    results = {
        "ai trends": "AI trends 2024: Multimodal AI, AI agents, Smaller models",
        "machine learning": "ML advances: Better efficiency, Edge deployment, AutoML"
    }
    for key in results:
        if key in query.lower():
            return results[key]
    return "No results found"

# TODO: Implement save_to_kb tool
@tool
def save_to_kb(content: str, topic: str) -> str:
    """Save information to knowledge base."""
    # TODO: Add to knowledge_base list with structure
    pass

# TODO: Implement query_kb tool  
@tool
def query_kb(topic: str) -> str:
    """Query knowledge base for information on a topic."""
    # TODO: Search knowledge_base list
    pass

# TODO: Implement summarize tool
@tool
def summarize(content: str) -> str:
    """Create a concise summary of content."""
    # TODO: Implement (can be simple for now)
    pass

# TODO: Create agent with explicit loop
# TODO: Test with: "Research AI trends for 2024 and create a summary"

# Your implementation here...
```

---

## ✅ Key Takeaways

1. **Use .bind_tools()** - Modern way to attach tools to LLM
2. **Build explicit loops** - Full control over agent behavior
3. **Messages are state** - Conversation history = memory
4. **No black boxes** - You see and control every step
5. **Tool execution is manual** - You invoke tools explicitly
6. **Streaming is native** - Use .stream() for real-time output
7. **SystemMessage for guidance** - Shape agent behavior
8. **Pydantic for structure** - Type-safe tool inputs
9. **Error handling in tools** - Return error messages, don't crash
10. **Iterations are explicit** - Set your own limits

**The Modern Way**: Explicit, flexible, transparent. You build the agent loop, you control the flow.

---

## 📝 Understanding Check

1. How do you attach tools to an LLM in LangChain 1.0+?
2. What's stored in the messages list during an agent conversation?
3. How do you handle multi-turn conversations?
4. Why build explicit loops instead of using AgentExecutor?

Ready for Advanced RAG? 🚀
