# Section 1: Models - The Heart of LangChain 🧠

Models are the core components that generate text, embeddings, or perform reasoning. LangChain supports multiple model types and providers.

---

## 📋 Types of Models in LangChain

### **1. LLMs (Language Models)**
- Input: String
- Output: String
- Use case: Simple text completion (legacy, less common now)

### **2. Chat Models**
- Input: List of messages
- Output: Message
- Use case: **Most common** - conversational AI, structured interactions
- Examples: GPT-4, Claude, Gemini, Llama

### **3. Text Embedding Models**
- Input: Text
- Output: Vector of numbers (embeddings)
- Use case: Semantic search, RAG, similarity comparison

**We'll focus primarily on Chat Models as they're the modern standard.**

---

## 🔌 Model Providers

LangChain integrates with 50+ providers. Here are the most important:

| Provider | Models | Package |
|----------|--------|---------|
| OpenAI | GPT-4, GPT-3.5 | `langchain-openai` |
| Anthropic | Claude 3.5 Sonnet, Opus | `langchain-anthropic` |
| Google | Gemini Pro, Ultra | `langchain-google-genai` |
| HuggingFace | 1000s of open models | `langchain-huggingface` |
| Ollama | Local models (Llama, Mistral) | `langchain-ollama` |
| Azure OpenAI | GPT-4, GPT-3.5 | `langchain-openai` |
| Cohere | Command, Command-R | `langchain-cohere` |

---

## 💻 Practical Examples: Loading Different Models

### **Example 1: OpenAI Chat Model**

```python
from langchain_openai import ChatOpenAI
import os

# Set your API key (best practice: use environment variables)
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Initialize the model
llm = ChatOpenAI(
    model="gpt-4",  # or "gpt-3.5-turbo" for faster/cheaper
    temperature=0.7,  # 0 = deterministic, 1 = creative
    max_tokens=500,   # Maximum length of response
)

# Simple invocation
response = llm.invoke("Explain quantum computing in one sentence.")
print(response.content)
```

**Output:**
```
Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously through superposition, enabling exponentially faster processing of certain complex problems compared to classical computers.
```

---

### **Example 2: Anthropic Claude**

```python
from langchain_anthropic import ChatAnthropic
import os

os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0.5,
    max_tokens=1000,
)

response = llm.invoke("What are the key differences between Python and JavaScript?")
print(response.content)
```

---

### **Example 3: Google Gemini**

```python
from langchain_google_genai import ChatGoogleGenerativeAI
import os

os.environ["GOOGLE_API_KEY"] = "your-api-key-here"

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.3,
    convert_system_message_to_human=True,  # Gemini-specific parameter
)

response = llm.invoke("List 3 benefits of using vector databases.")
print(response.content)
```

---

### **Example 4: HuggingFace Models (Open Source)**

```python
from langchain_huggingface import HuggingFaceEndpoint
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your-token-here"

# Using hosted inference API
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.7,
    max_new_tokens=512,
)

response = llm.invoke("Write a Python function to calculate fibonacci numbers.")
print(response)
```

**Alternative: Local HuggingFace models**

```python
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load model locally
model_id = "gpt2"  # or any model from HuggingFace
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
)

# Wrap in LangChain
llm = HuggingFacePipeline(pipeline=pipe)

response = llm.invoke("The future of AI is")
print(response)
```

---

### **Example 5: Ollama (Local Models)**

```python
from langchain_ollama import ChatOllama

# First, install Ollama and pull a model:
# curl -fsSL https://ollama.com/install.sh | sh
# ollama pull llama2

llm = ChatOllama(
    model="llama2",  # or llama3, mistral, codellama, etc.
    temperature=0.8,
)

response = llm.invoke("Explain machine learning to a 10-year-old.")
print(response.content)
```

**Benefits of Ollama:**
- ✅ Completely free
- ✅ Runs locally (no API costs, full privacy)
- ✅ Great for development and experimentation

---

## 🎛️ Important Model Parameters

### **Common Parameters Across Providers**

```python
llm = ChatOpenAI(
    model="gpt-4",
    
    # Creativity control
    temperature=0.7,  # 0-1, higher = more random
    
    # Length control
    max_tokens=500,  # Maximum response length
    
    # Determinism
    seed=42,  # Some providers support reproducible outputs
    
    # Advanced sampling
    top_p=0.9,  # Nucleus sampling (alternative to temperature)
    frequency_penalty=0.0,  # Penalize repetition (-2.0 to 2.0)
    presence_penalty=0.0,   # Encourage new topics (-2.0 to 2.0)
    
    # Streaming
    streaming=True,  # Enable token-by-token streaming
)
```

### **Temperature Guide**

```python
# Temperature = 0 → Deterministic, factual
llm_factual = ChatOpenAI(temperature=0)
# Good for: Math, coding, factual Q&A

# Temperature = 0.7 → Balanced
llm_balanced = ChatOpenAI(temperature=0.7)
# Good for: General conversation, explanations

# Temperature = 1.0+ → Creative
llm_creative = ChatOpenAI(temperature=1.2)
# Good for: Creative writing, brainstorming
```

---

## 📊 Comparing Model Outputs

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

# Initialize multiple models
models = {
    "GPT-4": ChatOpenAI(model="gpt-4", temperature=0.7),
    "Claude": ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.7),
    "Llama2": ChatOllama(model="llama2", temperature=0.7),
}

prompt = "Explain the concept of gradient descent in 2 sentences."

# Compare responses
for name, model in models.items():
    response = model.invoke(prompt)
    print(f"\n{name}:")
    print(response.content)
```

---

## 🔍 Understanding Model Response Objects

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
response = llm.invoke("Hello!")

# Response is an AIMessage object
print(type(response))  # <class 'langchain_core.messages.ai.AIMessage'>

# Access content
print(response.content)  # The actual text response

# Access metadata
print(response.response_metadata)  # Token usage, model info, etc.

# Example output:
# {
#     'token_usage': {'completion_tokens': 10, 'prompt_tokens': 8, 'total_tokens': 18},
#     'model_name': 'gpt-4',
#     'finish_reason': 'stop'
# }
```

---

## 🚀 Async Support (Important for Production)

```python
import asyncio
from langchain_openai import ChatOpenAI

async def async_example():
    llm = ChatOpenAI(model="gpt-4")
    
    # Single async call
    response = await llm.ainvoke("What is async programming?")
    print(response.content)
    
    # Batch async calls (parallel execution)
    prompts = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?"
    ]
    
    responses = await llm.abatch(prompts)
    for i, resp in enumerate(responses):
        print(f"\n{prompts[i]}")
        print(resp.content)

# Run async function
asyncio.run(async_example())
```

---

## 🌊 Streaming Responses

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", streaming=True)

# Stream tokens as they're generated
for chunk in llm.stream("Write a short poem about Python programming."):
    print(chunk.content, end="", flush=True)

print()  # New line after streaming completes
```

**Output (tokens arrive one by one):**
```
In lines of code, both clean and bright,
Python shines with elegant might...
```

---

## 🎯 Practical Exercise

Now let's practice! Try this:

```python
"""
Exercise: Multi-Model Comparison Tool

Task: Create a function that takes a question and gets responses 
from at least 2 different models, then compares them.

Requirements:
1. Use at least 2 different model providers
2. Print both responses
3. Calculate and display token usage for each
"""

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

def compare_models(question: str):
    # TODO: Initialize models
    # TODO: Get responses
    # TODO: Display results with token counts
    pass

# Test it
compare_models("What is the difference between supervised and unsupervised learning?")
```

**Try implementing this yourself!** I'll provide the solution when you're ready.

---

## ✅ Key Takeaways

1. **Chat Models are the standard** - use them over plain LLMs
2. **LangChain provides unified interfaces** across 50+ providers
3. **Temperature controls creativity** (0 = deterministic, 1+ = creative)
4. **Responses are AIMessage objects** with `.content` and metadata
5. **Async and streaming are supported** natively
6. **Ollama enables free local development** without API costs

---

## 📝 Before We Move On

**Understanding Check:**

1. Can you explain the difference between an LLM and a Chat Model?
2. Which model would you use for a cost-sensitive production application?
3. When would you use temperature=0 vs temperature=1?

**Ready to proceed to the next section on Prompts and Prompt Templates?** Or would you like to:
- Practice more with models?
- Ask questions about anything?
- See the exercise solution?