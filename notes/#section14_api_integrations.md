# Section 14: API Integration 🌐

Learn how to integrate LangChain with different LLM providers beyond OpenAI - Anthropic Claude, local models with Ollama, Azure OpenAI, Google Gemini, and more.

---

## 🎯 Why Multiple Providers?

**Benefits of Multi-Provider Support:**

```python
# 1. Cost Optimization
# - Different pricing models
# - Choose based on task complexity

# 2. Performance
# - Some models excel at specific tasks
# - Claude for long context, GPT-4 for reasoning

# 3. Redundancy
# - Fallback if one provider is down
# - Rate limit handling

# 4. Privacy
# - Local models (Ollama) for sensitive data
# - No data sent to external APIs

# 5. Specialized Capabilities
# - Gemini for multimodal
# - Claude for long documents
```

---

## 📋 Supported Providers

| Provider | Models | Key Features | Cost |
|----------|--------|--------------|------|
| **OpenAI** | GPT-4, GPT-3.5 | Excellent reasoning | $$$ |
| **Anthropic** | Claude 3.5 Sonnet, Opus | Long context (200k), safety | $$$ |
| **Google** | Gemini Pro, Ultra | Multimodal, free tier | $-$$ |
| **Azure OpenAI** | GPT-4, GPT-3.5 | Enterprise features | $$$ |
| **Ollama** | Llama 3, Mistral, etc. | Local, private, free | Free |
| **Cohere** | Command, Command-R | Multilingual | $$ |
| **HuggingFace** | 1000s of models | Open source, custom | Free-$$ |

---

## 💻 Example 1: Anthropic Claude Integration

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Set API key
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

# Initialize Claude
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",  # Latest model
    temperature=0.7,
    max_tokens=1024
)

# Simple usage
response = llm.invoke("Explain quantum entanglement in simple terms")
print(response.content)

# With chain
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question clearly and concisely.

Question: {question}

Answer:
""")

chain = prompt | llm | StrOutputParser()

result = chain.invoke({"question": "What makes Claude different from other LLMs?"})
print(result)

# Output:
# Claude is designed with a focus on safety and helpfulness...
```

**Claude-Specific Features:**
```python
# Long context (200k tokens)
llm_long = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    max_tokens=4096
)

# Process very long documents
long_document = "..." * 50000  # Very long text
response = llm_long.invoke(f"Summarize this document: {long_document}")
```

---

## 🏠 Example 2: Ollama (Local Models)

```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Prerequisites:
# 1. Install Ollama: https://ollama.com/download
# 2. Pull a model: ollama pull llama3

# Initialize Ollama
llm = ChatOllama(
    model="llama3",  # or llama3:70b, mistral, codellama, etc.
    temperature=0.7,
)

# Simple usage
response = llm.invoke("Write a Python function to calculate fibonacci numbers")
print(response.content)

# Available models in Ollama:
models = [
    "llama3",           # Meta's Llama 3 (8B, 70B)
    "llama3.1",         # Llama 3.1 with longer context
    "mistral",          # Mistral 7B
    "mixtral",          # Mixtral 8x7B (MoE)
    "codellama",        # Code-specialized
    "phi3",             # Microsoft Phi-3
    "gemma2",           # Google Gemma 2
]

# With chain
prompt = ChatPromptTemplate.from_template("""
You are a coding assistant. Help with this task:

Task: {task}

Code:
""")

chain = prompt | llm

result = chain.invoke({"task": "Create a REST API endpoint with FastAPI"})
print(result.content)

# Benefits of Ollama:
# ✅ Completely free
# ✅ Runs locally (privacy)
# ✅ No rate limits
# ✅ Works offline
# ✅ Multiple model choices
```

---

## 🔷 Example 3: Azure OpenAI Integration

```python
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os

# Azure OpenAI credentials
os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-key"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource.openai.azure.com/"

# Initialize Azure OpenAI
llm = AzureChatOpenAI(
    azure_deployment="gpt-4",  # Your deployment name
    api_version="2024-02-15-preview",
    temperature=0.7,
    max_tokens=1000
)

# Usage is identical to regular OpenAI
response = llm.invoke("Explain cloud computing")
print(response.content)

# With chain
prompt = ChatPromptTemplate.from_template("""
Analyze this business scenario:

Scenario: {scenario}

Analysis:
""")

chain = prompt | llm

result = chain.invoke({
    "scenario": "A startup wants to scale from 100 to 10,000 users"
})
print(result.content)

# Azure-specific features:
# - Enterprise security and compliance
# - Private endpoints
# - Data residency options
# - SLA guarantees
# - Integration with Azure services
```

---

## 🎨 Example 4: Google Gemini Integration

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
import os

# Set API key (get from https://makersuite.google.com/app/apikey)
os.environ["GOOGLE_API_KEY"] = "your-google-api-key"

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",  # or gemini-pro-vision for images
    temperature=0.7,
    convert_system_message_to_human=True  # Gemini-specific
)

# Simple usage
response = llm.invoke("Explain the difference between AI and ML")
print(response.content)

# With chain
prompt = ChatPromptTemplate.from_template("""
You are a creative writing assistant.

Topic: {topic}

Write a short story:
""")

chain = prompt | llm

result = chain.invoke({"topic": "a robot learning to paint"})
print(result.content)

# Multimodal with Gemini Vision
llm_vision = ChatGoogleGenerativeAI(
    model="gemini-pro-vision",
    temperature=0.4
)

# Note: Image handling requires special formatting
# See Google's documentation for image inputs
```

---

## 🔄 Example 5: Provider Fallback System

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import os

# Configure multiple providers
providers = {
    "openai": ChatOpenAI(
        model="gpt-4",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    ),
    "anthropic": ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0.7,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    ),
    "ollama": ChatOllama(
        model="llama3",
        temperature=0.7
    )
}

# Fallback logic
def invoke_with_fallback(prompt: str, preferred_order=None):
    """Try providers in order until one succeeds."""
    if preferred_order is None:
        preferred_order = ["openai", "anthropic", "ollama"]
    
    last_error = None
    
    for provider_name in preferred_order:
        try:
            llm = providers[provider_name]
            print(f"Trying {provider_name}...")
            response = llm.invoke(prompt)
            print(f"✅ Success with {provider_name}")
            return response.content
        
        except Exception as e:
            print(f"❌ {provider_name} failed: {str(e)}")
            last_error = e
            continue
    
    raise Exception(f"All providers failed. Last error: {last_error}")

# Test fallback
result = invoke_with_fallback(
    "What is the capital of France?",
    preferred_order=["openai", "anthropic", "ollama"]
)

print(f"\nResult: {result}")

# Use cases:
# - API rate limits
# - Service outages
# - Cost optimization (try cheap first)
# - Redundancy
```

---

## 💰 Example 6: Cost-Based Provider Selection

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Provider costs (approximate, per 1M tokens)
PROVIDER_COSTS = {
    "gpt-4": {"input": 30, "output": 60},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    "claude-3-5-sonnet": {"input": 3, "output": 15},
    "ollama": {"input": 0, "output": 0}  # Local, free
}

# Task complexity mapping
TASK_COMPLEXITY = {
    "simple": ["gpt-3.5-turbo", "ollama"],
    "medium": ["claude-3-5-sonnet", "gpt-4"],
    "complex": ["gpt-4", "claude-3-5-sonnet"]
}

def select_provider(task_type: str, budget_priority: bool = False):
    """Select provider based on task complexity and budget."""
    
    candidates = TASK_COMPLEXITY.get(task_type, ["gpt-4"])
    
    if budget_priority:
        # Sort by cost (ascending)
        candidates = sorted(
            candidates,
            key=lambda x: PROVIDER_COSTS[x]["input"]
        )
    
    provider_name = candidates[0]
    
    # Initialize the chosen provider
    if "gpt-4" in provider_name:
        return ChatOpenAI(model="gpt-4", temperature=0.7), provider_name
    elif "gpt-3.5" in provider_name:
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7), provider_name
    elif "claude" in provider_name:
        return ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.7), provider_name
    elif "ollama" in provider_name:
        return ChatOllama(model="llama3", temperature=0.7), provider_name

# Test
tasks = [
    ("What is 2+2?", "simple"),
    ("Explain quantum physics", "medium"),
    ("Write a complex business strategy", "complex")
]

for question, complexity in tasks:
    llm, provider = select_provider(complexity, budget_priority=True)
    print(f"\nTask: {question}")
    print(f"Complexity: {complexity}")
    print(f"Selected: {provider}")
    
    response = llm.invoke(question)
    print(f"Response: {response.content[:100]}...")
```

---

## 🎯 Example 7: Task-Specific Provider Routing

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from typing import Literal

# Task categories
TaskType = Literal["code", "creative", "analysis", "qa"]

def route_to_best_provider(task: str, task_type: TaskType):
    """Route to provider best suited for task type."""
    
    routing = {
        "code": {
            "provider": ChatOllama(model="codellama", temperature=0.3),
            "reason": "CodeLlama specialized for coding"
        },
        "creative": {
            "provider": ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.9),
            "reason": "Claude excels at creative writing"
        },
        "analysis": {
            "provider": ChatOpenAI(model="gpt-4", temperature=0.2),
            "reason": "GPT-4 best for analytical reasoning"
        },
        "qa": {
            "provider": ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5),
            "reason": "GPT-3.5 cost-effective for simple Q&A"
        }
    }
    
    config = routing[task_type]
    llm = config["provider"]
    
    print(f"Task type: {task_type}")
    print(f"Routing to: {config['reason']}")
    
    response = llm.invoke(task)
    return response.content

# Test different task types
tasks = [
    ("Write a Python function to sort a list", "code"),
    ("Write a poem about the ocean", "creative"),
    ("Analyze the pros and cons of remote work", "analysis"),
    ("What is the capital of Japan?", "qa")
]

for task, task_type in tasks:
    print(f"\n{'='*50}")
    print(f"Task: {task}")
    result = route_to_best_provider(task, task_type)
    print(f"Result: {result[:150]}...")
```

---

## 🔧 Example 8: Cohere Integration

```python
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
import os

# Set API key
os.environ["COHERE_API_KEY"] = "your-cohere-api-key"

# Initialize Cohere
llm = ChatCohere(
    model="command-r-plus",  # or "command-r", "command"
    temperature=0.7,
    max_tokens=1000
)

# Simple usage
response = llm.invoke("Explain natural language processing")
print(response.content)

# Cohere strengths:
# - Multilingual support (100+ languages)
# - Strong embeddings
# - RAG-optimized models

# With chain
prompt = ChatPromptTemplate.from_template("""
Translate the following to {language}:

Text: {text}

Translation:
""")

chain = prompt | llm

result = chain.invoke({
    "language": "Spanish",
    "text": "Hello, how are you today?"
})

print(result.content)
```

---

## 🌍 Example 9: HuggingFace Integration

```python
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
import os

# Set token (get from huggingface.co)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your-hf-token"

# Use hosted inference API
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.7,
    max_new_tokens=512,
    top_k=50,
)

# Wrap for chat interface
chat_llm = ChatHuggingFace(llm=llm)

# Usage
response = chat_llm.invoke("Explain machine learning")
print(response.content)

# Popular models on HuggingFace:
models = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Llama-2-7b-chat-hf",
    "google/flan-t5-xxl",
    "tiiuae/falcon-7b-instruct",
    "bigcode/starcoder",  # For code
]

# Local HuggingFace models (for privacy)
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100
)

local_llm = HuggingFacePipeline(pipeline=pipe)
response = local_llm.invoke("The future of AI is")
print(response)
```

---

## 🔀 Example 10: Multi-Provider RAG System

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Knowledge base
documents = [
    Document(page_content="Python is a high-level programming language."),
    Document(page_content="Machine learning enables computers to learn from data."),
    Document(page_content="Neural networks are inspired by biological neurons."),
]

# Embeddings (OpenAI for quality)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()

# Multiple LLM options
llms = {
    "gpt-4": ChatOpenAI(model="gpt-4", temperature=0),
    "claude": ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0),
    "llama": ChatOllama(model="llama3", temperature=0)
}

# RAG prompt
prompt = ChatPromptTemplate.from_template("""
Answer based on context:

Context: {context}

Question: {question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function to use any provider
def rag_with_provider(question: str, provider: str = "gpt-4"):
    """RAG with selectable provider."""
    
    llm = llms[provider]
    
    chain = (
        {"context": retriever | format_docs, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke(question)

# Test with different providers
question = "What is machine learning?"

for provider_name in llms.keys():
    print(f"\n{'='*50}")
    print(f"Provider: {provider_name}")
    print("="*50)
    
    try:
        answer = rag_with_provider(question, provider_name)
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"Error: {str(e)}")
```

---

## 🎨 Example 11: Custom Provider Wrapper

```python
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import List, Optional
import requests

class CustomAPIChat(BaseChatModel):
    """Custom wrapper for any API."""
    
    api_url: str
    api_key: str
    model_name: str = "custom-model"
    temperature: float = 0.7
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> ChatResult:
        """Call custom API."""
        
        # Convert messages to API format
        api_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            api_messages.append({
                "role": role,
                "content": msg.content
            })
        
        # Make API request
        response = requests.post(
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "messages": api_messages,
                "temperature": self.temperature,
                "model": self.model_name
            }
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Parse response
        content = result["choices"][0]["message"]["content"]
        
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )
    
    @property
    def _llm_type(self) -> str:
        return "custom-api"

# Usage
custom_llm = CustomAPIChat(
    api_url="https://api.example.com/chat",
    api_key="your-api-key",
    model_name="custom-model-v1"
)

# Use like any other LangChain LLM
response = custom_llm.invoke([HumanMessage(content="Hello!")])
print(response.content)
```

---

## 🔥 Best Practices

### **1. Provider Selection Strategy**
```python
# ✅ Decision matrix
decision_matrix = {
    "simple_qa": "gpt-3.5-turbo",  # Fast, cheap
    "complex_reasoning": "gpt-4",  # Best quality
    "long_context": "claude-3-5-sonnet",  # 200k context
    "code_generation": "codellama",  # Specialized
    "privacy_required": "ollama",  # Local
    "multilingual": "cohere",  # 100+ languages
}
```

### **2. Environment-Specific Configuration**
```python
# ✅ Different providers per environment
import os

if os.getenv("ENVIRONMENT") == "production":
    llm = ChatOpenAI(model="gpt-4")  # Reliable
elif os.getenv("ENVIRONMENT") == "development":
    llm = ChatOllama(model="llama3")  # Free, local
else:
    llm = ChatOpenAI(model="gpt-3.5-turbo")  # Balanced
```

### **3. API Key Management**
```python
# ✅ Use environment variables
import os

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")

# ❌ Never hardcode
# api_key = "sk-abc123..."  # DON'T DO THIS
```

### **4. Implement Retry Logic**
```python
# ✅ Handle failures gracefully
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def call_llm_with_retry(llm, prompt):
    return llm.invoke(prompt)
```

### **5. Monitor Costs**
```python
# ✅ Track token usage
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = llm.invoke("Long prompt...")
    print(f"Tokens: {cb.total_tokens}")
    print(f"Cost: ${cb.total_cost}")
```

### **6. Test Across Providers**
```python
# ✅ Ensure compatibility
providers = ["openai", "anthropic", "ollama"]

for provider in providers:
    try:
        llm = get_llm(provider)
        result = llm.invoke("Test prompt")
        assert len(result.content) > 0
        print(f"✅ {provider} works")
    except Exception as e:
        print(f"❌ {provider} failed: {e}")
```

---

## 📊 Provider Comparison Summary

| Feature | OpenAI | Anthropic | Ollama | Azure | Gemini |
|---------|--------|-----------|--------|-------|--------|
| Cost | $$$ | $$$ | Free | $$$ | $-$$ |
| Speed | Fast | Fast | Medium | Fast | Fast |
| Context | 128k | 200k | 32k | 128k | 32k |
| Quality | Excellent | Excellent | Good | Excellent | Very Good |
| Privacy | Cloud | Cloud | Local | Cloud | Cloud |
| Offline | No | No | Yes | No | No |
| Enterprise | Yes | Yes | Yes | Yes | Yes |

---

## 🎯 Practical Exercise

```python
"""
Exercise: Build a Multi-Provider Smart Router

Create an intelligent system that:

1. Provider Registry
   - Register multiple providers (OpenAI, Anthropic, Ollama)
   - Track capabilities and costs

2. Smart Routing
   - Analyze query complexity
   - Consider budget constraints
   - Route to optimal provider

3. Fallback Mechanism
   - Try primary provider
   - Fall back to secondary if fails
   - Track failure patterns

4. Cost Tracking
   - Monitor token usage
   - Calculate costs per provider
   - Generate cost reports

5. Performance Monitoring
   - Track latency per provider
   - Measure response quality
   - Log all interactions

6. A/B Testing
   - Compare providers for same queries
   - Measure quality differences
   - Optimize routing rules

Requirements:
- Support at least 3 providers
- Implement intelligent routing
- Add comprehensive error handling
- Track costs and performance
- Include fallback logic
- Provide usage analytics

Test with diverse queries:
- Simple: "What is 2+2?"
- Medium: "Explain machine learning"
- Complex: "Develop a business strategy"
- Code: "Write a sorting algorithm"
"""

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from typing import Dict, List, Literal
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ProviderMetrics:
    total_calls: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    failures: int = 0
    avg_latency: float = 0.0

class SmartRouter:
    def __init__(self):
        # TODO: Initialize providers
        # TODO: Setup metrics tracking
        # TODO: Configure routing rules
        pass
    
    def classify_query(self, query: str) -> str:
        """Classify query complexity."""
        # TODO: Implement classification
        pass
    
    def select_provider(self, query_type: str, budget_mode: bool = False):
        """Select optimal provider."""
        # TODO: Implement selection logic
        pass
    
    def invoke_with_fallback(self, query: str):
        """Invoke with fallback mechanism."""
        # TODO: Implement fallback
        pass
    
    def track_metrics(self, provider: str, tokens: int, latency: float):
        """Track provider metrics."""
        # TODO: Update metrics
        pass
    
    def get_analytics(self):
        """Get usage analytics."""
        # TODO: Return analytics
        pass

# TODO: Implement and test
# router = SmartRouter()
# result = router.invoke_with_fallback("Explain quantum computing")
# analytics = router.get_analytics()
```

---

## ✅ Key Takeaways

1. **LangChain supports 50+ providers** - unified interface
2. **Each provider has strengths** - choose based on use case
3. **OpenAI for reasoning** - GPT-4 excels at complex tasks
4. **Claude for long context** - 200k token context window
5. **Ollama for privacy** - run models locally
6. **Azure for enterprise** - compliance and SLA
7. **Implement fallbacks** - handle provider failures
8. **Cost optimization** - route based on task complexity
9. **Monitor performance** - track latency and quality
10. **Environment variables** - secure API key management

---

## 📝 Understanding Check

1. When would you choose Anthropic Claude over OpenAI?
2. What are the benefits of using Ollama?
3. How do you implement provider fallback?
4. What factors should influence provider selection?

**Ready for the final section on Deployment & Production Best Practices?** We'll cover FastAPI integration, monitoring, scaling, and taking your LangChain apps to production! 🏭

Or would you like to:
- See the exercise solution?
- Practice more with different providers?
- Deep dive into specific integrations?