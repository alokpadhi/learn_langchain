# Section 4: LCEL - LangChain Expression Language (Runnable Interface) 🚀

LCEL is the **game-changer** introduced in LangChain v1.0+. It's a declarative way to compose chains using the pipe operator `|`.

---

## 🎯 What is LCEL?

**LCEL (LangChain Expression Language)** is a unified interface for building chains. Think of it as the "modern way" to compose LangChain components.

### **The Core Concept: Runnable**

Everything in LCEL implements the `Runnable` interface:
- Prompts are Runnables
- Models are Runnables
- Output Parsers are Runnables
- Chains are Runnables
- Even custom functions can be Runnables

This means **you can chain them together** with the pipe operator `|`.

---

## 🔄 Old Way vs New Way

### **Pre-v1.0 (Legacy)**
```python
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
llm = ChatOpenAI(model="gpt-4")

# Old way - verbose, less flexible
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(topic="Python")
```

### **v1.0+ (LCEL)**
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
llm = ChatOpenAI(model="gpt-4")
parser = StrOutputParser()

# New way - clean, composable, powerful
chain = prompt | llm | parser
result = chain.invoke({"topic": "Python"})
```

---

## 💡 Why LCEL is Better

| Feature | Legacy Chains | LCEL |
|---------|--------------|------|
| Syntax | Verbose | Clean, readable |
| Streaming | Manual setup | Built-in |
| Async | Complex | Native support |
| Parallel execution | Manual | Automatic |
| Debugging | Limited | Excellent (LangSmith) |
| Composability | Rigid | Highly flexible |
| Type hints | Poor | Strong |

---

## 🔧 Core Runnable Methods

Every Runnable has these methods:

```python
# 1. invoke() - Synchronous, single input
result = chain.invoke({"input": "value"})

# 2. batch() - Synchronous, multiple inputs
results = chain.batch([{"input": "val1"}, {"input": "val2"}])

# 3. stream() - Streaming output
for chunk in chain.stream({"input": "value"}):
    print(chunk)

# 4. ainvoke() - Async, single input
result = await chain.ainvoke({"input": "value"})

# 5. abatch() - Async, multiple inputs
results = await chain.abatch([{"input": "val1"}, {"input": "val2"}])

# 6. astream() - Async streaming
async for chunk in chain.astream({"input": "value"}):
    print(chunk)
```

---

## 💻 Example 1: Basic LCEL Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Components
prompt = ChatPromptTemplate.from_template("Write a haiku about {topic}")
model = ChatOpenAI(model="gpt-4")
parser = StrOutputParser()

# Compose with pipe operator
chain = prompt | model | parser

# Invoke
result = chain.invoke({"topic": "machine learning"})
print(result)

# Output:
# Neural networks learn
# Patterns hidden in the data
# Intelligence grows
```

**What's happening behind the scenes:**
```python
# Step 1: prompt.invoke() → PromptValue
prompt_value = prompt.invoke({"topic": "machine learning"})

# Step 2: model.invoke() → AIMessage
ai_message = model.invoke(prompt_value)

# Step 3: parser.invoke() → str
final_result = parser.invoke(ai_message)
```

---

## 🔗 Example 2: Multi-Step Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Step 1: Generate topic ideas
idea_prompt = ChatPromptTemplate.from_template(
    "Generate 3 blog post ideas about {subject}"
)

# Step 2: Expand the first idea
expansion_prompt = ChatPromptTemplate.from_template(
    "Take this blog idea and write an outline:\n\n{idea}"
)

model = ChatOpenAI(model="gpt-4")
parser = StrOutputParser()

# Build chains
idea_chain = idea_prompt | model | parser

expansion_chain = expansion_prompt | model | parser

# Use them sequentially
subject = "artificial intelligence ethics"

# Get ideas
ideas = idea_chain.invoke({"subject": subject})
print("Ideas:\n", ideas)

# Expand first idea
outline = expansion_chain.invoke({"idea": ideas})
print("\nOutline:\n", outline)
```

---

## 🎨 Example 3: RunnablePassthrough (Passing Data Through)

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    """Answer the question based on context:
    
    Context: {context}
    Question: {question}
    """
)

model = ChatOpenAI(model="gpt-4")
parser = StrOutputParser()

# RunnablePassthrough passes input through unchanged
chain = (
    {
        "context": RunnablePassthrough(),  # Passes through as-is
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | parser
)

result = chain.invoke({
    "context": "LangChain is a framework for building LLM applications.",
    "question": "What is LangChain?"
})

print(result)
# Output: LangChain is a framework for building applications with large language models.
```

---

## 🔄 Example 4: RunnableParallel (Parallel Execution)

```python
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4")

# Create multiple chains
pros_chain = (
    ChatPromptTemplate.from_template("List 3 pros of {topic}")
    | model
    | StrOutputParser()
)

cons_chain = (
    ChatPromptTemplate.from_template("List 3 cons of {topic}")
    | model
    | StrOutputParser()
)

summary_chain = (
    ChatPromptTemplate.from_template("Summarize {topic} in one sentence")
    | model
    | StrOutputParser()
)

# Execute in parallel
parallel_chain = RunnableParallel(
    pros=pros_chain,
    cons=cons_chain,
    summary=summary_chain
)

result = parallel_chain.invoke({"topic": "remote work"})

print("Pros:", result["pros"])
print("\nCons:", result["cons"])
print("\nSummary:", result["summary"])

# All three chains run simultaneously!
```

---

## 🎯 Example 5: RunnableLambda (Custom Functions)

```python
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Custom processing function
def uppercase_output(text: str) -> str:
    return text.upper()

def add_emoji(text: str) -> str:
    return f"✨ {text} ✨"

# Build chain with custom functions
model = ChatOpenAI(model="gpt-4")

chain = (
    ChatPromptTemplate.from_template("Say hello to {name}")
    | model
    | StrOutputParser()
    | RunnableLambda(uppercase_output)  # Custom function as Runnable!
    | RunnableLambda(add_emoji)
)

result = chain.invoke({"name": "Alice"})
print(result)
# Output: ✨ HELLO ALICE! NICE TO MEET YOU! ✨
```

**Alternative syntax with decorator:**
```python
from langchain_core.runnables import chain as runnable_chain

@runnable_chain
def custom_chain(inputs: dict) -> str:
    """Custom processing logic."""
    name = inputs["name"]
    return f"Processed: {name.upper()}"

# Use it in a chain
full_chain = custom_chain | model | StrOutputParser()
```

---

## 🌊 Example 6: Streaming with LCEL

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "Write a short story about {topic}"
)
model = ChatOpenAI(model="gpt-4", streaming=True)
parser = StrOutputParser()

chain = prompt | model | parser

# Stream tokens as they arrive
print("Streaming story:")
for chunk in chain.stream({"topic": "a robot learning to paint"}):
    print(chunk, end="", flush=True)

print("\n\nDone!")
```

---

## ⚡ Example 7: Async LCEL

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

async def async_example():
    prompt = ChatPromptTemplate.from_template("Explain {concept} briefly")
    model = ChatOpenAI(model="gpt-4")
    parser = StrOutputParser()
    
    chain = prompt | model | parser
    
    # Single async invocation
    result = await chain.ainvoke({"concept": "neural networks"})
    print("Single result:", result)
    
    # Batch async (runs in parallel)
    concepts = ["gradient descent", "backpropagation", "activation functions"]
    results = await chain.abatch([{"concept": c} for c in concepts])
    
    for concept, result in zip(concepts, results):
        print(f"\n{concept}:")
        print(result)
    
    # Async streaming
    print("\n\nStreaming:")
    async for chunk in chain.astream({"concept": "transformers"}):
        print(chunk, end="", flush=True)

# Run async code
asyncio.run(async_example())
```

---

## 🔀 Example 8: RunnableBranch (Conditional Logic)

```python
from langchain_core.runnables import RunnableBranch
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4")
parser = StrOutputParser()

# Different prompts based on language
python_prompt = ChatPromptTemplate.from_template(
    "Explain {concept} with Python code examples"
)

javascript_prompt = ChatPromptTemplate.from_template(
    "Explain {concept} with JavaScript code examples"
)

general_prompt = ChatPromptTemplate.from_template(
    "Explain {concept} in general terms"
)

# Branch based on condition
branch = RunnableBranch(
    (lambda x: x["language"] == "python", python_prompt | model | parser),
    (lambda x: x["language"] == "javascript", javascript_prompt | model | parser),
    general_prompt | model | parser  # Default
)

# Test different branches
result1 = branch.invoke({"language": "python", "concept": "recursion"})
print("Python version:\n", result1)

result2 = branch.invoke({"language": "javascript", "concept": "recursion"})
print("\nJavaScript version:\n", result2)

result3 = branch.invoke({"language": "other", "concept": "recursion"})
print("\nGeneral version:\n", result3)
```

---

## 🎨 Example 9: Complex Chain with Multiple Operations

```python
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4")
parser = StrOutputParser()

# Custom function to format context
def format_docs(docs: list) -> str:
    return "\n\n".join(docs)

# Simulate document retrieval
def retrieve_docs(query: str) -> list:
    """Simulate retrieving relevant documents."""
    return [
        "Document 1: Python is a high-level programming language.",
        "Document 2: Python is widely used in data science.",
        "Document 3: Python has a simple, readable syntax."
    ]

# Build complex chain
chain = (
    {
        "context": RunnableLambda(lambda x: retrieve_docs(x["question"])) 
                   | RunnableLambda(format_docs),
        "question": RunnablePassthrough() | (lambda x: x["question"])
    }
    | ChatPromptTemplate.from_template(
        """Answer based on context:
        
        Context: {context}
        
        Question: {question}
        """
    )
    | model
    | parser
)

result = chain.invoke({"question": "What is Python used for?"})
print(result)
```

---

## 🔧 Example 10: RunnableMap (Transforming Inputs)

```python
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4")

# Process input in multiple ways
def get_word_count(text: str) -> int:
    return len(text.split())

def get_char_count(text: str) -> int:
    return len(text)

def analyze_text(text: str) -> dict:
    return {
        "words": get_word_count(text),
        "chars": get_char_count(text),
        "has_code": "```" in text
    }

# Chain that analyzes and summarizes
chain = (
    {
        "text": RunnablePassthrough(),
        "analysis": RunnablePassthrough() | analyze_text
    }
    | RunnablePassthrough()
    | (lambda x: f"Text has {x['analysis']['words']} words, "
                  f"{x['analysis']['chars']} characters. "
                  f"Contains code: {x['analysis']['has_code']}")
)

result = chain.invoke(
    "Machine learning is fascinating. ```python\nprint('hello')\n```"
)
print(result)
# Output: Text has 6 words, 62 characters. Contains code: True
```

---

## 📊 Example 11: Batch Processing

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Translate '{text}' to {language}")
model = ChatOpenAI(model="gpt-4")
parser = StrOutputParser()

chain = prompt | model | parser

# Process multiple inputs at once
inputs = [
    {"text": "Hello", "language": "Spanish"},
    {"text": "Goodbye", "language": "French"},
    {"text": "Thank you", "language": "German"},
    {"text": "Good morning", "language": "Italian"},
]

# Batch processing (potentially parallel on backend)
results = chain.batch(inputs)

for inp, result in zip(inputs, results):
    print(f"{inp['text']} → {inp['language']}: {result}")

# Output:
# Hello → Spanish: Hola
# Goodbye → French: Au revoir
# Thank you → German: Danke
# Good morning → Italian: Buongiorno
```

---

## 🎯 Example 12: Config and Runtime Arguments

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
model = ChatOpenAI()  # Model not specified
parser = StrOutputParser()

chain = prompt | model | parser

# Pass config at runtime
result = chain.invoke(
    {"topic": "quantum computing"},
    config={
        "configurable": {
            "model": "gpt-4",
            "temperature": 0.9
        }
    }
)

print(result)
```

---

## 🔥 Example 13: Error Handling in LCEL

```python
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def risky_operation(text: str) -> str:
    """Might raise an exception."""
    if "error" in text.lower():
        raise ValueError("Error keyword detected!")
    return text.upper()

def error_handler(error):
    """Handle errors gracefully."""
    return f"⚠️ Error occurred: {str(error)}"

model = ChatOpenAI(model="gpt-4")

# Chain with error handling
chain = (
    ChatPromptTemplate.from_template("Process: {text}")
    | model
    | StrOutputParser()
    | RunnableLambda(risky_operation).with_fallbacks(
        [RunnableLambda(lambda x: "Fallback: Safe processing")]
    )
)

# This will use fallback
try:
    result = chain.invoke({"text": "This contains error keyword"})
    print(result)
except Exception as e:
    print(f"Caught: {e}")
```

---

## 📋 LCEL Operators Summary

| Operator | Purpose | Example |
|----------|---------|---------|
| `\|` | Pipe/Chain | `prompt \| model \| parser` |
| `RunnablePassthrough` | Pass data unchanged | Keep original input |
| `RunnableParallel` | Run in parallel | Multiple operations simultaneously |
| `RunnableLambda` | Custom function | Any Python function |
| `RunnableBranch` | Conditional logic | Different paths based on input |
| `RunnableMap` | Transform inputs | Modify data structure |

---

## 🎯 Practical Exercise

```python
"""
Exercise: Build a Multi-Stage Text Processor

Create an LCEL chain that:
1. Takes a text input
2. Analyzes sentiment (positive/negative/neutral) in parallel with:
   - Extracting key topics
   - Counting words
3. Generates a summary based on all the analysis
4. Formats the output nicely

Use: RunnableParallel, RunnableLambda, and standard LCEL components
"""

from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# TODO: Define analysis functions
def analyze_sentiment(text: str) -> str:
    # Use LLM or simple logic
    pass

def extract_topics(text: str) -> list:
    # Extract main topics
    pass

def count_words(text: str) -> int:
    return len(text.split())

# TODO: Build the chain
# Hint: Use RunnableParallel for parallel operations

# Test text
sample_text = """
Artificial intelligence is transforming industries worldwide.
Machine learning algorithms are becoming more sophisticated,
enabling breakthrough applications in healthcare, finance, and education.
However, ethical concerns about AI bias and privacy remain important challenges.
"""

# TODO: Invoke and print results
```

---

## ✅ Key Takeaways

1. **LCEL uses pipe operator `|`** for clean, readable chains
2. **Everything is a Runnable** - prompts, models, parsers, functions
3. **Built-in support** for streaming, async, and batch processing
4. **RunnableParallel** executes operations simultaneously
5. **RunnableLambda** converts any function to a Runnable
6. **RunnableBranch** enables conditional logic
7. **RunnablePassthrough** preserves original data
8. **Method uniformity**: invoke, batch, stream, + async versions
9. **Config can be passed at runtime** for dynamic behavior
10. **Error handling** via fallbacks and try-except

---

## 📊 LCEL vs Legacy Comparison

```python
# ❌ Legacy way
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)
result = chain.run(input="test")

# ✅ LCEL way  
chain = prompt | llm | parser
result = chain.invoke({"input": "test"})

# LCEL advantages:
# - Cleaner syntax
# - Better composability
# - Native streaming/async
# - Easier debugging
# - More flexible
```

---

## 📝 Understanding Check

1. What is the main advantage of LCEL over legacy chains?
2. How do you run operations in parallel using LCEL?
3. What's the difference between `invoke()` and `stream()`?
4. How do you convert a regular Python function to a Runnable?

**Ready for the next section on Memory Systems?** Or would you like to:
- See the exercise solution?
- Practice more with LCEL?
- Ask questions about specific Runnable types?