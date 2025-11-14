# Section 2: Prompts and Prompt Templates 📝

Prompts are the instructions you give to LLMs. Prompt Templates make prompts reusable, dynamic, and maintainable.

---

## 🎯 Why Prompt Templates?

**Without Templates (Bad):**
```python
# Hardcoded, not reusable
response = llm.invoke("Translate 'Hello' to French")
response = llm.invoke("Translate 'Goodbye' to French")
response = llm.invoke("Translate 'Thank you' to French")
# Repetitive and error-prone!
```

**With Templates (Good):**
```python
template = PromptTemplate.from_template("Translate '{text}' to {language}")
prompt = template.invoke({"text": "Hello", "language": "French"})
# Reusable and maintainable!
```

---

## 📋 Types of Prompt Templates

### **1. PromptTemplate** - Simple string templates
### **2. ChatPromptTemplate** - For chat models (most common)
### **3. MessagesPlaceholder** - For dynamic message insertion
### **4. FewShotPromptTemplate** - For examples-based prompting
### **5. PipelinePromptTemplate** - For composing multiple templates

---

## 💻 Example 1: Basic PromptTemplate

```python
from langchain_core.prompts import PromptTemplate

# Method 1: Using from_template (recommended)
template = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {topic}."
)

# Format the template
prompt = template.invoke({"adjective": "funny", "topic": "programmers"})
print(prompt)
# Output: Tell me a funny joke about programmers.

# Method 2: Explicit definition
template = PromptTemplate(
    input_variables=["adjective", "topic"],
    template="Tell me a {adjective} joke about {topic}."
)

# Using with an LLM
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
chain = template | llm  # LCEL syntax!

response = chain.invoke({"adjective": "funny", "topic": "data scientists"})
print(response.content)
```

---

## 💬 Example 2: ChatPromptTemplate (Most Important!)

Chat models work with **messages**, not plain strings. Messages have **roles**:
- **System**: Instructions for the AI's behavior
- **Human**: User input
- **AI**: Assistant responses
- **Function/Tool**: Results from function calls

```python
from langchain_core.prompts import ChatPromptTemplate

# Create a chat prompt with multiple messages
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant specializing in {domain}."),
    ("human", "Hello! Can you help me?"),
    ("ai", "Of course! I'd be happy to help with {domain}. What do you need?"),
    ("human", "{user_input}")
])

# Format the prompt
messages = prompt.invoke({
    "domain": "machine learning",
    "user_input": "Explain gradient descent simply."
})

print(messages)
# Shows the full conversation structure

# Using with an LLM
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
chain = prompt | llm

response = chain.invoke({
    "domain": "machine learning",
    "user_input": "Explain gradient descent simply."
})

print(response.content)
```

**Output Structure:**
```python
messages = [
    SystemMessage(content="You are a helpful AI assistant specializing in machine learning."),
    HumanMessage(content="Hello! Can you help me?"),
    AIMessage(content="Of course! I'd be happy to help with machine learning. What do you need?"),
    HumanMessage(content="Explain gradient descent simply.")
]
```

---

## 🎨 Example 3: Different Template Styles

### **Style 1: f-string style (default)**
```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "Summarize this text in {num_sentences} sentences:\n\n{text}"
)
```

### **Style 2: Jinja2 style (powerful)**
```python
template = PromptTemplate.from_template(
    """
    Summarize this text in {{ num_sentences }} sentences:
    
    {% if include_keywords %}
    Focus on these keywords: {{ keywords }}
    {% endif %}
    
    Text: {{ text }}
    """,
    template_format="jinja2"
)

prompt = template.invoke({
    "num_sentences": 3,
    "text": "Long article here...",
    "include_keywords": True,
    "keywords": "AI, machine learning"
})
```

### **Style 3: mustache style**
```python
template = PromptTemplate.from_template(
    "Hello {{name}}, welcome to {{platform}}!",
    template_format="mustache"
)
```

---

## 🔧 Example 4: SystemMessage and Message Types

```python
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Detailed message construction
system_template = SystemMessagePromptTemplate.from_template(
    "You are an expert {role}. "
    "You always provide {response_style} responses. "
    "Your expertise level is {expertise_level}/10."
)

human_template = HumanMessagePromptTemplate.from_template(
    "{user_question}"
)

# Combine into chat prompt
chat_prompt = ChatPromptTemplate.from_messages([
    system_template,
    human_template
])

# Use it
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
chain = chat_prompt | llm

response = chain.invoke({
    "role": "Python developer",
    "response_style": "concise and practical",
    "expertise_level": 9,
    "user_question": "How do I handle exceptions in async functions?"
})

print(response.content)
```

---

## 🎓 Example 5: Few-Shot Prompting (Learning from Examples)

Few-shot prompting teaches the model by providing examples.

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# Define examples
examples = [
    {
        "input": "What's 2+2?",
        "output": "The answer is 4. Here's why: 2 + 2 = 4"
    },
    {
        "input": "What's 5*3?",
        "output": "The answer is 15. Here's why: 5 * 3 = 15"
    },
    {
        "input": "What's 10-7?",
        "output": "The answer is 3. Here's why: 10 - 7 = 3"
    }
]

# Define the example template
example_template = PromptTemplate.from_template(
    "Question: {input}\nAnswer: {output}"
)

# Create few-shot prompt
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="You are a math tutor. Answer questions by showing your work.",
    suffix="Question: {input}\nAnswer:",
    input_variables=["input"]
)

# Generate the prompt
prompt = few_shot_prompt.invoke({"input": "What's 8+5?"})
print(prompt.to_string())

# Use with LLM
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
response = llm.invoke(prompt.to_string())
print(response.content)
```

**Output:**
```
You are a math tutor. Answer questions by showing your work.

Question: What's 2+2?
Answer: The answer is 4. Here's why: 2 + 2 = 4

Question: What's 5*3?
Answer: The answer is 15. Here's why: 5 * 3 = 15

Question: What's 10-7?
Answer: The answer is 3. Here's why: 10 - 7 = 3

Question: What's 8+5?
Answer:
```

---

## 🎯 Example 6: Dynamic Few-Shot with ExampleSelector

When you have many examples, use a selector to pick the most relevant ones.

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Many examples
examples = [
    {"input": "How do I import pandas?", "output": "import pandas as pd"},
    {"input": "How do I read a CSV?", "output": "df = pd.read_csv('file.csv')"},
    {"input": "How do I filter rows?", "output": "df[df['column'] > 5]"},
    {"input": "How do I sort data?", "output": "df.sort_values('column')"},
    {"input": "How do I handle missing values?", "output": "df.fillna(0) or df.dropna()"},
    {"input": "How do I group data?", "output": "df.groupby('column').mean()"},
]

# Create example selector
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    FAISS,
    k=2  # Select 2 most similar examples
)

# Create few-shot prompt with selector
example_template = PromptTemplate.from_template(
    "Question: {input}\nAnswer: {output}"
)

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,  # Dynamic selection!
    example_prompt=example_template,
    prefix="You are a Python pandas expert. Answer concisely.",
    suffix="Question: {input}\nAnswer:",
    input_variables=["input"]
)

# Test it - it will select the most relevant examples
prompt = few_shot_prompt.invoke({"input": "How do I remove duplicates?"})
print(prompt.to_string())
```

---

## 📨 Example 7: MessagesPlaceholder (For Chat History)

Used when you need to insert a variable number of messages dynamically.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Template with placeholder for chat history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),  # Dynamic!
    ("human", "{user_input}")
])

# Example usage with history
chat_history = [
    HumanMessage(content="What's machine learning?"),
    AIMessage(content="Machine learning is a subset of AI where systems learn from data."),
    HumanMessage(content="What are its types?"),
    AIMessage(content="Main types are supervised, unsupervised, and reinforcement learning."),
]

# Format with history
messages = prompt.invoke({
    "chat_history": chat_history,
    "user_input": "Can you explain supervised learning in detail?"
})

# Use with LLM
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
response = llm.invoke(messages)
print(response.content)
```

---

## 🧩 Example 8: Partial Prompts (Pre-filling Variables)

```python
from langchain_core.prompts import PromptTemplate
from datetime import datetime

# Template with multiple variables
template = PromptTemplate.from_template(
    "Today is {date}. You are a {role}. Answer this: {question}"
)

# Partially fill some variables
partial_template = template.partial(
    date=datetime.now().strftime("%Y-%m-%d"),
    role="senior Python developer"
)

# Now only need to provide 'question'
prompt = partial_template.invoke({"question": "How do I optimize list comprehensions?"})
print(prompt.to_string())

# Using with function for dynamic values
def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")

template_with_fn = template.partial(
    date=get_current_date,  # Function, not value!
    role="data scientist"
)

# Date is always current when invoked
prompt = template_with_fn.invoke({"question": "Best practices for feature engineering?"})
```

---

## 🔗 Example 9: PipelinePromptTemplate (Composing Templates)

```python
from langchain_core.prompts import PromptTemplate, PipelinePromptTemplate

# Component templates
introduction_template = PromptTemplate.from_template(
    "You are a {role} with {years} years of experience."
)

task_template = PromptTemplate.from_template(
    "Your task is to {task}."
)

constraints_template = PromptTemplate.from_template(
    "Constraints: {constraints}"
)

question_template = PromptTemplate.from_template(
    "Question: {question}"
)

# Combine them
final_template = PromptTemplate.from_template(
    """{introduction}

{task}

{constraints}

{question}"""
)

pipeline_prompt = PipelinePromptTemplate(
    final_prompt=final_template,
    pipeline_prompts=[
        ("introduction", introduction_template),
        ("task", task_template),
        ("constraints", constraints_template),
        ("question", question_template),
    ]
)

# Use it
prompt = pipeline_prompt.invoke({
    "role": "ML engineer",
    "years": 5,
    "task": "design a recommendation system",
    "constraints": "Must work with cold-start users, handle 1M+ users",
    "question": "What architecture would you suggest?"
})

print(prompt.to_string())
```

---

## 🎨 Example 10: Custom Prompt Templates

```python
from langchain_core.prompts import StringPromptTemplate
from typing import Dict, Any

class CustomCodePromptTemplate(StringPromptTemplate):
    """Custom template for code generation."""
    
    def format(self, **kwargs: Any) -> str:
        # Custom logic
        language = kwargs.get("language", "Python")
        task = kwargs.get("task", "")
        include_tests = kwargs.get("include_tests", False)
        
        prompt = f"Write a {language} function that {task}.\n\n"
        prompt += "Requirements:\n"
        prompt += "- Use type hints\n"
        prompt += "- Add docstrings\n"
        prompt += "- Follow PEP 8 style\n"
        
        if include_tests:
            prompt += "- Include unit tests\n"
        
        return prompt

# Use custom template
template = CustomCodePromptTemplate(input_variables=["language", "task", "include_tests"])

prompt = template.invoke({
    "language": "Python",
    "task": "calculates the factorial of a number",
    "include_tests": True
})

print(prompt)
```

---

## 🔥 Best Practices for Prompts

### **1. Be Specific and Clear**
```python
# ❌ Bad
"Explain ML"

# ✅ Good
"Explain supervised machine learning in 3 sentences for someone with a programming background"
```

### **2. Use System Messages for Behavior**
```python
ChatPromptTemplate.from_messages([
    ("system", "You are a Python expert. Always provide code examples. Keep responses under 200 words."),
    ("human", "{question}")
])
```

### **3. Include Context**
```python
template = """Given this context: {context}

Answer this question: {question}

Base your answer only on the provided context."""
```

### **4. Use Few-Shot for Consistency**
```python
# Teach the model your desired format
examples = [
    {"input": "task 1", "output": "formatted response 1"},
    {"input": "task 2", "output": "formatted response 2"},
]
```

### **5. Validate Inputs**
```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["topic", "length"],
    template="Write a {length} article about {topic}.",
    validate_template=True  # Ensures all variables are provided
)
```

---

## 🎯 Practical Exercise

```python
"""
Exercise: Build a Code Review Assistant Prompt

Create a ChatPromptTemplate that:
1. Has a system message defining the assistant as a senior code reviewer
2. Accepts parameters: programming_language, code_snippet, focus_areas
3. Includes few-shot examples of good code reviews
4. Outputs structured feedback

Test it with sample code.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# TODO: Implement the prompt template
# TODO: Test with a code snippet

# Example test case:
code = """
def calculate_sum(numbers):
    total = 0
    for i in range(len(numbers)):
        total = total + numbers[i]
    return total
"""

# Expected output: Structured review with suggestions
```

---

## ✅ Key Takeaways

1. **ChatPromptTemplate is the standard** for chat models
2. **Use system messages** to define assistant behavior
3. **Few-shot prompting** improves consistency and quality
4. **MessagesPlaceholder** enables dynamic chat history
5. **LCEL syntax** (`template | llm`) is the modern way
6. **Partial prompts** reduce repetition
7. **Semantic selectors** choose relevant examples automatically

---

## 📊 Quick Comparison

| Template Type | Use Case | Complexity |
|--------------|----------|------------|
| PromptTemplate | Simple string formatting | Low |
| ChatPromptTemplate | Chat models (most common) | Medium |
| FewShotPromptTemplate | Learning from examples | Medium |
| MessagesPlaceholder | Dynamic chat history | Medium |
| PipelinePromptTemplate | Composing multiple prompts | High |

---

## 📝 Understanding Check

1. What's the difference between PromptTemplate and ChatPromptTemplate?
2. When would you use Few-Shot prompting?
3. What is MessagesPlaceholder used for?

**Ready for the next section on Messages and Output Parsers?** Or would you like to:
- See the exercise solution?
- Practice more with prompts?
- Ask questions?