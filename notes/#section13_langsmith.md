# Section 13: LangSmith (Monitoring & Debugging) 📊

LangSmith is Anthropic's platform for debugging, testing, evaluating, and monitoring LLM applications in development and production.

---

## 🎯 What is LangSmith?

**LangSmith** = Observability and evaluation platform for LLM applications

**Why LangSmith?**

Without LangSmith:
```python
# 😰 Problems:
# - Can't see what LLM is doing internally
# - Hard to debug chains with multiple steps
# - No visibility into token usage/costs
# - Can't track performance over time
# - Difficult to test changes
```

With LangSmith:
```python
# ✅ Solutions:
# - Trace every LLM call
# - See full chain execution
# - Track costs and latency
# - Monitor production performance
# - A/B test prompts
# - Create evaluation datasets
```

---

## 📋 Key Features

### **1. Tracing**
See every step of your LLM application execution

### **2. Debugging**
Inspect inputs, outputs, and intermediate steps

### **3. Evaluation**
Test your application against datasets

### **4. Monitoring**
Track production metrics and performance

### **5. Datasets**
Create and manage test datasets

### **6. Prompt Hub**
Version control for prompts

---

## 🚀 Setup

```python
# Install LangSmith
# pip install langsmith

import os

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"  # Get from smith.langchain.com
os.environ["LANGCHAIN_PROJECT"] = "my-project"  # Project name

# That's it! Tracing is now enabled automatically
```

---

## 💻 Example 1: Basic Tracing

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Enable tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "basic-tracing-demo"

# Create a simple chain
llm = ChatOpenAI(model="gpt-4", temperature=0)
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | llm | StrOutputParser()

# Run - automatically traced!
result = chain.invoke({"topic": "programming"})
print(result)

# Check LangSmith dashboard at smith.langchain.com
# You'll see:
# - Full trace of execution
# - Input/output at each step
# - Latency for each component
# - Token usage
# - Cost estimate
```

**What you see in LangSmith:**
```
Trace: Tell me a joke
├─ PromptTemplate: Format prompt
│  ├─ Input: {"topic": "programming"}
│  └─ Output: "Tell me a joke about programming"
│
├─ ChatOpenAI: Generate response
│  ├─ Input: "Tell me a joke about programming"
│  ├─ Tokens: 45
│  ├─ Latency: 1.2s
│  └─ Output: "Why do programmers prefer dark mode?..."
│
└─ StrOutputParser: Parse output
   └─ Output: "Why do programmers prefer dark mode?..."
```

---

## 🔍 Example 2: Tracing Complex Chains

```python
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "rag-tracing-demo"

# Setup RAG
documents = [
    Document(page_content="LangSmith helps debug LLM applications."),
    Document(page_content="LangSmith provides tracing and monitoring."),
    Document(page_content="You can create evaluation datasets in LangSmith."),
]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(model="gpt-4", temperature=0)

prompt = ChatPromptTemplate.from_template("""
Answer based on context:

Context: {context}

Question: {question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Run - see detailed trace in LangSmith
result = rag_chain.invoke("What does LangSmith do?")
print(result)

# LangSmith shows:
# - Document retrieval step
# - Which docs were retrieved
# - How they were formatted
# - Full prompt sent to LLM
# - LLM response
# - Token usage at each step
```

---

## 🏷️ Example 3: Custom Tags and Metadata

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "tagged-runs"

llm = ChatOpenAI(model="gpt-4", temperature=0)
prompt = ChatPromptTemplate.from_template("Explain {concept} to a {audience}")
chain = prompt | llm

# Add tags and metadata to traces
result = chain.invoke(
    {"concept": "machine learning", "audience": "5 year old"},
    config={
        "tags": ["educational", "eli5", "ml"],
        "metadata": {
            "user_id": "user_123",
            "session_id": "session_456",
            "version": "v1.2",
            "experiment": "simplified_explanations"
        }
    }
)

print(result.content)

# Benefits:
# - Filter traces by tags in LangSmith
# - Search by metadata
# - Track experiments
# - Analyze user-specific performance
```

---

## 📊 Example 4: Creating Evaluation Datasets

```python
from langsmith import Client

# Initialize client
client = Client()

# Create a dataset
dataset_name = "qa_evaluation_dataset"

# Example: Question-Answer pairs
examples = [
    {
        "inputs": {"question": "What is machine learning?"},
        "outputs": {"answer": "Machine learning is a subset of AI that enables systems to learn from data."}
    },
    {
        "inputs": {"question": "What is supervised learning?"},
        "outputs": {"answer": "Supervised learning uses labeled data to train models."}
    },
    {
        "inputs": {"question": "What is a neural network?"},
        "outputs": {"answer": "A neural network is a computational model inspired by biological neurons."}
    },
]

# Create dataset
dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="QA evaluation dataset for ML concepts"
)

# Add examples
for example in examples:
    client.create_example(
        inputs=example["inputs"],
        outputs=example["outputs"],
        dataset_id=dataset.id
    )

print(f"Created dataset '{dataset_name}' with {len(examples)} examples")

# Now you can evaluate your chain against this dataset
```

---

## 🧪 Example 5: Running Evaluations

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client
from langsmith.evaluation import evaluate

# System to evaluate
llm = ChatOpenAI(model="gpt-4", temperature=0)
prompt = ChatPromptTemplate.from_template("Answer this question: {question}")
chain = prompt | llm | StrOutputParser()

# Wrapper function
def qa_system(inputs: dict) -> dict:
    """Wrapper for evaluation."""
    answer = chain.invoke({"question": inputs["question"]})
    return {"answer": answer}

# Custom evaluator
def check_answer_length(run, example):
    """Check if answer is not too short."""
    answer = run.outputs.get("answer", "")
    
    # Score 1 if answer is at least 20 chars, 0 otherwise
    score = 1.0 if len(answer) >= 20 else 0.0
    
    return {
        "key": "answer_length",
        "score": score,
    }

def check_mentions_ml(run, example):
    """Check if answer mentions key ML terms."""
    answer = run.outputs.get("answer", "").lower()
    ml_terms = ["machine learning", "algorithm", "data", "model", "train"]
    
    mentions_count = sum(1 for term in ml_terms if term in answer)
    score = min(mentions_count / 2.0, 1.0)  # Normalize to 0-1
    
    return {
        "key": "ml_terminology",
        "score": score,
    }

# Run evaluation
client = Client()

results = evaluate(
    qa_system,
    data="qa_evaluation_dataset",  # Dataset name
    evaluators=[check_answer_length, check_mentions_ml],
    experiment_prefix="qa-eval",
    metadata={
        "model": "gpt-4",
        "version": "1.0"
    }
)

# View results
print("\nEvaluation Results:")
print(f"Dataset: {results.dataset_name}")
print(f"Total examples: {len(results.results)}")

for result in results.results:
    print(f"\nQuestion: {result.example.inputs['question']}")
    print(f"Answer: {result.outputs['answer'][:100]}...")
    print(f"Scores: {result.scores}")

# Results are also visible in LangSmith dashboard with:
# - Aggregate metrics
# - Per-example results
# - Comparison with other runs
```

---

## 🎯 Example 6: LLM-as-Judge Evaluation

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith.evaluation import LangChainStringEvaluator, evaluate

# System to evaluate
llm = ChatOpenAI(model="gpt-4", temperature=0)
prompt = ChatPromptTemplate.from_template("""
Answer the question in a helpful and accurate way.

Question: {question}

Answer:
""")

chain = prompt | llm | StrOutputParser()

def qa_bot(inputs: dict) -> dict:
    return {"answer": chain.invoke({"question": inputs["question"]})}

# LLM-as-Judge evaluators
qa_evaluator = LangChainStringEvaluator(
    "qa",
    config={
        "criteria": {
            "accuracy": "Is the answer factually correct?",
            "helpfulness": "Is the answer helpful to the user?",
            "completeness": "Does the answer fully address the question?"
        }
    }
)

# Run evaluation with LLM judge
results = evaluate(
    qa_bot,
    data="qa_evaluation_dataset",
    evaluators=[qa_evaluator],
    experiment_prefix="llm-judge-eval",
)

print("\nLLM-as-Judge Evaluation Complete")
print(f"Check results at: {results.experiment_url}")

# LLM judge provides:
# - Detailed feedback
# - Scores for each criterion
# - Reasoning for scores
```

---

## 🔄 Example 7: A/B Testing Prompts

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith.evaluation import evaluate

os.environ["LANGCHAIN_TRACING_V2"] = "true"

llm = ChatOpenAI(model="gpt-4", temperature=0)

# Prompt A - Direct
prompt_a = ChatPromptTemplate.from_template("""
Answer the question.

Question: {question}

Answer:
""")

# Prompt B - With instructions
prompt_b = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question clearly and concisely.
Use simple language and provide examples when helpful.

Question: {question}

Answer:
""")

# Create chains
chain_a = prompt_a | llm | StrOutputParser()
chain_b = prompt_b | llm | StrOutputParser()

# Wrapper functions
def system_a(inputs):
    return {"answer": chain_a.invoke({"question": inputs["question"]})}

def system_b(inputs):
    return {"answer": chain_b.invoke({"question": inputs["question"]})}

# Evaluation function
def check_clarity(run, example):
    """Check answer clarity (simplified)."""
    answer = run.outputs.get("answer", "")
    # Simple heuristic: longer is clearer
    score = min(len(answer) / 200.0, 1.0)
    return {"key": "clarity", "score": score}

# Evaluate both
print("Evaluating Prompt A...")
results_a = evaluate(
    system_a,
    data="qa_evaluation_dataset",
    evaluators=[check_clarity],
    experiment_prefix="prompt-a",
)

print("\nEvaluating Prompt B...")
results_b = evaluate(
    system_b,
    data="qa_evaluation_dataset",
    evaluators=[check_clarity],
    experiment_prefix="prompt-b",
)

# Compare in LangSmith dashboard
print("\nA/B Test Complete!")
print(f"Prompt A: {results_a.experiment_url}")
print(f"Prompt B: {results_b.experiment_url}")
print("\nCompare results in LangSmith to see which performs better")
```

---

## 📈 Example 8: Production Monitoring

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import random
import time

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "production-app"

# Production chain
llm = ChatOpenAI(model="gpt-4", temperature=0.7)
prompt = ChatPromptTemplate.from_template("Help the user: {request}")
chain = prompt | llm | StrOutputParser()

# Simulate production traffic
def simulate_production():
    """Simulate production requests with metadata."""
    
    requests = [
        "Explain machine learning",
        "What is Python?",
        "How do I learn coding?",
        "What's the weather?",  # Might fail - no weather tool
    ]
    
    for i, request in enumerate(requests):
        user_id = f"user_{random.randint(1, 100)}"
        session_id = f"session_{random.randint(1, 50)}"
        
        try:
            start = time.time()
            
            result = chain.invoke(
                {"request": request},
                config={
                    "tags": ["production", "customer-support"],
                    "metadata": {
                        "user_id": user_id,
                        "session_id": session_id,
                        "request_id": f"req_{i}",
                        "timestamp": time.time()
                    }
                }
            )
            
            latency = time.time() - start
            
            print(f"✅ Request {i+1}: {request[:30]}... (user: {user_id}, latency: {latency:.2f}s)")
            
        except Exception as e:
            print(f"❌ Request {i+1} failed: {str(e)}")
        
        time.sleep(1)  # Delay between requests

# Run simulation
print("Simulating production traffic...")
print("="*50)
simulate_production()

print("\n" + "="*50)
print("Check LangSmith dashboard for:")
print("- Request latency over time")
print("- Success/failure rates")
print("- Cost per request")
print("- User-specific metrics")
print("- Error patterns")
```

---

## 🐛 Example 9: Debugging Failed Runs

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "debugging-demo"

# Chain with potential failure point
llm = ChatOpenAI(model="gpt-4", temperature=0)

def risky_transform(x):
    """Function that might fail."""
    if "error" in x.lower():
        raise ValueError("Error keyword detected!")
    return x.upper()

prompt = ChatPromptTemplate.from_template("Process this: {text}")

chain = (
    {"text": RunnablePassthrough() | risky_transform}
    | prompt
    | llm
    | StrOutputParser()
)

# Test cases
test_inputs = [
    "hello world",  # Should work
    "this is an error",  # Should fail
    "another test",  # Should work
]

for i, text in enumerate(test_inputs):
    print(f"\nTest {i+1}: {text}")
    try:
        result = chain.invoke(text)
        print(f"✅ Success: {result[:50]}...")
    except Exception as e:
        print(f"❌ Failed: {str(e)}")

print("\n" + "="*50)
print("In LangSmith, failed runs show:")
print("- Exact step where failure occurred")
print("- Full error message and stack trace")
print("- Input that caused the failure")
print("- State before failure")
```

---

## 📊 Example 10: Custom Metrics and Feedback

```python
from langsmith import Client
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"

client = Client()

# After running your application, you can add feedback
def add_feedback_to_run(run_id: str, score: float, comment: str = None):
    """Add custom feedback to a run."""
    client.create_feedback(
        run_id=run_id,
        key="user_satisfaction",
        score=score,  # 0-1
        comment=comment
    )

# Example: Collect user feedback in production
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import collect_runs

llm = ChatOpenAI(model="gpt-4")

# Use callback to capture run ID
with collect_runs() as cb:
    result = llm.invoke("What is Python?")
    run_id = cb.traced_runs[0].id
    
    print(f"Response: {result.content}")
    
    # Simulate user feedback
    user_rating = 0.9  # User rates 9/10
    add_feedback_to_run(
        run_id=run_id,
        score=user_rating,
        comment="Very helpful explanation"
    )
    
    print(f"Added feedback to run {run_id}")

# View feedback in LangSmith:
# - Aggregate satisfaction scores
# - Identify low-scoring runs
# - Correlate feedback with prompt versions
```

---

## 🎨 Example 11: Playground for Prompt Iteration

```python
"""
LangSmith Playground allows you to:

1. Open any trace in the playground
2. Modify the prompt
3. Re-run with different inputs
4. Compare outputs side-by-side
5. Save successful prompts to Prompt Hub

Workflow:
1. Run your chain (creates trace)
2. Open trace in LangSmith
3. Click "Open in Playground"
4. Modify prompt
5. Test with different inputs
6. Save to Prompt Hub when satisfied

This is done via the LangSmith UI, not programmatically.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "playground-demo"

# Original prompt
prompt = ChatPromptTemplate.from_template("""
Explain {concept} to a {audience}.
""")

llm = ChatOpenAI(model="gpt-4", temperature=0.7)
chain = prompt | llm

# Run to create traces
result = chain.invoke({
    "concept": "neural networks",
    "audience": "beginner"
})

print(result.content)

print("\n" + "="*50)
print("Now in LangSmith:")
print("1. Open this trace")
print("2. Click 'Open in Playground'")
print("3. Try variations like:")
print("   - 'Explain {concept} clearly with examples'")
print("   - 'Explain {concept} to a {audience} using analogies'")
print("4. Compare outputs")
print("5. Save best prompt to Hub")
```

---

## 🔧 Example 12: Integration with CI/CD

```python
"""
Example: Running evaluations in CI/CD

This ensures your changes don't degrade performance before deployment.
"""

# test_llm_system.py
import os
import pytest
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langsmith.evaluation import evaluate

# Set environment
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ci-cd-tests"

@pytest.fixture
def qa_chain():
    """Create chain for testing."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    prompt = ChatPromptTemplate.from_template("Answer: {question}")
    return prompt | llm

def test_qa_performance(qa_chain):
    """Test QA system meets performance thresholds."""
    
    def qa_system(inputs):
        answer = qa_chain.invoke({"question": inputs["question"]})
        return {"answer": answer.content}
    
    # Simple evaluator
    def check_length(run, example):
        answer = run.outputs.get("answer", "")
        score = 1.0 if len(answer) >= 20 else 0.0
        return {"key": "min_length", "score": score}
    
    # Run evaluation
    results = evaluate(
        qa_system,
        data="qa_evaluation_dataset",
        evaluators=[check_length],
        experiment_prefix="ci-test",
    )
    
    # Check threshold
    avg_score = sum(r.scores.get("min_length", 0) for r in results.results) / len(results.results)
    
    assert avg_score >= 0.8, f"Performance below threshold: {avg_score}"
    print(f"✅ Tests passed with score: {avg_score}")

# Run with: pytest test_llm_system.py

# In CI/CD (e.g., GitHub Actions):
# - name: Run LangSmith Tests
#   run: |
#     export LANGCHAIN_API_KEY=${{ secrets.LANGCHAIN_API_KEY }}
#     pytest test_llm_system.py
```

---

## 🔥 Best Practices

### **1. Always Use Tracing in Development**
```python
# ✅ Set once at startup
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "dev-project"
```

### **2. Use Meaningful Project Names**
```python
# ❌ Bad
os.environ["LANGCHAIN_PROJECT"] = "test"

# ✅ Good
os.environ["LANGCHAIN_PROJECT"] = "customer-support-bot-dev"
```

### **3. Tag and Annotate Production Runs**
```python
# ✅ Rich metadata
chain.invoke(
    input,
    config={
        "tags": ["production", "premium-user"],
        "metadata": {
            "user_id": "123",
            "session_id": "abc",
            "version": "v2.1"
        }
    }
)
```

### **4. Create Evaluation Datasets Early**
```python
# ✅ Build datasets as you develop
# - Add edge cases
# - Include failure examples
# - Cover all use cases
```

### **5. Run Evaluations Before Deployment**
```python
# ✅ CI/CD integration
# pytest tests/
# If tests fail, block deployment
```

### **6. Monitor Key Metrics**
```python
# ✅ Track in production:
# - Latency (p50, p95, p99)
# - Cost per request
# - Error rate
# - User satisfaction (via feedback)
```

### **7. Use LLM-as-Judge for Complex Evaluation**
```python
# ✅ For subjective metrics
# - Helpfulness
# - Tone
# - Completeness
# Let LLM evaluate instead of hardcoded rules
```

---

## 📊 Key Metrics to Track

### **Development**
- Chain execution time
- Token usage per step
- Error rates
- Cost per run

### **Evaluation**
- Accuracy on test sets
- Consistency across runs
- Performance on edge cases
- Comparison between versions

### **Production**
- Request latency (p50, p95, p99)
- Throughput (requests/sec)
- Error rate
- Cost per user
- User satisfaction scores

---

## 🎯 Practical Exercise

```python
"""
Exercise: Build a Monitored RAG System

Create a RAG system with comprehensive LangSmith integration:

1. Basic Setup
   - Enable tracing
   - Set project name
   - Add tags/metadata

2. Create Evaluation Dataset
   - 10+ question-answer pairs
   - Include edge cases
   - Cover different query types

3. Implement RAG System
   - Document loading
   - Vector store
   - Retrieval + generation

4. Custom Evaluators
   - Check answer relevance
   - Verify citations
   - Measure completeness

5. A/B Test Two Approaches
   - Different retrieval strategies
   - Compare performance

6. Production Simulation
   - Add user feedback
   - Track metrics
   - Monitor errors

7. CI/CD Integration
   - Write pytest tests
   - Set performance thresholds
   - Automate evaluation

Requirements:
- Full tracing enabled
- Evaluation dataset with 10+ examples
- At least 3 custom evaluators
- A/B test results comparison
- Simulated production traffic with feedback
"""

import os
from langsmith import Client
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# TODO: Setup LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "rag-monitoring-exercise"

# TODO: Create evaluation dataset
# TODO: Implement RAG system with tracing
# TODO: Create custom evaluators
# TODO: Run A/B tests
# TODO: Simulate production with feedback
# TODO: Write CI/CD tests
```

---

## ✅ Key Takeaways

1. **LangSmith provides observability** - see inside your LLM apps
2. **Tracing is automatic** - just set environment variables
3. **Tags and metadata** - organize and filter traces
4. **Datasets enable evaluation** - test against ground truth
5. **Custom evaluators** - measure what matters to you
6. **LLM-as-judge** - automate subjective evaluation
7. **A/B testing** - compare prompts and approaches scientifically
8. **Production monitoring** - track performance and costs
9. **Debugging** - find failures quickly with detailed traces
10. **CI/CD integration** - prevent regressions before deployment

---

## 📝 Understanding Check

1. How do you enable LangSmith tracing?
2. What's the difference between tags and metadata?
3. When should you use LLM-as-judge evaluation?
4. How can LangSmith help in production?

**Ready for Section 14 on API Integration?** We'll explore integrating with different LLM providers (Anthropic, Ollama, Azure, Google)! 🌐

Or would you like to:
- Practice more with LangSmith?
- See the exercise solution?
- Deep dive into evaluation strategies?