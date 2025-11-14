# Section 11: Advanced RAG Techniques 🚀

Advanced RAG techniques improve retrieval quality, handle complex queries, and enhance answer accuracy beyond basic RAG.

---

## 🎯 Why Advanced RAG?

**Basic RAG Limitations:**
```python
# Problem 1: Poor query formulation
User: "Tell me about that new AI thing"
→ Retrieves irrelevant docs (vague query)

# Problem 2: Missing relevant docs
User: "What are the benefits?"
→ No context about what "benefits" refers to

# Problem 3: Retrieved docs not optimal
→ Gets 5 docs, but best answer is in #6

# Problem 4: Irrelevant chunks included
→ Some retrieved chunks don't help answer
```

**Advanced RAG Solutions:**
- Query transformation
- Hypothetical Document Embeddings (HyDE)
- Reranking
- Query decomposition
- RAG Fusion
- Corrective RAG (CRAG)

---

## 📋 Advanced RAG Techniques Overview

| Technique | Problem Solved | Complexity |
|-----------|---------------|------------|
| Query Rewriting | Vague/poor queries | Low |
| HyDE | No direct keyword match | Medium |
| Multi-Query | Single perspective limits results | Medium |
| Query Decomposition | Complex multi-part questions | Medium |
| Reranking | Retrieved docs not optimal order | Medium |
| Contextual Compression | Too much irrelevant info | Medium |
| RAG Fusion | Limited retrieval diversity | High |
| CRAG | Incorrect/irrelevant retrievals | High |

---

## 💻 Example 1: Query Rewriting

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough

# Create knowledge base
documents = [
    Document(page_content="Python is a high-level programming language created in 1991."),
    Document(page_content="Machine learning enables computers to learn from data."),
    Document(page_content="Neural networks are inspired by biological neurons."),
    Document(page_content="Deep learning uses multi-layer neural networks."),
]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# Query rewriter
llm = ChatOpenAI(model="gpt-4", temperature=0)

query_rewriter_prompt = ChatPromptTemplate.from_template("""
Given a vague or poorly formed question, rewrite it to be more specific and 
searchable while preserving the original intent.

Original question: {question}

Rewritten question (only output the question, nothing else):
""")

query_rewriter = query_rewriter_prompt | llm | StrOutputParser()

# RAG chain with query rewriting
rag_prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below.

Context: {context}

Question: {question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain: Rewrite query → Retrieve → Answer
rag_with_rewriting = (
    {
        "context": (
            {"question": RunnablePassthrough()} 
            | query_rewriter  # Rewrite first
            | vectorstore.as_retriever() 
            | format_docs
        ),
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Test with vague questions
vague_questions = [
    "that language",
    "the brain thing for computers",
    "what uses multiple layers"
]

for question in vague_questions:
    print(f"\nOriginal Q: {question}")
    
    # Show rewritten query
    rewritten = query_rewriter.invoke({"question": question})
    print(f"Rewritten Q: {rewritten}")
    
    # Get answer
    answer = rag_with_rewriting.invoke(question)
    print(f"Answer: {answer}")
    print("-" * 50)

# Output shows improved retrieval with rewritten queries
```

---

## 🎨 Example 2: HyDE (Hypothetical Document Embeddings)

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Knowledge base
documents = [
    Document(page_content="Photosynthesis is the process by which plants convert sunlight into energy. Chlorophyll in leaves absorbs light, and carbon dioxide combines with water to produce glucose and oxygen."),
    Document(page_content="The mitochondria is the powerhouse of the cell. It produces ATP through cellular respiration, converting nutrients into usable energy."),
    Document(page_content="DNA contains genetic information. It consists of four nucleotides: adenine, thymine, guanine, and cytosine arranged in a double helix structure."),
]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

llm = ChatOpenAI(model="gpt-4", temperature=0)

# HyDE: Generate a hypothetical answer, then search for similar documents
hyde_prompt = ChatPromptTemplate.from_template("""
Please write a passage that would answer the following question. 
The passage should be detailed and factual.

Question: {question}

Hypothetical passage:
""")

hyde_generator = hyde_prompt | llm | StrOutputParser()

# RAG chain with HyDE
rag_prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context.

Context: {context}

Question: {question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Regular RAG
regular_rag = (
    {
        "context": vectorstore.as_retriever() | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# HyDE RAG: Generate hypothetical doc → Search with it → Answer
hyde_rag = (
    {
        "context": (
            {"question": RunnablePassthrough()}
            | hyde_generator  # Generate hypothetical answer
            | vectorstore.as_retriever()  # Search with hypothetical answer
            | format_docs
        ),
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Test
question = "How do plants make energy?"

print("Regular RAG:")
regular_answer = regular_rag.invoke(question)
print(regular_answer)

print("\n" + "="*50)

print("\nHyDE RAG:")
# Show hypothetical document
hypothetical = hyde_generator.invoke({"question": question})
print(f"Hypothetical doc: {hypothetical[:200]}...")

hyde_answer = hyde_rag.invoke(question)
print(f"\nAnswer: {hyde_answer}")

# HyDE often retrieves better documents because the hypothetical answer
# is semantically closer to the actual answer than the question
```

---

## 🔍 Example 3: Multi-Query Retrieval

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Knowledge base
documents = [
    Document(page_content="Supervised learning uses labeled data to train models."),
    Document(page_content="Unsupervised learning finds patterns in unlabeled data."),
    Document(page_content="Reinforcement learning learns through trial and error with rewards."),
    Document(page_content="Deep learning uses neural networks with multiple layers."),
    Document(page_content="Transfer learning reuses pre-trained models for new tasks."),
]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

llm = ChatOpenAI(model="gpt-4", temperature=0)

# Generate multiple query variations
multi_query_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant. Generate 3 different versions of the given question 
to retrieve relevant documents. Provide these alternative questions separated by newlines.

Original question: {question}

Alternative questions:
""")

multi_query_generator = multi_query_prompt | llm | StrOutputParser()

# Multi-query retrieval
def multi_query_retrieve(question: str, k: int = 2):
    # Generate alternative questions
    alternatives = multi_query_generator.invoke({"question": question})
    all_questions = [question] + [q.strip() for q in alternatives.split("\n") if q.strip()]
    
    print(f"Generated {len(all_questions)} queries:")
    for i, q in enumerate(all_questions, 1):
        print(f"{i}. {q}")
    
    # Retrieve documents for each query
    all_docs = []
    seen_content = set()
    
    for query in all_questions:
        docs = vectorstore.similarity_search(query, k=k)
        for doc in docs:
            if doc.page_content not in seen_content:
                all_docs.append(doc)
                seen_content.add(doc.page_content)
    
    print(f"\nRetrieved {len(all_docs)} unique documents")
    return all_docs

# RAG with multi-query
rag_prompt = ChatPromptTemplate.from_template("""
Answer based on the context.

Context: {context}

Question: {question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Test
question = "What are ways computers can learn?"

docs = multi_query_retrieve(question, k=2)
context = format_docs(docs)

answer = (rag_prompt | llm | StrOutputParser()).invoke({
    "context": context,
    "question": question
})

print(f"\nFinal Answer: {answer}")

# Multi-query retrieves more diverse, comprehensive results
```

---

## 🧩 Example 4: Query Decomposition

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Knowledge base
documents = [
    Document(page_content="Python was created by Guido van Rossum in 1991."),
    Document(page_content="Python is popular for data science due to libraries like NumPy and Pandas."),
    Document(page_content="JavaScript was created by Brendan Eich in 1995."),
    Document(page_content="JavaScript is mainly used for web development."),
    Document(page_content="Both Python and JavaScript support multiple programming paradigms."),
]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

llm = ChatOpenAI(model="gpt-4", temperature=0)

# Decompose complex question into sub-questions
decomposition_prompt = ChatPromptTemplate.from_template("""
Break down the following complex question into simpler sub-questions. 
Each sub-question should be independently answerable.
Provide the sub-questions as a numbered list.

Complex question: {question}

Sub-questions:
""")

decomposer = decomposition_prompt | llm | StrOutputParser()

# Answer sub-questions and synthesize
def decomposed_rag(complex_question: str):
    # Decompose
    sub_questions_text = decomposer.invoke({"question": complex_question})
    sub_questions = [
        q.split(". ", 1)[1] if ". " in q else q 
        for q in sub_questions_text.split("\n") 
        if q.strip()
    ]
    
    print(f"Decomposed into {len(sub_questions)} sub-questions:")
    for i, sq in enumerate(sub_questions, 1):
        print(f"{i}. {sq}")
    
    # Answer each sub-question
    sub_answers = []
    for sq in sub_questions:
        docs = vectorstore.similarity_search(sq, k=2)
        context = "\n".join([doc.page_content for doc in docs])
        
        answer_prompt = ChatPromptTemplate.from_template("""
        Context: {context}
        Question: {question}
        Answer:
        """)
        
        answer = (answer_prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "question": sq
        })
        
        sub_answers.append(f"Q: {sq}\nA: {answer}")
        print(f"\nSub-answer {len(sub_answers)}: {answer}")
    
    # Synthesize final answer
    synthesis_prompt = ChatPromptTemplate.from_template("""
    Given the following sub-questions and their answers, provide a comprehensive 
    answer to the original question.

    Original question: {original_question}

    Sub-questions and answers:
    {sub_answers}

    Comprehensive answer:
    """)
    
    final_answer = (synthesis_prompt | llm | StrOutputParser()).invoke({
        "original_question": complex_question,
        "sub_answers": "\n\n".join(sub_answers)
    })
    
    return final_answer

# Test with complex question
complex_q = "Compare Python and JavaScript in terms of their history and primary use cases."

print("="*50)
final = decomposed_rag(complex_q)
print("\n" + "="*50)
print(f"Final Answer:\n{final}")
```

---

## 🏆 Example 5: Reranking Retrieved Documents

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Knowledge base
documents = [
    Document(page_content="Machine learning models can overfit when trained on too little data.", metadata={"id": 1}),
    Document(page_content="Regularization techniques like L1 and L2 help prevent overfitting.", metadata={"id": 2}),
    Document(page_content="Cross-validation splits data to evaluate model performance.", metadata={"id": 3}),
    Document(page_content="Overfitting occurs when a model learns noise instead of patterns.", metadata={"id": 4}),
    Document(page_content="Early stopping prevents overfitting by halting training at optimal point.", metadata={"id": 5}),
]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

llm = ChatOpenAI(model="gpt-4", temperature=0)

# Reranker: Score documents by relevance
reranking_prompt = ChatPromptTemplate.from_template("""
Given a question and a document, rate how relevant the document is to answering 
the question on a scale of 1-10.

Question: {question}

Document: {document}

Relevance score (1-10):
""")

def rerank_documents(question: str, docs: list, top_k: int = 3):
    """Rerank documents by relevance score."""
    scored_docs = []
    
    for doc in docs:
        score_text = (reranking_prompt | llm | StrOutputParser()).invoke({
            "question": question,
            "document": doc.page_content
        })
        
        # Extract numeric score
        try:
            score = float(score_text.strip())
        except:
            score = 5.0  # Default score
        
        scored_docs.append((score, doc))
        print(f"Doc {doc.metadata['id']}: Score {score}")
    
    # Sort by score (descending)
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Return top_k
    return [doc for _, doc in scored_docs[:top_k]]

# RAG with reranking
question = "How can I prevent overfitting in my model?"

# Initial retrieval (get more than needed)
initial_docs = vectorstore.similarity_search(question, k=5)

print("Initial retrieval (all 5 docs)")
print("="*50)

# Rerank
print("\nReranking...")
reranked_docs = rerank_documents(question, initial_docs, top_k=3)

print(f"\nTop 3 after reranking:")
for doc in reranked_docs:
    print(f"- Doc {doc.metadata['id']}: {doc.page_content}")

# Generate answer with reranked docs
rag_prompt = ChatPromptTemplate.from_template("""
Context: {context}

Question: {question}

Answer:
""")

context = "\n\n".join([doc.page_content for doc in reranked_docs])
answer = (rag_prompt | llm | StrOutputParser()).invoke({
    "context": context,
    "question": question
})

print(f"\nFinal Answer: {answer}")
```

---

## 🗜️ Example 6: Contextual Compression

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Knowledge base with verbose documents
documents = [
    Document(page_content="""
    Machine learning is a broad field with many applications. It includes supervised learning,
    unsupervised learning, and reinforcement learning. Supervised learning, in particular,
    uses labeled data where each example has an input and corresponding output. This allows
    the model to learn the mapping between inputs and outputs. Common supervised learning
    algorithms include linear regression, logistic regression, and neural networks. The field
    has grown significantly in recent years due to increased computing power and data availability.
    """),
    Document(page_content="""
    Neural networks are computational models inspired by biological neurons in the brain.
    They consist of layers of interconnected nodes. Each connection has a weight that is
    adjusted during training. The first successful neural network was the perceptron in 1958.
    Modern neural networks can have millions of parameters. Deep learning refers to neural
    networks with many layers. These deep networks have achieved breakthrough performance
    in computer vision, natural language processing, and speech recognition.
    """),
]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Compressor - extracts only relevant parts
llm = ChatOpenAI(model="gpt-4", temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

# Compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Compare regular vs compressed retrieval
question = "What is supervised learning?"

print("Regular Retrieval:")
print("="*50)
regular_docs = base_retriever.invoke(question)
for i, doc in enumerate(regular_docs, 1):
    print(f"\nDoc {i} (length: {len(doc.page_content)} chars):")
    print(doc.page_content[:200] + "...")

print("\n" + "="*50)
print("\nCompressed Retrieval:")
print("="*50)
compressed_docs = compression_retriever.invoke(question)
for i, doc in enumerate(compressed_docs, 1):
    print(f"\nDoc {i} (length: {len(doc.page_content)} chars):")
    print(doc.page_content)

# RAG with compressed context
rag_prompt = ChatPromptTemplate.from_template("""
Context: {context}

Question: {question}

Answer:
""")

context = "\n\n".join([doc.page_content for doc in compressed_docs])
answer = (rag_prompt | llm | StrOutputParser()).invoke({
    "context": context,
    "question": question
})

print("\n" + "="*50)
print(f"Final Answer: {answer}")

# Compressed retrieval gives focused, relevant excerpts only
```

---

## 🔀 Example 7: RAG Fusion

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from collections import defaultdict

# Knowledge base
documents = [
    Document(page_content="Python supports object-oriented programming.", metadata={"id": 1}),
    Document(page_content="Python is dynamically typed.", metadata={"id": 2}),
    Document(page_content="Python has extensive libraries for data science.", metadata={"id": 3}),
    Document(page_content="Python uses indentation for code blocks.", metadata={"id": 4}),
    Document(page_content="Python is interpreted, not compiled.", metadata={"id": 5}),
]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

llm = ChatOpenAI(model="gpt-4", temperature=0)

# Generate multiple query variations
multi_query_prompt = ChatPromptTemplate.from_template("""
Generate 3 different phrasings of this question:

Original: {question}

Variations (one per line):
""")

def rag_fusion(question: str, k: int = 3):
    """RAG Fusion: Generate multiple queries, retrieve for each, fuse results."""
    
    # Generate query variations
    variations_text = (multi_query_prompt | llm | StrOutputParser()).invoke({
        "question": question
    })
    
    queries = [question] + [
        q.strip() for q in variations_text.split("\n") if q.strip()
    ]
    
    print(f"Generated {len(queries)} query variations:")
    for i, q in enumerate(queries, 1):
        print(f"{i}. {q}")
    
    # Retrieve for each query and track rankings
    doc_scores = defaultdict(float)
    
    for query in queries:
        # Get docs with scores
        results = vectorstore.similarity_search_with_score(query, k=k)
        
        # Reciprocal Rank Fusion (RRF)
        for rank, (doc, score) in enumerate(results, 1):
            doc_id = doc.metadata["id"]
            # RRF formula: 1 / (rank + k), where k=60 is common
            doc_scores[doc_id] += 1.0 / (rank + 60)
    
    # Sort documents by fused score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nFused scores:")
    for doc_id, score in sorted_docs[:k]:
        print(f"Doc {doc_id}: {score:.4f}")
    
    # Retrieve top documents
    top_doc_ids = [doc_id for doc_id, _ in sorted_docs[:k]]
    top_docs = [doc for doc in documents if doc.metadata["id"] in top_doc_ids]
    
    return top_docs

# Test RAG Fusion
question = "What are Python's features?"

print("="*50)
fused_docs = rag_fusion(question, k=3)

print(f"\nTop 3 documents after fusion:")
for doc in fused_docs:
    print(f"- Doc {doc.metadata['id']}: {doc.page_content}")

# RAG Fusion combines rankings from multiple query variations
# for more robust retrieval
```

---

## 🔧 Example 8: Corrective RAG (CRAG)

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from enum import Enum

class RelevanceDecision(str, Enum):
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    AMBIGUOUS = "ambiguous"

# Knowledge base
documents = [
    Document(page_content="Paris is the capital of France.", metadata={"id": 1}),
    Document(page_content="The Eiffel Tower is located in Paris.", metadata={"id": 2}),
    Document(page_content="French cuisine is world-renowned.", metadata={"id": 3}),
    Document(page_content="Machine learning models learn from data.", metadata={"id": 4}),
    Document(page_content="Python is a popular programming language.", metadata={"id": 5}),
]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

llm = ChatOpenAI(model="gpt-4", temperature=0)

# Relevance grader
relevance_prompt = ChatPromptTemplate.from_template("""
Determine if the document is relevant to the question.

Question: {question}

Document: {document}

Is this document relevant? Answer only with: relevant, irrelevant, or ambiguous
""")

def grade_relevance(question: str, doc: Document) -> RelevanceDecision:
    """Grade document relevance."""
    result = (relevance_prompt | llm | StrOutputParser()).invoke({
        "question": question,
        "document": doc.page_content
    })
    
    result = result.strip().lower()
    if "relevant" in result and "irrelevant" not in result:
        return RelevanceDecision.RELEVANT
    elif "irrelevant" in result:
        return RelevanceDecision.IRRELEVANT
    else:
        return RelevanceDecision.AMBIGUOUS

# Web search fallback (simulated)
def web_search(query: str) -> str:
    """Fallback to web search if retrieval fails."""
    print(f"📡 Falling back to web search for: {query}")
    # Simulate web search
    return "According to web sources, Paris is indeed the capital of France, known for its art, fashion, and culture."

def corrective_rag(question: str):
    """CRAG: Evaluate retrieved docs, correct if needed."""
    
    # Initial retrieval
    docs = vectorstore.similarity_search(question, k=3)
    
    print(f"Retrieved {len(docs)} documents")
    print("="*50)
    
    # Grade each document
    relevant_docs = []
    ambiguous_docs = []
    
    for doc in docs:
        decision = grade_relevance(question, doc)
        print(f"Doc {doc.metadata['id']}: {decision.value}")
        
        if decision == RelevanceDecision.RELEVANT:
            relevant_docs.append(doc)
        elif decision == RelevanceDecision.AMBIGUOUS:
            ambiguous_docs.append(doc)
    
    # Decision logic
    if len(relevant_docs) >= 2:
        # Good retrieval - use docs
        print("\n✅ Sufficient relevant documents found")
        context_docs = relevant_docs
        
    elif len(relevant_docs) == 1 or ambiguous_docs:
        # Partial retrieval - use relevant + rewrite query
        print("\n⚠️ Limited relevant documents - using what we have")
        context_docs = relevant_docs + ambiguous_docs
        
    else:
        # Poor retrieval - fall back to web search
        print("\n❌ No relevant documents - falling back to web search")
        web_content = web_search(question)
        context_docs = [Document(page_content=web_content)]
    
    # Generate answer
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    answer_prompt = ChatPromptTemplate.from_template("""
    Context: {context}
    
    Question: {question}
    
    Answer:
    """)
    
    answer = (answer_prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question
    })
    
    return answer

# Test CRAG
questions = [
    "What is the capital of France?",  # Should find relevant docs
    "What is quantum computing?",  # Should fall back to web search
]

for q in questions:
    print("\n" + "="*50)
    print(f"Question: {q}")
    print("="*50)
    answer = corrective_rag(q)
    print(f"\nAnswer: {answer}\n")
```

---

## 📊 Example 9: Self-RAG (Self-Reflective RAG)

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Knowledge base
documents = [
    Document(page_content="The Earth orbits the Sun once per year."),
    Document(page_content="The Moon orbits the Earth approximately every 27 days."),
    Document(page_content="The solar system contains 8 planets."),
    Document(page_content="Jupiter is the largest planet in our solar system."),
]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

llm = ChatOpenAI(model="gpt-4", temperature=0)

def self_rag(question: str, max_iterations: int = 3):
    """Self-RAG: Generate answer, reflect, and iterate."""
    
    # Initial retrieval
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    for iteration in range(max_iterations):
        print(f"\n{'='*50}")
        print(f"Iteration {iteration + 1}")
        print('='*50)
        
        # Generate answer
        answer_prompt = ChatPromptTemplate.from_template("""
        Context: {context}
        
        Question: {question}
        
        Answer:
        """)
        
        answer = (answer_prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "question": question
        })
        
        print(f"Generated answer: {answer}")
        
        # Self-reflection: Is answer good?
        reflection_prompt = ChatPromptTemplate.from_template("""
        Evaluate this answer to the question:
        
        Question: {question}
        Answer: {answer}
        Context used: {context}
        
        Is this answer:
        1. Accurate based on the context?
        2. Complete?
        3. Well-supported?
        
        Respond with:
        - "GOOD" if the answer is satisfactory
        - "NEEDS_MORE_CONTEXT" if more information is needed
        - "NEEDS_REVISION" if the answer should be rephrased
        
        Then provide a brief explanation.
        """)
        
        reflection = (reflection_prompt | llm | StrOutputParser()).invoke({
            "question": question,
            "answer": answer,
            "context": context
        })
        
        print(f"\nReflection: {reflection}")
        
        if "GOOD" in reflection.upper():
            print("\n✅ Answer is satisfactory")
            return answer
        
        elif "NEEDS_MORE_CONTEXT" in reflection.upper():
            print("\n🔍 Retrieving more context...")
            # Retrieve more documents
            more_docs = vectorstore.similarity_search(question, k=5)
            context = "\n".join([doc.page_content for doc in more_docs])
        
        elif "NEEDS_REVISION" in reflection.upper():
            print("\n✏️ Revising answer...")
            # Keep context, regenerate with explicit instruction
            revision_prompt = ChatPromptTemplate.from_template("""
            The previous answer needs revision: {previous_answer}
            
            Context: {context}
            Question: {question}
            
            Generate an improved answer:
            """)
            
            answer = (revision_prompt | llm | StrOutputParser()).invoke({
                "previous_answer": answer,
                "context": context,
                "question": question
            })
    
    return answer

# Test Self-RAG
question = "How long does it take for the Earth to orbit the Sun?"

final_answer = self_rag(question, max_iterations=3)

print("\n" + "="*50)
print(f"Final Answer: {final_answer}")
```

---

## 🔥 Best Practices for Advanced RAG

### **1. Choose Techniques Based on Use Case**
```python
# Vague queries → Query rewriting
# No keyword matches → HyDE
# Complex questions → Query decomposition
# Poor initial results → Reranking or CRAG
# Need diversity → Multi-query or RAG Fusion
```

### **2. Combine Multiple Techniques**
```python
# Example: Query rewriting + Reranking + Compression
# 1. Rewrite query for clarity
# 2. Retrieve more documents than needed
# 3. Rerank by relevance
# 4. Compress to extract key info
# 5. Generate answer
```

### **3. Monitor and Iterate**
```python
# ✅ Track metrics
# - Retrieval precision/recall
# - Answer accuracy
# - Latency
# - Cost

# ✅ A/B test techniques
# Compare basic RAG vs advanced techniques
```

### **4. Balance Complexity vs Performance**
```python
# Simple queries → Basic RAG (fast, cheap)
# Complex queries → Advanced techniques (slower, better)

# Use routing:
# If query_complexity == "simple":
#     use_basic_rag()
# else:
#     use_advanced_rag()
```

---

## 📊 Technique Comparison

| Technique | Latency | Cost | Improvement | Best For |
|-----------|---------|------|-------------|----------|
| Query Rewriting | Low | Low | Moderate | Vague queries |
| HyDE | Medium | Medium | High | Semantic gaps |
| Multi-Query | Medium | Medium | Moderate-High | Diversity |
| Decomposition | High | High | High | Complex questions |
| Reranking | Medium | Medium | Moderate-High | Optimizing order |
| Compression | Medium | Medium | Moderate | Long docs |
| RAG Fusion | High | High | High | Comprehensive retrieval |
| CRAG | High | High | Very High | Quality control |

---

## 🎯 Practical Exercise

```python
"""
Exercise: Build a Hybrid Advanced RAG System

Create a system that intelligently chooses techniques based on query characteristics:

1. Classify query type (simple, complex, vague)
2. Apply appropriate technique(s):
   - Simple → Basic RAG
   - Vague → Query rewriting
   - Complex → Decomposition
   - Needs verification → CRAG
3. Always apply reranking for top results
4. Return answer with metadata (techniques used, confidence)

Test with diverse questions:
- "Python" (vague)
- "Compare supervised and unsupervised learning" (complex)
- "What is the capital of France?" (simple)
- "What are quantum computers?" (might need web fallback)
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from enum import Enum

class QueryType(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"
    VAGUE = "vague"

# TODO: Build knowledge base
# TODO: Implement query classifier
# TODO: Implement each RAG variant
# TODO: Implement routing logic
# TODO: Test with diverse queries

def classify_query(question: str) -> QueryType:
    """Classify query type."""
    pass

def hybrid_rag(question: str):
    """Route to appropriate RAG technique."""
    query_type = classify_query(question)
    
    if query_type == QueryType.SIMPLE:
        # Use basic RAG
        pass
    elif query_type == QueryType.VAGUE:
        # Use query rewriting
        pass
    elif query_type == QueryType.COMPLEX:
        # Use decomposition
        pass
    
    # Always rerank
    # Return answer with metadata

# Test
questions = [
    "Python",
    "Compare Python and Java",
    "What is the capital of France?",
]

for q in questions:
    result = hybrid_rag(q)
    print(f"Q: {q}")
    print(f"Type: {result['query_type']}")
    print(f"Techniques: {result['techniques_used']}")
    print(f"Answer: {result['answer']}\n")
```

---

## ✅ Key Takeaways

1. **Query rewriting** - clarifies vague questions
2. **HyDE** - bridges semantic gaps with hypothetical documents
3. **Multi-query** - retrieves diverse results with variations
4. **Query decomposition** - breaks complex questions into sub-questions
5. **Reranking** - optimizes retrieval order by relevance
6. **Contextual compression** - extracts only relevant excerpts
7. **RAG Fusion** - combines rankings from multiple queries
8. **CRAG** - evaluates and corrects poor retrievals
9. **Self-RAG** - iteratively improves answers through reflection
10. **Combine techniques** - use multiple for best results

---

## 📝 Understanding Check

1. When would you use HyDE over regular RAG?
2. What's the difference between multi-query and query decomposition?
3. Why is reranking important?
4. When should you use CRAG?

**Ready for Section 12 on LangGraph?** This is where we build stateful, multi-step agent workflows with graphs! 🕸️

Or would you like to:
- See the exercise solution?
- Practice more with advanced RAG?
- Deep dive into specific techniques?