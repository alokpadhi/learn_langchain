# Section 7: Vector Stores and Embeddings 🔍

Vector Stores enable semantic search by converting text into numerical vectors (embeddings) and finding similar content.

---

## 🎯 What are Embeddings and Vector Stores?

### **The Concept**

**Embeddings** = Converting text into numerical vectors that capture semantic meaning

```python
# Text
"Machine learning is fascinating"

# Embedding (simplified - actually 1536 dimensions for OpenAI)
[0.23, -0.45, 0.67, 0.12, -0.89, ...]

# Similar text has similar vectors
"AI and ML are interesting"  →  [0.25, -0.43, 0.65, 0.15, -0.87, ...]

# Different text has different vectors
"I love pizza"  →  [-0.12, 0.78, -0.34, 0.91, 0.45, ...]
```

**Vector Store** = Database optimized for storing and searching embeddings

---

## 🤔 Why Vector Stores?

**Problem with traditional search:**
```python
query = "What is ML?"
documents = ["Machine learning tutorial", "Python basics", "Deep learning guide"]

# Keyword search misses semantic similarity
# "ML" doesn't match "Machine learning" or "Deep learning"
```

**Solution with vector search:**
```python
# Convert query to embedding
query_embedding = embeddings.embed_query("What is ML?")

# Find documents with similar embeddings (semantic search!)
# ✅ Finds "Machine learning tutorial" (semantically similar)
# ✅ Finds "Deep learning guide" (related concept)
# ❌ Skips "Python basics" (not related)
```

---

## 📋 Popular Vector Stores

| Vector Store | Type | Best For | Pros | Cons |
|-------------|------|----------|------|------|
| FAISS | Local | Development, small scale | Fast, free, no setup | Not distributed |
| Chroma | Local/Cloud | Development | Easy to use, persistent | Limited scale |
| Pinecone | Cloud | Production | Fully managed, scalable | Paid service |
| Weaviate | Self-hosted | Production | Feature-rich, open source | Complex setup |
| Qdrant | Local/Cloud | Production | Fast, flexible | Newer, smaller community |
| Milvus | Self-hosted | Large scale | Highly scalable | Complex |

---

## 🧮 Embedding Models

| Provider | Model | Dimensions | Cost |
|----------|-------|------------|------|
| OpenAI | text-embedding-3-small | 1536 | $0.02/1M tokens |
| OpenAI | text-embedding-3-large | 3072 | $0.13/1M tokens |
| HuggingFace | all-MiniLM-L6-v2 | 384 | Free (local) |
| Cohere | embed-english-v3.0 | 1024 | $0.10/1M tokens |
| Google | textembedding-gecko | 768 | Varies |

---

## 💻 Example 1: Basic Embeddings

```python
from langchain_openai import OpenAIEmbeddings

# Initialize embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Embed a single text
text = "Machine learning is a subset of AI"
vector = embeddings.embed_query(text)

print(f"Embedding dimensions: {len(vector)}")
print(f"First 10 values: {vector[:10]}")

# Output:
# Embedding dimensions: 1536
# First 10 values: [0.0234, -0.0456, 0.0789, ...]

# Embed multiple documents
documents = [
    "Python is a programming language",
    "Machine learning uses Python",
    "I love pizza"
]

vectors = embeddings.embed_documents(documents)
print(f"\nEmbedded {len(vectors)} documents")
print(f"Each has {len(vectors[0])} dimensions")
```

---

## 🗄️ Example 2: FAISS Vector Store (Local, Fast)

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Create documents
documents = [
    Document(page_content="Machine learning is a subset of AI", metadata={"source": "intro.txt"}),
    Document(page_content="Deep learning uses neural networks", metadata={"source": "dl.txt"}),
    Document(page_content="Python is popular for data science", metadata={"source": "python.txt"}),
    Document(page_content="Natural language processing handles text", metadata={"source": "nlp.txt"}),
    Document(page_content="Computer vision processes images", metadata={"source": "cv.txt"}),
]

# Create vector store
vectorstore = FAISS.from_documents(documents, embeddings)

print("Vector store created with", vectorstore.index.ntotal, "vectors")

# Search for similar documents
query = "What is deep learning?"
results = vectorstore.similarity_search(query, k=3)

print(f"\nTop {len(results)} results for: '{query}'")
for i, doc in enumerate(results):
    print(f"\n{i+1}. {doc.page_content}")
    print(f"   Source: {doc.metadata['source']}")

# Output:
# 1. Deep learning uses neural networks
#    Source: dl.txt
# 2. Machine learning is a subset of AI
#    Source: intro.txt
# 3. Natural language processing handles text
#    Source: nlp.txt
```

---

## 📊 Example 3: Similarity Search with Scores

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings()

documents = [
    Document(page_content="Python is a programming language"),
    Document(page_content="Java is used for enterprise applications"),
    Document(page_content="JavaScript runs in browsers"),
    Document(page_content="Python is great for machine learning"),
    Document(page_content="I love eating pizza"),
]

vectorstore = FAISS.from_documents(documents, embeddings)

# Search with similarity scores
query = "What's good for ML?"
results = vectorstore.similarity_search_with_score(query, k=3)

print(f"Query: '{query}'\n")
for doc, score in results:
    print(f"Score: {score:.4f}")
    print(f"Content: {doc.page_content}")
    print()

# Output:
# Score: 0.1234  (lower = more similar in FAISS)
# Content: Python is great for machine learning
#
# Score: 0.2345
# Content: Python is a programming language
#
# Score: 0.4567
# Content: JavaScript runs in browsers
```

---

## 💾 Example 4: Saving and Loading Vector Stores

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings()

# Create and populate vector store
documents = [
    Document(page_content="LangChain is a framework for LLMs"),
    Document(page_content="Vector stores enable semantic search"),
    Document(page_content="FAISS is a fast similarity search library"),
]

vectorstore = FAISS.from_documents(documents, embeddings)

# Save to disk
vectorstore.save_local("my_vectorstore")
print("Vector store saved!")

# Later... load from disk
loaded_vectorstore = FAISS.load_local(
    "my_vectorstore", 
    embeddings,
    allow_dangerous_deserialization=True  # Required for FAISS
)

print("Vector store loaded!")

# Use loaded vector store
results = loaded_vectorstore.similarity_search("What is LangChain?", k=2)
for doc in results:
    print(doc.page_content)
```

---

## 🎨 Example 5: Chroma Vector Store (Persistent)

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings()

# Create documents
documents = [
    Document(page_content="Chroma is an open-source vector database", metadata={"id": "1"}),
    Document(page_content="Vector databases store embeddings", metadata={"id": "2"}),
    Document(page_content="Embeddings capture semantic meaning", metadata={"id": "3"}),
]

# Create persistent Chroma store
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"  # Persists to disk
)

print("Chroma vector store created")

# Search
query = "What is Chroma?"
results = vectorstore.similarity_search(query, k=2)

for doc in results:
    print(f"ID: {doc.metadata['id']}")
    print(f"Content: {doc.page_content}\n")

# Later sessions - automatically loads from disk
vectorstore2 = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

print(f"Loaded vector store with {vectorstore2._collection.count()} documents")
```

---

## ➕ Example 6: Adding and Deleting Documents

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings()

# Create initial vector store
documents = [
    Document(page_content="Initial document 1", metadata={"id": "doc1"}),
    Document(page_content="Initial document 2", metadata={"id": "doc2"}),
]

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db2"
)

print(f"Initial count: {vectorstore._collection.count()}")

# Add more documents
new_docs = [
    Document(page_content="New document 3", metadata={"id": "doc3"}),
    Document(page_content="New document 4", metadata={"id": "doc4"}),
]

ids = vectorstore.add_documents(new_docs)
print(f"Added documents with IDs: {ids}")
print(f"New count: {vectorstore._collection.count()}")

# Delete documents
vectorstore.delete(ids=[ids[0]])  # Delete first new document
print(f"After deletion: {vectorstore._collection.count()}")

# Update documents (delete + add)
updated_doc = Document(page_content="Updated document 3", metadata={"id": "doc3"})
vectorstore.add_documents([updated_doc])
```

---

## 🔍 Example 7: Advanced Filtering

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings()

# Create documents with rich metadata
documents = [
    Document(
        page_content="Python tutorial for beginners",
        metadata={"language": "python", "level": "beginner", "year": 2024}
    ),
    Document(
        page_content="Advanced Python techniques",
        metadata={"language": "python", "level": "advanced", "year": 2024}
    ),
    Document(
        page_content="JavaScript basics",
        metadata={"language": "javascript", "level": "beginner", "year": 2023}
    ),
    Document(
        page_content="Python for data science",
        metadata={"language": "python", "level": "intermediate", "year": 2024}
    ),
]

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_filtered"
)

# Search with metadata filter
query = "programming tutorial"

# Filter: Only Python documents from 2024
results = vectorstore.similarity_search(
    query,
    k=3,
    filter={"language": "python", "year": 2024}
)

print("Filtered results (Python, 2024):")
for doc in results:
    print(f"- {doc.page_content}")
    print(f"  Metadata: {doc.metadata}\n")

# Output: Only Python documents from 2024
```

---

## 🎯 Example 8: MMR (Maximum Marginal Relevance) Search

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings()

# Documents with some redundancy
documents = [
    Document(page_content="Python is a programming language"),
    Document(page_content="Python is used for programming"),  # Similar to above
    Document(page_content="Python is great for scripting"),    # Similar to above
    Document(page_content="Machine learning uses Python"),
    Document(page_content="JavaScript is for web development"),
]

vectorstore = FAISS.from_documents(documents, embeddings)

query = "Tell me about Python"

# Regular similarity search - returns similar documents (redundant)
print("Regular similarity search:")
results = vectorstore.similarity_search(query, k=3)
for doc in results:
    print(f"- {doc.page_content}")

# MMR search - balances relevance with diversity
print("\n\nMMR search (diverse results):")
results = vectorstore.max_marginal_relevance_search(
    query, 
    k=3,
    fetch_k=5,  # Fetch more candidates
    lambda_mult=0.5  # Balance: 0=max diversity, 1=max relevance
)
for doc in results:
    print(f"- {doc.page_content}")

# MMR returns more diverse results, avoiding redundancy
```

---

## 🔧 Example 9: Different Embedding Models

```python
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

documents = [
    Document(page_content="Machine learning is fascinating"),
    Document(page_content="Deep learning uses neural networks"),
]

# OpenAI embeddings (paid, high quality)
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
openai_store = FAISS.from_documents(documents, openai_embeddings)
print(f"OpenAI embedding dimensions: {openai_store.index.d}")

# HuggingFace embeddings (free, local)
hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
hf_store = FAISS.from_documents(documents, hf_embeddings)
print(f"HuggingFace embedding dimensions: {hf_store.index.d}")

# Compare search results
query = "What is deep learning?"

print("\nOpenAI results:")
for doc in openai_store.similarity_search(query, k=2):
    print(f"- {doc.page_content}")

print("\nHuggingFace results:")
for doc in hf_store.similarity_search(query, k=2):
    print(f"- {doc.page_content}")

# Both should return similar results, but OpenAI often performs better
```

---

## 🌐 Example 10: Pinecone (Cloud Vector Store)

```python
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
import os

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Create index (do this once)
index_name = "langchain-demo"

# Check if index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embedding dimensions
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Create documents
documents = [
    Document(page_content="Pinecone is a cloud vector database"),
    Document(page_content="Vector databases enable semantic search"),
    Document(page_content="Embeddings represent text as vectors"),
]

# Create vector store
vectorstore = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    index_name=index_name
)

# Search
query = "What is Pinecone?"
results = vectorstore.similarity_search(query, k=2)

for doc in results:
    print(doc.page_content)

# Benefits: Fully managed, scalable, persistent
# Drawbacks: Requires API key, paid service
```

---

## 📚 Example 11: Complete RAG Pipeline with Vector Store

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Step 1: Load documents
loader = TextLoader("knowledge_base.txt")
documents = loader.load()
print(f"Loaded {len(documents)} document(s)")

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# Step 3: Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
print("Vector store created")

# Step 4: Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Return top 3 chunks
)

# Step 5: Create RAG chain
llm = ChatOpenAI(model="gpt-4", temperature=0)

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

Question: {question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Step 6: Ask questions
question = "What is machine learning?"
answer = rag_chain.invoke(question)

print(f"\nQuestion: {question}")
print(f"Answer: {answer}")

# The chain:
# 1. Retrieves relevant chunks from vector store
# 2. Formats them as context
# 3. Passes to LLM with question
# 4. Returns answer based on retrieved context
```

---

## 🔄 Example 12: Retriever Types

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings()

documents = [
    Document(page_content=f"Document {i}: Content about topic {i}")
    for i in range(10)
]

vectorstore = FAISS.from_documents(documents, embeddings)

# 1. Similarity Search (default)
retriever_similarity = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 2. MMR (diversity)
retriever_mmr = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# 3. Similarity with score threshold
retriever_threshold = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.8,  # Only return if similarity > 0.8
        "k": 5
    }
)

# Test different retrievers
query = "topic 5"

print("Similarity retriever:")
for doc in retriever_similarity.invoke(query):
    print(f"- {doc.page_content}")

print("\nMMR retriever:")
for doc in retriever_mmr.invoke(query):
    print(f"- {doc.page_content}")

print("\nThreshold retriever:")
for doc in retriever_threshold.invoke(query):
    print(f"- {doc.page_content}")
```

---

## 📊 Example 13: Hybrid Search (Keyword + Semantic)

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

embeddings = OpenAIEmbeddings()

documents = [
    Document(page_content="Python programming language is versatile"),
    Document(page_content="Machine learning with scikit-learn"),
    Document(page_content="Deep learning frameworks like TensorFlow"),
    Document(page_content="Python is great for data science"),
]

# Semantic search (vector store)
vectorstore = Chroma.from_documents(documents, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Keyword search (BM25)
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 2

# Hybrid: Combine both
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]  # Equal weight to both
)

# Test
query = "Python for ML"

print("Vector search only:")
for doc in vector_retriever.invoke(query):
    print(f"- {doc.page_content}")

print("\nKeyword search only:")
for doc in bm25_retriever.invoke(query):
    print(f"- {doc.page_content}")

print("\nHybrid search:")
for doc in ensemble_retriever.invoke(query):
    print(f"- {doc.page_content}")

# Hybrid often gives best results by combining both approaches
```

---

## 🎯 Example 14: Contextual Compression (Re-ranking)

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create documents
documents = [
    Document(page_content="Python is a high-level programming language. It was created by Guido van Rossum in 1991. Python emphasizes code readability with its notable use of significant indentation."),
    Document(page_content="Machine learning is a subset of AI. It involves training algorithms on data. Python is commonly used for ML because of libraries like scikit-learn and TensorFlow."),
    Document(page_content="Deep learning uses neural networks with many layers. It has achieved breakthroughs in computer vision and NLP. Popular frameworks include PyTorch and TensorFlow."),
]

vectorstore = FAISS.from_documents(documents, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create compressor - extracts only relevant parts
compressor = LLMChainExtractor.from_llm(llm)

# Compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Test
query = "Why is Python used for machine learning?"

print("Base retriever (full documents):")
for doc in base_retriever.invoke(query):
    print(f"- {doc.page_content}\n")

print("\nCompression retriever (relevant parts only):")
for doc in compression_retriever.invoke(query):
    print(f"- {doc.page_content}\n")

# Compression retriever extracts only the relevant sentences!
```

---

## 🔥 Best Practices

### **1. Choose the Right Vector Store**
```python
# Development / Prototyping
vectorstore = FAISS.from_documents(docs, embeddings)

# Small production app
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./db")

# Large scale production
vectorstore = PineconeVectorStore.from_documents(docs, embeddings, index_name="prod")
```

### **2. Optimize Chunk Size for Embeddings**
```python
# Too small - loses context
splitter = RecursiveCharacterTextSplitter(chunk_size=100)  # ❌

# Good balance
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # ✅

# Too large - less precise retrieval
splitter = RecursiveCharacterTextSplitter(chunk_size=5000)  # ❌
```

### **3. Use Metadata for Filtering**
```python
# ✅ Add useful metadata
Document(
    page_content="...",
    metadata={
        "source": "file.pdf",
        "page": 5,
        "author": "John Doe",
        "date": "2024-01-15",
        "category": "technical"
    }
)

# Filter during search
results = vectorstore.similarity_search(
    query,
    filter={"category": "technical", "date": {"$gte": "2024-01-01"}}
)
```

### **4. Save Vector Stores for Reuse**
```python
# ✅ Save after creation
vectorstore.save_local("./my_index")

# Load later - much faster than recreating
vectorstore = FAISS.load_local("./my_index", embeddings)
```

### **5. Use MMR for Diverse Results**
```python
# ✅ When you want variety
results = vectorstore.max_marginal_relevance_search(
    query,
    k=5,
    fetch_k=20,
    lambda_mult=0.5
)
```

### **6. Monitor Embedding Costs**
```python
# OpenAI charges per token embedded
# text-embedding-3-small: $0.02 / 1M tokens
# text-embedding-3-large: $0.13 / 1M tokens

# For development, consider free local embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

---

## 📊 Quick Comparison

| Aspect | FAISS | Chroma | Pinecone |
|--------|-------|--------|----------|
| Hosting | Local | Local/Cloud | Cloud |
| Persistence | Manual save/load | Automatic | Automatic |
| Scalability | Limited | Medium | High |
| Cost | Free | Free | Paid |
| Setup | Easy | Easy | Medium |
| Best for | Dev/testing | Small apps | Production |

---

## 🎯 Practical Exercise

```python
"""
Exercise: Build a Document Q&A System

Create a system that:
1. Loads multiple documents from a directory
2. Splits them into chunks
3. Creates a vector store
4. Enables semantic search
5. Answers questions using RAG

Requirements:
- Support PDF and text files
- Use FAISS vector store
- Implement MMR search
- Include metadata filtering
- Save/load vector store

Test with sample documents and questions.
"""

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class DocumentQA:
    def __init__(self, docs_directory: str, vector_store_path: str = None):
        """Initialize the Q&A system."""
        # TODO: Initialize embeddings, LLM
        # TODO: Load or create vector store
        pass
    
    def load_and_process_documents(self, docs_directory: str):
        """Load documents and create vector store."""
        # TODO: Load documents
        # TODO: Split into chunks
        # TODO: Create vector store
        # TODO: Save vector store
        pass
    
    def ask(self, question: str, use_mmr: bool = False, filter_metadata: dict = None):
        """Answer a question using RAG."""
        # TODO: Retrieve relevant documents
        # TODO: Format context
        # TODO: Generate answer
        pass

# Test
qa_system = DocumentQA(
    docs_directory="./documents",
    vector_store_path="./qa_vectorstore"
)

# Ask questions
print(qa_system.ask("What is machine learning?"))
print(qa_system.ask("Explain neural networks", use_mmr=True))
```

---

## ✅ Key Takeaways

1. **Embeddings convert text to vectors** that capture semantic meaning
2. **Vector stores enable semantic search** - find similar content, not just keywords
3. **FAISS is best for development** - fast, free, local
4. **Chroma is good for small production** - persistent, easy to use
5. **Pinecone for large scale** - fully managed, scalable
6. **Chunk size matters** - 500-1000 chars usually works well
7. **Use MMR for diversity** - avoid redundant results
8. **Metadata filtering** enables precise search
9. **Save vector stores** - avoid re-embedding every time
10. **RAG = Retrieval + Generation** - retrieve context, then generate answer

---

## 📝 Understanding Check

1. What's the difference between similarity search and MMR search?
2. Why save a vector store instead of recreating it?
3. When would you use metadata filtering?
4. What's the benefit of using Pinecone over FAISS?

**Ready for the next section on RAG (Retrieval-Augmented Generation)?** This is where everything comes together! Or would you like to:
- See the exercise solution?
- Practice more with vector stores?
- Ask questions about embeddings?