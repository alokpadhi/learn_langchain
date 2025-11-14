# Section 8: RAG (Retrieval-Augmented Generation) 🎯

RAG combines retrieval from a knowledge base with LLM generation to answer questions accurately with external knowledge.

---

## 🎯 What is RAG?

**RAG = Retrieval + Augmented + Generation**

**The Problem:**
```python
# LLMs have limitations:
# 1. Knowledge cutoff (outdated information)
# 2. No access to private/proprietary data
# 3. Hallucinations (making up facts)

llm.invoke("What are our Q4 2024 sales figures?")
# ❌ "I don't have access to that information"
# or worse: makes up numbers
```

**The Solution:**
```python
# RAG Process:
# 1. User asks question
# 2. Retrieve relevant documents from knowledge base
# 3. Pass documents + question to LLM
# 4. LLM generates answer based on retrieved context
# ✅ Accurate, grounded in your data
```

---

## 🔄 RAG Architecture

```
User Question
      ↓
   Embedding Model (convert question to vector)
      ↓
   Vector Store (semantic search for relevant docs)
      ↓
   Retrieve Top K Documents
      ↓
   Format: Context + Question
      ↓
   LLM (generate answer based on context)
      ↓
   Answer (grounded in retrieved documents)
```

---

## 💻 Example 1: Basic RAG System

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Step 1: Create knowledge base
documents = [
    Document(page_content="LangChain is a framework for building LLM applications. It was created in 2022."),
    Document(page_content="LangChain supports multiple LLM providers including OpenAI, Anthropic, and HuggingFace."),
    Document(page_content="RAG (Retrieval-Augmented Generation) is a technique that combines retrieval with generation."),
    Document(page_content="Vector stores like FAISS and Chroma enable semantic search in LangChain."),
    Document(page_content="LCEL (LangChain Expression Language) uses the pipe operator for composing chains."),
]

# Step 2: Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Step 3: Create RAG prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

Question: {question}

Answer:
""")

# Step 4: Define formatting function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Step 5: Create RAG chain
llm = ChatOpenAI(model="gpt-4", temperature=0)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Step 6: Ask questions
question = "What is LangChain?"
answer = rag_chain.invoke(question)

print(f"Question: {question}")
print(f"Answer: {answer}")

# Output: "LangChain is a framework for building LLM applications, created in 2022..."
```

---

## 📚 Example 2: RAG with Document Loading and Splitting

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
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
splits = text_splitter.split_documents(documents)
print(f"Created {len(splits)} chunks")

# Step 3: Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(splits, embeddings)

# Step 4: Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Step 5: Create RAG chain
llm = ChatOpenAI(model="gpt-4", temperature=0)

prompt = ChatPromptTemplate.from_template("""
You are an assistant for question-answering tasks. Use the following pieces of 
retrieved context to answer the question. If you don't know the answer, say that 
you don't know. Keep the answer concise.

Context: {context}

Question: {question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Step 6: Interactive Q&A
questions = [
    "What is the main topic of the document?",
    "Can you summarize the key points?",
    "What recommendations are mentioned?"
]

for question in questions:
    answer = rag_chain.invoke(question)
    print(f"\nQ: {question}")
    print(f"A: {answer}")
```

---

## 🔍 Example 3: RAG with Source Attribution

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Create documents with metadata
documents = [
    Document(
        page_content="Python was created by Guido van Rossum in 1991.",
        metadata={"source": "python_history.txt", "page": 1}
    ),
    Document(
        page_content="Python is widely used for data science and machine learning.",
        metadata={"source": "python_applications.txt", "page": 1}
    ),
    Document(
        page_content="Popular Python libraries include NumPy, Pandas, and TensorFlow.",
        metadata={"source": "python_libraries.txt", "page": 1}
    ),
]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Format with sources
def format_docs_with_sources(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        formatted.append(f"[Source {i}: {source}, Page {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)

# Prompt that encourages citing sources
prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below. Cite your sources by 
referencing [Source X] in your answer.

Context:
{context}

Question: {question}

Answer (with source citations):
""")

llm = ChatOpenAI(model="gpt-4", temperature=0)

rag_chain = (
    {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Test
question = "Who created Python and what is it used for?"
answer = rag_chain.invoke(question)

print(f"Question: {question}\n")
print(f"Answer: {answer}")

# Output includes source citations like:
# "Python was created by Guido van Rossum [Source 1] and is widely used 
#  for data science and machine learning [Source 2]."
```

---

## 🎨 Example 4: RAG with Conversational Memory

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory

# Setup knowledge base
documents = [
    Document(page_content="Machine learning is a subset of AI that enables systems to learn from data."),
    Document(page_content="Supervised learning uses labeled data for training."),
    Document(page_content="Unsupervised learning finds patterns in unlabeled data."),
    Document(page_content="Neural networks are inspired by biological neurons."),
]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Memory for conversation history
memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history"
)

# Prompt with memory
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context to answer questions."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Context: {context}\n\nQuestion: {question}")
])

llm = ChatOpenAI(model="gpt-4", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Conversational RAG function
def conversational_rag(question: str):
    # Retrieve context
    docs = retriever.invoke(question)
    context = format_docs(docs)
    
    # Get memory
    memory_vars = memory.load_memory_variables({})
    chat_history = memory_vars.get("chat_history", [])
    
    # Generate response
    response = (prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question,
        "chat_history": chat_history
    })
    
    # Save to memory
    memory.save_context({"input": question}, {"output": response})
    
    return response

# Test conversational RAG
print(conversational_rag("What is machine learning?"))
print("\n" + conversational_rag("What are its types?"))  # Uses context from previous Q
print("\n" + conversational_rag("Tell me more about the second type"))  # References "unsupervised"

# The system remembers the conversation!
```

---

## 🔧 Example 5: RAG with Query Transformation

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Knowledge base
documents = [
    Document(page_content="Python is a high-level programming language created in 1991."),
    Document(page_content="Python uses dynamic typing and automatic memory management."),
    Document(page_content="Popular Python frameworks include Django, Flask, and FastAPI."),
]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

llm = ChatOpenAI(model="gpt-4", temperature=0)

# Query transformation - improve vague questions
query_transform_prompt = ChatPromptTemplate.from_template("""
Given a vague or poorly formed question, reformulate it to be more specific 
and searchable. Only output the reformulated question.

Original question: {question}

Reformulated question:
""")

query_transformer = query_transform_prompt | llm | StrOutputParser()

# RAG prompt
rag_prompt = ChatPromptTemplate.from_template("""
Context: {context}

Question: {question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG chain with query transformation
rag_with_transform = (
    {
        "context": (
            {"question": RunnablePassthrough()} 
            | query_transformer  # Transform query first
            | vectorstore.as_retriever() 
            | format_docs
        ),
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Test with vague question
vague_question = "tell me about that language"
answer = rag_with_transform.invoke(vague_question)

print(f"Original question: {vague_question}")
print(f"Answer: {answer}")

# The system reformulates "that language" to "Python programming language"
# before retrieval, getting better results!
```

---

## 🎯 Example 6: Multi-Query RAG

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import MultiQueryRetriever

# Knowledge base
documents = [
    Document(page_content="Deep learning uses neural networks with multiple layers."),
    Document(page_content="Convolutional neural networks excel at image processing."),
    Document(page_content="Recurrent neural networks are good for sequential data."),
    Document(page_content="Transformers have revolutionized natural language processing."),
]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Multi-query retriever - generates multiple search queries
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    llm=llm
)

# Test
question = "What neural network works best for images?"

# The retriever will:
# 1. Generate variations: "CNN for image processing", "image neural networks", etc.
# 2. Search with each variation
# 3. Return unique documents from all searches

docs = retriever.invoke(question)

print(f"Question: {question}")
print(f"\nRetrieved {len(docs)} unique documents:")
for i, doc in enumerate(docs, 1):
    print(f"{i}. {doc.page_content}")

# Gets more comprehensive results by searching with multiple variations
```

---

## 📊 Example 7: RAG with Metadata Filtering

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Documents with rich metadata
documents = [
    Document(
        page_content="Python 3.12 introduced new features including improved error messages.",
        metadata={"language": "python", "version": "3.12", "year": 2024, "category": "release"}
    ),
    Document(
        page_content="JavaScript ES2024 added new array methods and temporal API.",
        metadata={"language": "javascript", "version": "ES2024", "year": 2024, "category": "release"}
    ),
    Document(
        page_content="Python best practices recommend using type hints and docstrings.",
        metadata={"language": "python", "year": 2024, "category": "best_practices"}
    ),
    Document(
        page_content="Python 3.11 improved performance significantly.",
        metadata={"language": "python", "version": "3.11", "year": 2023, "category": "release"}
    ),
]

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_rag"
)

# RAG with filtering
def rag_with_filter(question: str, filter_dict: dict = None):
    # Retrieve with filter
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3, "filter": filter_dict} if filter_dict else {"k": 3}
    )
    
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Generate answer
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    prompt = ChatPromptTemplate.from_template("""
    Context: {context}
    
    Question: {question}
    
    Answer:
    """)
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})

# Test with filters
print("Without filter:")
print(rag_with_filter("What are the new features?"))

print("\n\nWith Python filter:")
print(rag_with_filter(
    "What are the new features?",
    filter_dict={"language": "python"}
))

print("\n\nWith Python + 2024 filter:")
print(rag_with_filter(
    "What are the new features?",
    filter_dict={"language": "python", "year": 2024}
))
```

---

## 🔄 Example 8: Self-Query RAG

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

# Documents with metadata
documents = [
    Document(
        page_content="Inception is a sci-fi thriller about dreams.",
        metadata={"genre": "sci-fi", "year": 2010, "rating": 8.8}
    ),
    Document(
        page_content="The Dark Knight is a superhero crime film.",
        metadata={"genre": "action", "year": 2008, "rating": 9.0}
    ),
    Document(
        page_content="Interstellar explores space and time travel.",
        metadata={"genre": "sci-fi", "year": 2014, "rating": 8.6}
    ),
    Document(
        page_content="The Godfather is a classic crime drama.",
        metadata={"genre": "drama", "year": 1972, "rating": 9.2}
    ),
]

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# Define metadata fields
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie",
        type="string"
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer"
    ),
    AttributeInfo(
        name="rating",
        description="The movie rating (0-10)",
        type="float"
    ),
]

# Self-query retriever - automatically extracts filters from natural language
llm = ChatOpenAI(model="gpt-4", temperature=0)

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Movie descriptions",
    metadata_field_info=metadata_field_info,
    verbose=True
)

# Test with natural language that includes filters
questions = [
    "Show me sci-fi movies",
    "Find movies released after 2010",
    "Which movies have a rating above 9.0?",
    "Show sci-fi films from the 2010s with high ratings"
]

for question in questions:
    print(f"\nQuestion: {question}")
    docs = retriever.invoke(question)
    for doc in docs:
        print(f"  - {doc.page_content} | {doc.metadata}")

# The retriever automatically extracts filters from natural language!
# "sci-fi movies" → filter: {"genre": "sci-fi"}
# "released after 2010" → filter: {"year": {"$gt": 2010}}
```

---

## 🎨 Example 9: Parent Document Retriever

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Long documents
documents = [
    Document(
        page_content="""
        Machine Learning Overview:
        Machine learning is a subset of artificial intelligence. It focuses on 
        training algorithms to learn patterns from data without explicit programming.
        
        Types of Machine Learning:
        1. Supervised Learning: Uses labeled data
        2. Unsupervised Learning: Finds patterns in unlabeled data
        3. Reinforcement Learning: Learns through trial and error
        
        Applications:
        ML is used in image recognition, NLP, recommendation systems, and more.
        """,
        metadata={"source": "ml_guide.txt"}
    )
]

embeddings = OpenAIEmbeddings()

# Small chunks for retrieval
child_splitter = RecursiveCharacterTextSplitter(chunk_size=100)

# Parent store - keeps full documents
parent_store = InMemoryStore()

# Vector store - stores small chunks
vectorstore = Chroma(
    collection_name="parent_doc_retriever",
    embedding_function=embeddings
)

# Parent document retriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=parent_store,
    child_splitter=child_splitter,
)

# Add documents
retriever.add_documents(documents, ids=None)

# Search with small chunks, retrieve full parent documents
query = "What are the types of machine learning?"
retrieved_docs = retriever.invoke(query)

print(f"Query: {query}\n")
print("Retrieved documents:")
for doc in retrieved_docs:
    print(f"Source: {doc.metadata['source']}")
    print(f"Content length: {len(doc.page_content)} chars")
    print(f"Preview: {doc.page_content[:200]}...\n")

# Benefits:
# - Search precise small chunks
# - Return full context (parent document)
# - Best of both worlds!
```

---

## 🔥 Example 10: Complete Production RAG System

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

class ProductionRAG:
    """Production-ready RAG system with all best practices."""
    
    def __init__(self, docs_directory: str, persist_directory: str = "./rag_db"):
        self.docs_directory = docs_directory
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        
    def load_documents(self):
        """Load documents from directory."""
        print("Loading documents...")
        loader = DirectoryLoader(
            self.docs_directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def split_documents(self, documents):
        """Split documents into chunks."""
        print("Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)
        print(f"Created {len(splits)} chunks")
        return splits
    
    def create_vectorstore(self, splits):
        """Create or load vector store."""
        if os.path.exists(self.persist_directory):
            print("Loading existing vector store...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            print("Creating new vector store...")
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        print("Vector store ready")
    
    def setup_retriever(self, search_type="mmr", k=4):
        """Setup retriever with specified search type."""
        self.retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k, "fetch_k": 20} if search_type == "mmr" else {"k": k}
        )
        print(f"Retriever configured: {search_type}, k={k}")
    
    def create_rag_chain(self):
        """Create RAG chain."""
        prompt = ChatPromptTemplate.from_template("""
        You are an assistant for question-answering tasks. Use the following 
        pieces of retrieved context to answer the question. If you don't know 
        the answer, say that you don't know. Use three sentences maximum and 
        keep the answer concise.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        print("RAG chain created")
    
    def initialize(self, force_reload=False):
        """Initialize the entire RAG system."""
        if force_reload or not os.path.exists(self.persist_directory):
            documents = self.load_documents()
            splits = self.split_documents(documents)
            self.create_vectorstore(splits)
        else:
            self.create_vectorstore(None)
        
        self.setup_retriever(search_type="mmr", k=4)
        self.create_rag_chain()
        print("\n✅ RAG system ready!\n")
    
    def query(self, question: str, return_sources=False):
        """Query the RAG system."""
        if self.rag_chain is None:
            raise ValueError("RAG system not initialized. Call initialize() first.")
        
        # Get answer
        answer = self.rag_chain.invoke(question)
        
        if return_sources:
            # Get source documents
            docs = self.retriever.invoke(question)
            sources = [doc.metadata.get("source", "Unknown") for doc in docs]
            return {"answer": answer, "sources": list(set(sources))}
        
        return answer
    
    def interactive_mode(self):
        """Run in interactive Q&A mode."""
        print("Interactive RAG Mode (type 'exit' to quit)")
        print("-" * 50)
        
        while True:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            result = self.query(question, return_sources=True)
            print(f"\nAnswer: {result['answer']}")
            print(f"\nSources: {', '.join(result['sources'])}")

# Usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = ProductionRAG(
        docs_directory="./documents",
        persist_directory="./production_rag_db"
    )
    
    # Initialize (loads or creates vector store)
    rag.initialize(force_reload=False)
    
    # Query
    result = rag.query("What is machine learning?", return_sources=True)
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    
    # Or run in interactive mode
    # rag.interactive_mode()
```

---

## 🎯 RAG Best Practices

### **1. Chunk Size Optimization**
```python
# Too small - loses context
splitter = RecursiveCharacterTextSplitter(chunk_size=100)  # ❌

# Good for Q&A
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # ✅

# Good for summarization
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)  # ✅
```

### **2. Use MMR for Diversity**
```python
# ✅ Avoid redundant results
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5}
)
```

### **3. Add Source Attribution**
```python
# ✅ Always cite sources in production
prompt = ChatPromptTemplate.from_template("""
Answer based on context and cite your sources.

Context: {context}
Question: {question}

Answer (with sources):
""")
```

### **4. Handle "I Don't Know"**
```python
# ✅ Teach model to admit uncertainty
prompt = ChatPromptTemplate.from_template("""
Use the context to answer. If the context doesn't contain the answer, 
say "I don't have enough information to answer that question."

Context: {context}
Question: {question}

Answer:
""")
```

### **5. Use Temperature=0 for Factual Tasks**
```python
# ✅ Consistent, factual answers
llm = ChatOpenAI(model="gpt-4", temperature=0)
```

### **6. Monitor Retrieved Context Quality**
```python
# ✅ Log retrieved documents for debugging
docs = retriever.invoke(question)
for i, doc in enumerate(docs):
    print(f"Doc {i}: {doc.metadata}")
    print(f"Similarity: {doc.page_content[:100]}...")
```

---

## 📊 RAG Evaluation Metrics

```python
from langchain.evaluation import load_evaluator

# Relevance: Are retrieved docs relevant?
relevance_evaluator = load_evaluator("relevance")

# Faithfulness: Is answer based on context?
faithfulness_evaluator = load_evaluator("faithfulness")

# Example evaluation
question = "What is machine learning?"
retrieved_docs = retriever.invoke(question)
answer = rag_chain.invoke(question)

# Check relevance
relevance_score = relevance_evaluator.evaluate_strings(
    prediction=retrieved_docs[0].page_content,
    input=question
)
print(f"Relevance: {relevance_score}")

# Check faithfulness
faithfulness_score = faithfulness_evaluator.evaluate_strings(
    prediction=answer,
    input=question,
    reference=format_docs(retrieved_docs)
)
print(f"Faithfulness: {faithfulness_score}")
```

---

## 🎯 Practical Exercise

```python
"""
Exercise: Build a Multi-Source RAG System

Create a RAG system that:
1. Loads documents from multiple sources (PDFs, websites, text files)
2. Creates separate collections for each source type
3. Allows querying specific sources or all sources
4. Returns answers with source attribution
5. Includes conversational memory

Requirements:
- Support DirectoryLoader for PDFs
- Support WebBaseLoader for URLs
- Use Chroma with multiple collections
- Implement metadata filtering
- Add conversation memory
- Interactive CLI

Test with mixed document types.
"""

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory

class MultiSourceRAG:
    def __init__(self):
        # TODO: Initialize components
        pass
    
    def load_pdfs(self, directory: str):
        # TODO: Load and process PDFs
        pass
    
    def load_websites(self, urls: list):
        # TODO: Load and process websites
        pass
    
    def create_collections(self):
        # TODO: Create separate collections for each source
        pass
    
    def query(self, question: str, source_filter: str = None):
        # TODO: Query with optional source filtering
        # TODO: Use memory for context
        pass
    
    def interactive_mode(self):
        # TODO: Interactive CLI with commands
        # Commands: query, filter:pdf, filter:web, clear, exit
        pass

# Test
# rag = MultiSourceRAG()
# rag.load_pdfs("./pdfs")
# rag.load_websites(["https://example.com/article1", "https://example.com/article2"])
# rag.create_collections()
# rag.interactive_mode()
```

---

## ✅ Key Takeaways

1. **RAG = Retrieval + Generation** - combine knowledge base with LLM
2. **Vector search finds relevant context** - semantic, not keyword-based
3. **Chunk size matters** - 500-1000 chars for Q&A, larger for summarization
4. **Use MMR for diversity** - avoid redundant retrieved documents
5. **Always cite sources** - build trust and transparency
6. **Temperature=0 for facts** - consistent, grounded answers
7. **Handle uncertainty** - teach model to say "I don't know"
8. **Metadata filtering** - search specific document types/dates
9. **Query transformation** - improve vague questions
10. **Evaluate performance** - measure relevance and faithfulness

---

## 📝 Understanding Check

1. What problem does RAG solve?
2. Why use MMR instead of regular similarity search?
3. How does query transformation improve RAG?
4. What's the benefit of Parent Document Retriever?

**Ready for the next section on Tools and Function Calling?** This enables LLMs to interact with external systems! Or would you like to:
- See the exercise solution?
- Practice more with RAG patterns?
- Ask questions about specific RAG techniques?