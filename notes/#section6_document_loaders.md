# Section 6: Document Loaders and Text Splitters 📄

Document Loaders extract text from various sources, and Text Splitters break documents into manageable chunks for LLMs.

---

## 🎯 Why Document Loaders and Text Splitters?

**The Problem:**
- LLMs have token limits (e.g., GPT-4: 128k tokens)
- Documents can be massive (books, research papers, codebases)
- Need to break documents into chunks for processing
- Must maintain semantic coherence

**The Solution:**
1. **Document Loaders** - Extract text from PDFs, Word docs, websites, etc.
2. **Text Splitters** - Intelligently split text into chunks
3. Use chunks in RAG (Retrieval-Augmented Generation)

---

## 📋 Part A: Document Loaders

### **Types of Document Loaders**

| Loader | Source Type | Package |
|--------|-------------|---------|
| TextLoader | Plain text files | langchain-community |
| PyPDFLoader | PDF files | langchain-community |
| UnstructuredWordDocumentLoader | Word docs (.docx) | langchain-community |
| CSVLoader | CSV files | langchain-community |
| WebBaseLoader | Websites/URLs | langchain-community |
| GitbookLoader | Gitbook docs | langchain-community |
| NotionDirectoryLoader | Notion exports | langchain-community |
| DirectoryLoader | Entire directories | langchain-community |
| UnstructuredMarkdownLoader | Markdown files | langchain-community |
| JSONLoader | JSON files | langchain-community |

---

## 💻 Example 1: TextLoader (Simple Text Files)

```python
from langchain_community.document_loaders import TextLoader

# Load a text file
loader = TextLoader("sample.txt", encoding="utf-8")
documents = loader.load()

print(f"Loaded {len(documents)} document(s)")
print(f"First document:")
print(f"Content length: {len(documents[0].page_content)}")
print(f"Metadata: {documents[0].metadata}")
print(f"Content preview: {documents[0].page_content[:200]}")

# Document structure:
# Document(
#     page_content="The actual text content...",
#     metadata={'source': 'sample.txt'}
# )
```

---

## 📄 Example 2: PyPDFLoader (PDF Files)

```python
from langchain_community.document_loaders import PyPDFLoader

# Load PDF
loader = PyPDFLoader("research_paper.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages")

# Each page is a separate document
for i, doc in enumerate(documents[:3]):  # First 3 pages
    print(f"\n--- Page {i+1} ---")
    print(f"Content length: {len(doc.page_content)}")
    print(f"Metadata: {doc.metadata}")
    print(f"Preview: {doc.page_content[:150]}")

# Output metadata includes:
# {'source': 'research_paper.pdf', 'page': 0}
```

**Alternative PDF loaders:**
```python
# PyMuPDFLoader - faster, better formatting
from langchain_community.document_loaders import PyMuPDFLoader
loader = PyMuPDFLoader("document.pdf")

# PDFPlumberLoader - better table extraction
from langchain_community.document_loaders import PDFPlumberLoader
loader = PDFPlumberLoader("document.pdf")

# UnstructuredPDFLoader - advanced parsing
from langchain_community.document_loaders import UnstructuredPDFLoader
loader = UnstructuredPDFLoader("document.pdf")
```

---

## 📝 Example 3: UnstructuredWordDocumentLoader (Word Docs)

```python
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

# Load .docx file
loader = UnstructuredWordDocumentLoader("report.docx")
documents = loader.load()

print(f"Loaded {len(documents)} document(s)")
print(f"Content: {documents[0].page_content[:500]}")

# Note: Requires 'unstructured' package
# pip install unstructured
```

---

## 📊 Example 4: CSVLoader (CSV Files)

```python
from langchain_community.document_loaders import CSVLoader

# Load CSV - each row becomes a document
loader = CSVLoader(
    file_path="data.csv",
    csv_args={
        'delimiter': ',',
        'quotechar': '"',
    }
)

documents = loader.load()

print(f"Loaded {len(documents)} rows")
print(f"First row as document:")
print(documents[0].page_content)
print(f"Metadata: {documents[0].metadata}")

# Output example:
# page_content: "name: John, age: 30, occupation: Engineer"
# metadata: {'source': 'data.csv', 'row': 0}
```

**Specify columns:**
```python
# Only load specific columns
loader = CSVLoader(
    file_path="data.csv",
    source_column="id",  # Use this as source in metadata
)
```

---

## 🌐 Example 5: WebBaseLoader (Websites)

```python
from langchain_community.document_loaders import WebBaseLoader

# Load single webpage
loader = WebBaseLoader("https://python.langchain.com/docs/get_started/introduction")
documents = loader.load()

print(f"Loaded {len(documents)} document(s)")
print(f"Title: {documents[0].metadata.get('title', 'N/A')}")
print(f"Content length: {len(documents[0].page_content)}")
print(f"Preview: {documents[0].page_content[:300]}")

# Load multiple URLs
urls = [
    "https://example.com/page1",
    "https://example.com/page2",
    "https://example.com/page3",
]

loader = WebBaseLoader(urls)
documents = loader.load()
print(f"Loaded {len(documents)} pages from {len(urls)} URLs")
```

**Advanced web scraping:**
```python
from langchain_community.document_loaders import WebBaseLoader
import bs4

# Use BeautifulSoup to extract specific elements
loader = WebBaseLoader(
    web_paths=["https://example.com/article"],
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(
            class_=("article-content", "post-content")
        )
    }
)

documents = loader.load()
```

---

## 📁 Example 6: DirectoryLoader (Load Multiple Files)

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Load all .txt files from a directory
loader = DirectoryLoader(
    path="./documents",
    glob="**/*.txt",  # Recursive search for .txt files
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'}
)

documents = loader.load()

print(f"Loaded {len(documents)} documents")
for doc in documents[:3]:
    print(f"\nSource: {doc.metadata['source']}")
    print(f"Length: {len(doc.page_content)}")

# Load different file types
# PDFs
pdf_loader = DirectoryLoader(
    path="./documents",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

# Markdown files
md_loader = DirectoryLoader(
    path="./documents",
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader
)
```

---

## 🔖 Example 7: JSONLoader (JSON Files)

```python
from langchain_community.document_loaders import JSONLoader

# Simple JSON loading
loader = JSONLoader(
    file_path="data.json",
    jq_schema=".messages[].content",  # Extract specific fields
    text_content=False
)

documents = loader.load()

# Example JSON:
# {
#     "messages": [
#         {"role": "user", "content": "Hello"},
#         {"role": "assistant", "content": "Hi there!"}
#     ]
# }

# More complex extraction
loader = JSONLoader(
    file_path="users.json",
    jq_schema=".users[] | {name: .name, bio: .bio}",
    text_content=False
)

# Extracts name and bio from each user
```

---

## 📚 Example 8: Custom Document Loader

```python
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from typing import Iterator

class CustomAPILoader(BaseLoader):
    """Load documents from a custom API."""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
    
    def lazy_load(self) -> Iterator[Document]:
        """Lazy load documents (memory efficient)."""
        import requests
        
        response = requests.get(self.api_url)
        data = response.json()
        
        for item in data['items']:
            yield Document(
                page_content=item['content'],
                metadata={
                    'source': self.api_url,
                    'id': item['id'],
                    'created_at': item['created_at']
                }
            )
    
    def load(self) -> list[Document]:
        """Load all documents."""
        return list(self.lazy_load())

# Use custom loader
loader = CustomAPILoader("https://api.example.com/documents")
documents = loader.load()
```

---

## 📋 Part B: Text Splitters

### **Why Split Text?**

```python
# Without splitting - exceeds context window
long_document = "... 100,000 words ..."
llm.invoke(long_document)  # ❌ Error: Token limit exceeded

# With splitting - manageable chunks
chunks = text_splitter.split_text(long_document)
for chunk in chunks:
    llm.invoke(chunk)  # ✅ Works!
```

---

## 🔪 Types of Text Splitters

| Splitter | Split By | Best For |
|----------|----------|----------|
| CharacterTextSplitter | Characters | Simple splitting |
| RecursiveCharacterTextSplitter | Multiple separators | General purpose (best) |
| TokenTextSplitter | Tokens | Precise token control |
| MarkdownTextSplitter | Markdown structure | Markdown docs |
| PythonCodeTextSplitter | Python syntax | Python code |
| LatexTextSplitter | LaTeX structure | LaTeX documents |

---

## 💻 Example 9: CharacterTextSplitter (Basic)

```python
from langchain_text_splitters import CharacterTextSplitter

text = """
Machine learning is a subset of artificial intelligence. 
It focuses on training algorithms to learn patterns from data.

Deep learning is a subset of machine learning.
It uses neural networks with multiple layers.

Natural language processing enables computers to understand human language.
It's used in chatbots, translation, and sentiment analysis.
"""

# Split by character count
splitter = CharacterTextSplitter(
    separator="\n\n",  # Split on double newlines (paragraphs)
    chunk_size=100,     # Max characters per chunk
    chunk_overlap=20,   # Overlap between chunks
    length_function=len,
)

chunks = splitter.split_text(text)

print(f"Split into {len(chunks)} chunks\n")
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk)
    print()
```

---

## 🎯 Example 10: RecursiveCharacterTextSplitter (Best for Most Cases)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
# Machine Learning Guide

Machine learning (ML) is a subset of artificial intelligence. It focuses on training 
algorithms to learn patterns from data without being explicitly programmed.

## Types of Machine Learning

1. Supervised Learning: Uses labeled data
2. Unsupervised Learning: Finds patterns in unlabeled data
3. Reinforcement Learning: Learns through trial and error

## Applications

ML is used in: image recognition, natural language processing, recommendation systems,
fraud detection, and autonomous vehicles.
"""

# Recursive splitter - tries multiple separators in order
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,      # Characters per chunk
    chunk_overlap=50,    # Overlap for context continuity
    length_function=len,
    separators=["\n\n", "\n", " ", ""]  # Try these in order
)

chunks = splitter.split_text(text)

print(f"Split into {len(chunks)} chunks\n")
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk)
    print(f"Length: {len(chunk)}\n")

# Why it's better: Maintains semantic boundaries
# Tries to split at paragraph boundaries first, then sentences, then words
```

---

## 🎫 Example 11: TokenTextSplitter (Precise Token Control)

```python
from langchain_text_splitters import TokenTextSplitter

text = """
Artificial intelligence encompasses machine learning, deep learning, 
natural language processing, computer vision, and robotics. These technologies
are transforming industries including healthcare, finance, transportation,
and entertainment.
"""

# Split by actual token count (important for LLM context limits)
splitter = TokenTextSplitter(
    chunk_size=50,      # Tokens per chunk
    chunk_overlap=10    # Token overlap
)

chunks = splitter.split_text(text)

print(f"Split into {len(chunks)} chunks\n")
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk)
    
    # Count actual tokens (rough estimate)
    tokens = len(chunk.split())
    print(f"~{tokens} words\n")
```

**With specific tokenizer:**
```python
from langchain_text_splitters import CharacterTextSplitter
from transformers import GPT2TokenizerFast

# Use GPT-2 tokenizer for accurate counts
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=100,
    chunk_overlap=20
)
```

---

## 🐍 Example 12: PythonCodeTextSplitter (Code-Aware)

```python
from langchain_text_splitters import PythonCodeTextSplitter

python_code = """
import numpy as np
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, data):
        self.data = data
    
    def clean_data(self):
        # Remove nulls
        self.data = self.data.dropna()
        return self
    
    def normalize(self):
        # Normalize features
        self.data = (self.data - self.data.mean()) / self.data.std()
        return self

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    return model

# Main execution
if __name__ == "__main__":
    processor = DataProcessor(data)
    processor.clean_data().normalize()
"""

# Split by Python syntax (classes, functions, etc.)
splitter = PythonCodeTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)

chunks = splitter.split_text(python_code)

print(f"Split into {len(chunks)} chunks\n")
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk)
    print()

# Maintains syntactic integrity - doesn't split in middle of functions
```

**Other code splitters:**
```python
# JavaScript
from langchain_text_splitters import JSTextSplitter

# Markdown
from langchain_text_splitters import MarkdownTextSplitter

# LaTeX
from langchain_text_splitters import LatexTextSplitter
```

---

## 📄 Example 13: Split Documents (Not Just Text)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Load document
loader = TextLoader("article.txt")
documents = loader.load()

# Split documents (preserves metadata!)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

split_docs = splitter.split_documents(documents)

print(f"Original: {len(documents)} documents")
print(f"After splitting: {len(split_docs)} chunks")

for i, doc in enumerate(split_docs[:3]):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Content length: {len(doc.page_content)}")
    print(f"Metadata: {doc.metadata}")
    print(f"Preview: {doc.page_content[:100]}...")

# Metadata is preserved and enhanced:
# {'source': 'article.txt', 'chunk': 0}
```

---

## 🎯 Example 14: Choosing the Right Chunk Size

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = "A long document..." * 1000  # Simulate long text

# Experiment with different chunk sizes
chunk_sizes = [100, 500, 1000, 2000]

for size in chunk_sizes:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=size // 10  # 10% overlap
    )
    
    chunks = splitter.split_text(text)
    
    print(f"Chunk size {size}:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Avg chunk length: {sum(len(c) for c in chunks) // len(chunks)}")
    print()

# Guidelines:
# - Small chunks (100-300): Precise retrieval, more chunks to search
# - Medium chunks (500-1000): Good balance (recommended)
# - Large chunks (1000-2000): More context, fewer chunks
```

---

## 🔄 Example 15: Semantic Chunking (Advanced)

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

text = """
The history of artificial intelligence began in the 1950s. 
Alan Turing proposed the Turing Test in 1950.

Machine learning emerged as a subfield in the 1980s.
Neural networks gained popularity during this period.

Deep learning revolutionized AI in the 2010s.
ImageNet competition in 2012 was a turning point.

Today, large language models dominate the field.
GPT and BERT architectures are widely used.
"""

# Split based on semantic similarity
embeddings = OpenAIEmbeddings()

splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile"  # or "standard_deviation", "interquartile"
)

chunks = splitter.create_documents([text])

print(f"Semantically split into {len(chunks)} chunks\n")
for i, chunk in enumerate(chunks):
    print(f"--- Semantic Chunk {i+1} ---")
    print(chunk.page_content)
    print()

# Groups semantically related sentences together
# More intelligent than character-based splitting
```

---

## 📊 Example 16: Complete Pipeline (Load → Split → Process)

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Step 1: Load documents
print("Loading documents...")
loader = DirectoryLoader(
    path="./articles",
    glob="**/*.txt",
    loader_cls=TextLoader
)
documents = loader.load()
print(f"Loaded {len(documents)} documents")

# Step 2: Split into chunks
print("\nSplitting documents...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Step 3: Process each chunk
print("\nProcessing chunks...")
llm = ChatOpenAI(model="gpt-4", temperature=0)
prompt = ChatPromptTemplate.from_template(
    "Summarize the following text in one sentence:\n\n{text}"
)
chain = prompt | llm | StrOutputParser()

summaries = []
for i, chunk in enumerate(chunks[:5]):  # Process first 5 chunks
    print(f"Processing chunk {i+1}...")
    summary = chain.invoke({"text": chunk.page_content})
    summaries.append({
        "source": chunk.metadata["source"],
        "summary": summary,
        "length": len(chunk.page_content)
    })

# Display results
print("\n=== Summaries ===")
for i, item in enumerate(summaries):
    print(f"\n{i+1}. Source: {item['source']}")
    print(f"   Length: {item['length']} chars")
    print(f"   Summary: {item['summary']}")
```

---

## 🔥 Best Practices

### **1. Choose the Right Splitter**
```python
# ✅ General text - use RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# ✅ Code - use language-specific splitters
splitter = PythonCodeTextSplitter(chunk_size=500)

# ✅ Markdown - use MarkdownTextSplitter
splitter = MarkdownTextSplitter(chunk_size=1000)
```

### **2. Set Appropriate Chunk Size**
```python
# Consider your use case:

# RAG (retrieval) - smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Summarization - larger chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

# Q&A - medium chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
```

### **3. Use Overlap for Context**
```python
# ✅ Good - 10-20% overlap
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200  # 20% overlap
)

# ❌ No overlap - context loss
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)
```

### **4. Preserve Metadata**
```python
# ✅ Use split_documents to keep metadata
chunks = splitter.split_documents(documents)

# ❌ split_text loses metadata
chunks = splitter.split_text(text)
```

### **5. Consider Token Limits**
```python
# Know your model's limits:
# GPT-4: 128k tokens
# GPT-3.5: 16k tokens
# Claude: 200k tokens

# For GPT-3.5, use smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,  # ~750 tokens
    chunk_overlap=300
)
```

---

## 📊 Quick Reference

| Document Type | Recommended Loader | Recommended Splitter |
|--------------|-------------------|---------------------|
| Text files | TextLoader | RecursiveCharacterTextSplitter |
| PDFs | PyPDFLoader | RecursiveCharacterTextSplitter |
| Word docs | UnstructuredWordDocumentLoader | RecursiveCharacterTextSplitter |
| Websites | WebBaseLoader | RecursiveCharacterTextSplitter |
| Python code | TextLoader | PythonCodeTextSplitter |
| Markdown | UnstructuredMarkdownLoader | MarkdownTextSplitter |
| CSV | CSVLoader | N/A (already structured) |

---

## 🎯 Practical Exercise

```python
"""
Exercise: Build a Document Processing Pipeline

Create a system that:
1. Loads all PDFs from a directory
2. Splits them into appropriate chunks
3. Extracts key information from each chunk
4. Saves results with metadata

Requirements:
- Use DirectoryLoader with PyPDFLoader
- Use RecursiveCharacterTextSplitter
- Extract: title, main topics, summary
- Save to JSON with source tracking

Test with at least 3 sample PDF files.
"""

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import json

class ChunkAnalysis(BaseModel):
    main_topic: str = Field(description="Main topic of the chunk")
    key_points: list[str] = Field(description="3-5 key points")
    summary: str = Field(description="One sentence summary")

def process_documents(directory_path: str):
    # TODO: Load PDFs from directory
    # TODO: Split into chunks
    # TODO: Process each chunk with LLM
    # TODO: Save results to JSON
    pass

# Test
# process_documents("./pdf_files")
```

---

## ✅ Key Takeaways

1. **Document Loaders extract text** from various sources (PDF, Word, Web, etc.)
2. **DirectoryLoader processes multiple files** at once
3. **Text Splitters break documents into chunks** for LLM processing
4. **RecursiveCharacterTextSplitter is best** for general use
5. **Use language-specific splitters** for code
6. **Chunk size: 500-1000 characters** is usually good
7. **Overlap: 10-20%** maintains context
8. **split_documents preserves metadata** (better than split_text)
9. **Consider token limits** when setting chunk size
10. **Semantic chunking** is more intelligent but slower

---

## 📝 Understanding Check

1. What's the difference between PyPDFLoader and TextLoader?
2. Why use RecursiveCharacterTextSplitter over CharacterTextSplitter?
3. What's the purpose of chunk_overlap?
4. When would you use PythonCodeTextSplitter?

**Ready for the next section on Vector Stores and Embeddings?** This is where we store and search through chunks efficiently! Or would you like to:
- See the exercise solution?
- Practice more with loaders and splitters?
- Ask questions about specific document types?