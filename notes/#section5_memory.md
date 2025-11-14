# Section 5: Memory Systems 🧠

Memory allows your LLM applications to remember past interactions, making conversations contextual and coherent.

---

## 🎯 What is Memory?

**Memory** = Storing and retrieving conversation history

Without memory:
```
User: "My name is Alice"
AI: "Nice to meet you, Alice!"
User: "What's my name?"
AI: "I don't know your name."  ❌
```

With memory:
```
User: "My name is Alice"
AI: "Nice to meet you, Alice!"
User: "What's my name?"
AI: "Your name is Alice!"  ✅
```

---

## 📋 Types of Memory in LangChain

### **1. ConversationBufferMemory** - Stores all messages
### **2. ConversationBufferWindowMemory** - Keeps last N messages
### **3. ConversationSummaryMemory** - Summarizes old conversations
### **4. ConversationSummaryBufferMemory** - Hybrid approach
### **5. ConversationEntityMemory** - Tracks entities (people, places)
### **6. ConversationKGMemory** - Knowledge graph of relationships
### **7. VectorStoreBackedMemory** - Semantic search through history

---

## 💻 Example 1: ConversationBufferMemory (Basic)

```python
from langchain.memory import ConversationBufferMemory

# Create memory
memory = ConversationBufferMemory()

# Save context
memory.save_context(
    {"input": "Hi, my name is Alice"},
    {"output": "Hello Alice! Nice to meet you."}
)

memory.save_context(
    {"input": "I work as a data scientist"},
    {"output": "That's interesting! Data science is a great field."}
)

# Load memory
print(memory.load_memory_variables({}))

# Output:
# {
#     'history': 'Human: Hi, my name is Alice\n'
#                'AI: Hello Alice! Nice to meet you.\n'
#                'Human: I work as a data scientist\n'
#                'AI: That\'s interesting! Data science is a great field.'
# }
```

---

## 🔗 Example 2: Using Memory with LCEL

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Setup
llm = ChatOpenAI(model="gpt-4", temperature=0.7)
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

# Prompt with memory placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Build chain
chain = prompt | llm | StrOutputParser()

# Conversation function
def chat(user_input: str):
    # Get memory
    memory_vars = memory.load_memory_variables({})
    
    # Invoke chain
    response = chain.invoke({
        "input": user_input,
        "chat_history": memory_vars.get("chat_history", [])
    })
    
    # Save to memory
    memory.save_context(
        {"input": user_input},
        {"output": response}
    )
    
    return response

# Test conversation
print(chat("Hi, my name is Bob"))
print("\n" + chat("What's my name?"))
print("\n" + chat("I love Python programming"))
print("\n" + chat("What do I love?"))

# Output:
# Hello Bob! Nice to meet you.
# Your name is Bob!
# That's great! Python is an excellent programming language...
# You love Python programming!
```

---

## 🪟 Example 3: ConversationBufferWindowMemory (Limited History)

```python
from langchain.memory import ConversationBufferWindowMemory

# Keep only last 2 interactions (k=2)
memory = ConversationBufferWindowMemory(k=2, return_messages=True)

# Add 4 interactions
conversations = [
    ("What's 2+2?", "2+2 equals 4."),
    ("What's 5+5?", "5+5 equals 10."),
    ("What's 10+10?", "10+10 equals 20."),
    ("What's 20+20?", "20+20 equals 40."),
]

for inp, out in conversations:
    memory.save_context({"input": inp}, {"output": out})

# Load memory - only last 2 interactions
messages = memory.load_memory_variables({})
print(f"Number of messages: {len(messages['history'])}")
print(messages['history'])

# Output: Only the last 2 conversations (10+10 and 20+20)
```

**Use Case:** Prevent context from getting too long, control costs

---

## 📝 Example 4: ConversationSummaryMemory (Summarize Old History)

```python
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

# Setup
llm = ChatOpenAI(model="gpt-4", temperature=0)
memory = ConversationSummaryMemory(llm=llm)

# Have a long conversation
conversations = [
    ("Hi, I'm planning a trip to Japan", "That sounds exciting! Japan is beautiful."),
    ("I want to visit Tokyo and Kyoto", "Great choices! Tokyo is modern, Kyoto is traditional."),
    ("What's the best time to visit?", "Spring (March-May) for cherry blossoms or Fall (Sept-Nov) for autumn colors."),
    ("How many days should I spend?", "I'd recommend at least 10-14 days to see both cities properly."),
]

for inp, out in conversations:
    memory.save_context({"input": inp}, {"output": out})

# Load memory - returns SUMMARY, not full history
summary = memory.load_memory_variables({})
print(summary['history'])

# Output (summarized):
# "The human is planning a trip to Japan, specifically Tokyo and Kyoto. 
#  The AI recommended visiting in spring or fall and suggested 10-14 days."
```

**Use Case:** Long conversations that would exceed context limits

---

## 🔀 Example 5: ConversationSummaryBufferMemory (Best of Both)

```python
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)

# Keep recent messages as-is, summarize old ones
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=100,  # When to start summarizing
    return_messages=True
)

# Long conversation
conversations = [
    ("I'm learning Python", "Great! Python is beginner-friendly."),
    ("I've learned about lists and dictionaries", "Those are fundamental data structures!"),
    ("Now I'm studying functions", "Functions help organize code effectively."),
    ("I'm confused about decorators", "Decorators modify function behavior. Let me explain..."),
    ("Can you give me an example?", "Sure! Here's a simple timing decorator..."),
]

for inp, out in conversations:
    memory.save_context({"input": inp}, {"output": out})

# Memory contains: summary of old messages + recent full messages
messages = memory.load_memory_variables({})
print(messages)

# Output: 
# Summary of first few conversations + full text of recent ones
```

**Use Case:** Balance between detail and efficiency

---

## 🏷️ Example 6: ConversationEntityMemory (Track Entities)

```python
from langchain.memory import ConversationEntityMemory
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)
memory = ConversationEntityMemory(llm=llm)

# Conversation with multiple entities
conversations = [
    ("My friend Alice works at Google", "That's interesting! Google is a great company."),
    ("She's a software engineer in the Search team", "Search is Google's core product!"),
    ("My other friend Bob works at Meta", "Meta is doing innovative work in VR/AR."),
    ("Alice and Bob are planning to start a startup together", "Exciting! What kind of startup?"),
]

for inp, out in conversations:
    memory.save_context({"input": inp}, {"output": out})

# Get information about specific entities
print("About Alice:")
print(memory.entity_store.get("Alice"))

print("\nAbout Bob:")
print(memory.entity_store.get("Bob"))

print("\nAbout Google:")
print(memory.entity_store.get("Google"))

# Output:
# About Alice: Works at Google, software engineer in Search team, planning startup with Bob
# About Bob: Works at Meta, planning startup with Alice
# About Google: Great company, core product is Search, Alice works there
```

**Use Case:** Complex conversations with many people, places, or things

---

## 🌐 Example 7: ConversationKGMemory (Knowledge Graph)

```python
from langchain.memory import ConversationKGMemory
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)
memory = ConversationKGMemory(llm=llm)

# Build relationships
conversations = [
    ("Alice is Bob's sister", "I'll remember that relationship."),
    ("Bob is a doctor", "Got it, Bob is a doctor."),
    ("Alice studies at MIT", "MIT is a prestigious university."),
    ("Bob works at Mayo Clinic", "Mayo Clinic is a renowned medical institution."),
]

for inp, out in conversations:
    memory.save_context({"input": inp}, {"output": out})

# Query relationships
print("Knowledge Graph:")
print(memory.kg)

# Get entities related to a person
print("\nFacts about Bob:")
print(memory.load_memory_variables({"input": "Tell me about Bob"}))

# Output shows relationships:
# Bob -> is -> doctor
# Bob -> works_at -> Mayo Clinic
# Bob -> sister -> Alice
```

**Use Case:** Complex relationship tracking

---

## 🔍 Example 8: VectorStore-Backed Memory (Semantic Search)

```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create vector store for memory
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(
    ["Initial setup"],  # Start with dummy text
    embedding=embeddings
)

# Create memory
memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2})
)

# Save conversations
conversations = [
    ("I love Italian food, especially pasta", "Pasta is delicious!"),
    ("My favorite color is blue", "Blue is a calming color."),
    ("I work as a data scientist in healthcare", "Healthcare data science is important."),
    ("I have a dog named Max", "Dogs are wonderful pets!"),
    ("I'm learning French", "French is a beautiful language."),
]

for inp, out in conversations:
    memory.save_context({"input": inp}, {"output": out})

# Semantic search through memory
# Query about food - should retrieve Italian food conversation
relevant_memories = memory.load_memory_variables(
    {"prompt": "What food does the user like?"}
)

print("Relevant memories:", relevant_memories)

# Output: Returns the conversation about Italian food/pasta
# (most semantically similar to the query)
```

**Use Case:** Long conversation histories where you want to find relevant past context

---

## 💬 Example 9: Complete Chatbot with Memory

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser

class ChatBot:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.7)
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Remember user preferences and context."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def chat(self, user_input: str) -> str:
        """Send a message and get a response."""
        # Load memory
        memory_vars = self.memory.load_memory_variables({})
        
        # Get response
        response = self.chain.invoke({
            "input": user_input,
            "chat_history": memory_vars.get("chat_history", [])
        })
        
        # Save to memory
        self.memory.save_context(
            {"input": user_input},
            {"output": response}
        )
        
        return response
    
    def clear_memory(self):
        """Clear conversation history."""
        self.memory.clear()

# Use the chatbot
bot = ChatBot()

print("Bot:", bot.chat("Hi! My name is Alice and I'm a Python developer."))
print("\nBot:", bot.chat("What's my name?"))
print("\nBot:", bot.chat("What do I do for work?"))
print("\nBot:", bot.chat("Can you recommend some Python libraries for me?"))

# Clear and start fresh
bot.clear_memory()
print("\n--- Memory cleared ---")
print("\nBot:", bot.chat("What's my name?"))
# Output: "I don't know your name" (memory was cleared)
```

---

## 🎨 Example 10: Multiple Memory Keys

```python
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# Separate memories for different aspects
personal_memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="personal_history"
)

work_memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="work_history"
)

# Prompt with multiple memory placeholders
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant that remembers both personal and work context separately."),
    MessagesPlaceholder(variable_name="personal_history"),
    MessagesPlaceholder(variable_name="work_history"),
    ("human", "{input}")
])

llm = ChatOpenAI(model="gpt-4")
chain = prompt | llm | StrOutputParser()

# Function to categorize and save
def chat_with_category(user_input: str, category: str):
    # Select appropriate memory
    if category == "personal":
        memory = personal_memory
    else:
        memory = work_memory
    
    # Load both memories
    personal_vars = personal_memory.load_memory_variables({})
    work_vars = work_memory.load_memory_variables({})
    
    # Get response
    response = chain.invoke({
        "input": user_input,
        "personal_history": personal_vars.get("personal_history", []),
        "work_history": work_vars.get("work_history", [])
    })
    
    # Save to appropriate memory
    memory.save_context({"input": user_input}, {"output": response})
    
    return response

# Test
print(chat_with_category("I have a dog named Max", "personal"))
print(chat_with_category("I work as a data scientist", "work"))
print(chat_with_category("What's my dog's name?", "personal"))
print(chat_with_category("What's my job?", "work"))
```

---

## 💾 Example 11: Persistent Memory (Save to Disk)

```python
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import json
import os

class PersistentChatBot:
    def __init__(self, memory_file="chat_memory.json"):
        self.memory_file = memory_file
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.7)
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Load existing memory
        self.load_memory()
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def load_memory(self):
        """Load memory from disk."""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                data = json.load(f)
                # Restore conversations
                for item in data:
                    self.memory.save_context(
                        {"input": item["input"]},
                        {"output": item["output"]}
                    )
            print(f"Loaded {len(data)} conversations from memory.")
    
    def save_memory(self):
        """Save memory to disk."""
        # Extract conversation history
        messages = self.memory.chat_memory.messages
        data = []
        
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                data.append({
                    "input": messages[i].content,
                    "output": messages[i + 1].content
                })
        
        with open(self.memory_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} conversations to memory.")
    
    def chat(self, user_input: str) -> str:
        """Chat and auto-save."""
        memory_vars = self.memory.load_memory_variables({})
        
        response = self.chain.invoke({
            "input": user_input,
            "chat_history": memory_vars.get("chat_history", [])
        })
        
        self.memory.save_context(
            {"input": user_input},
            {"output": response}
        )
        
        # Save to disk after each interaction
        self.save_memory()
        
        return response

# Test
bot = PersistentChatBot("my_chat_history.json")

print(bot.chat("My favorite color is purple"))
print(bot.chat("I love hiking"))

# Later, create new instance - memory persists!
print("\n--- New session ---")
bot2 = PersistentChatBot("my_chat_history.json")
print(bot2.chat("What's my favorite color?"))
# Output: "Your favorite color is purple"
```

---

## 📊 Memory Type Comparison

| Memory Type | Stores | Best For | Pros | Cons |
|-------------|--------|----------|------|------|
| ConversationBufferMemory | All messages | Short conversations | Simple, complete | Can get large |
| ConversationBufferWindowMemory | Last N messages | Fixed context | Controlled size | Loses old context |
| ConversationSummaryMemory | Summary | Long conversations | Compact | Loses details |
| ConversationSummaryBufferMemory | Summary + recent | Balanced needs | Best of both | Complex |
| ConversationEntityMemory | Entities | Multi-entity tracking | Organized | Requires entity extraction |
| ConversationKGMemory | Relationships | Complex relationships | Structured knowledge | Overhead |
| VectorStoreRetrieverMemory | Semantic vectors | Huge histories | Semantic search | Setup complexity |

---

## 🔥 Best Practices

### **1. Choose the Right Memory Type**
```python
# Short conversations (<10 messages)
memory = ConversationBufferMemory()

# Medium conversations (10-50 messages)
memory = ConversationBufferWindowMemory(k=10)

# Long conversations (>50 messages)
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=500)

# Semantic search needed
memory = VectorStoreRetrieverMemory(retriever=...)
```

### **2. Always Use return_messages=True with Chat Models**
```python
# ✅ For chat models
memory = ConversationBufferMemory(return_messages=True)

# ❌ Default returns string (for legacy LLMs)
memory = ConversationBufferMemory()
```

### **3. Set Appropriate Window Size**
```python
# Too small - loses context
memory = ConversationBufferWindowMemory(k=1)  # ❌

# Good balance
memory = ConversationBufferWindowMemory(k=5)  # ✅

# Consider token limits
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=2000  # Match model's context window
)
```

### **4. Clear Memory When Needed**
```python
# Start fresh conversation
memory.clear()

# Or create new memory instance per user/session
user_memories = {}
user_memories[user_id] = ConversationBufferMemory()
```

### **5. Monitor Memory Size**
```python
# Check message count
num_messages = len(memory.chat_memory.messages)
print(f"Messages in memory: {num_messages}")

# Estimate tokens (rough)
history = memory.load_memory_variables({})
token_estimate = len(str(history)) // 4  # Rough estimate
print(f"Estimated tokens: {token_estimate}")
```

---

## 🎯 Practical Exercise

```python
"""
Exercise: Build a Personal Assistant with Contextual Memory

Requirements:
1. Remember user's name, preferences, and past conversations
2. Use ConversationSummaryBufferMemory for efficiency
3. Support these commands:
   - Regular chat
   - "clear memory" - clear conversation history
   - "show memory" - display what's remembered
4. Save memory to disk between sessions

Test with:
- "My name is X, I work as Y"
- "What's my name?"
- Ask about preferences
- Close and reopen - should remember
"""

from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import json

class PersonalAssistant:
    def __init__(self, memory_file="assistant_memory.json"):
        # TODO: Initialize LLM, memory, prompt, chain
        # TODO: Load existing memory from file
        pass
    
    def chat(self, user_input: str) -> str:
        # TODO: Handle special commands (clear, show)
        # TODO: Process regular chat
        # TODO: Save memory after each interaction
        pass
    
    def save_memory(self):
        # TODO: Save to JSON file
        pass
    
    def load_memory(self):
        # TODO: Load from JSON file
        pass

# Test your assistant
assistant = PersonalAssistant()

# Test conversation with memory
# Your code here...
```

---

## ✅ Key Takeaways

1. **Memory enables contextual conversations** - remember past interactions
2. **ConversationBufferMemory** - simplest, stores everything
3. **ConversationBufferWindowMemory** - fixed-size sliding window
4. **ConversationSummaryBufferMemory** - best balance for long conversations
5. **Use return_messages=True** with chat models
6. **MessagesPlaceholder** integrates memory into prompts
7. **VectorStoreRetrieverMemory** for semantic search through history
8. **Save/load memory** for persistence across sessions
9. **Clear memory** when starting new topics/users
10. **Monitor memory size** to avoid context limit issues

---

## 📝 Understanding Check

1. What's the difference between ConversationBufferMemory and ConversationBufferWindowMemory?
2. When would you use ConversationSummaryMemory?
3. Why use return_messages=True?
4. How do you integrate memory into an LCEL chain?

**Ready for the next section on Document Loaders and Text Splitters?** This is crucial for RAG systems! Or would you like to:
- See the exercise solution?
- Practice more with memory?
- Ask questions about specific memory types?