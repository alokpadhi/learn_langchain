# Section 3: Messages and Output Parsers 🎯

Messages structure the conversation with LLMs, and Output Parsers transform raw text responses into structured, usable data.

---

## 📨 Part A: Messages

### **What are Messages?**

Messages are the building blocks of conversations with chat models. Each message has:
- **Role**: Who's speaking (system, human, AI, function)
- **Content**: What's being said
- **Metadata**: Additional information (optional)

---

## 🎭 Message Types

### **1. SystemMessage** - Instructions for the AI
```python
from langchain_core.messages import SystemMessage

msg = SystemMessage(content="You are a helpful Python programming assistant.")
print(msg)
# SystemMessage(content='You are a helpful Python programming assistant.')
```

### **2. HumanMessage** - User input
```python
from langchain_core.messages import HumanMessage

msg = HumanMessage(content="How do I reverse a list in Python?")
print(msg)
# HumanMessage(content='How do I reverse a list in Python?')
```

### **3. AIMessage** - Assistant responses
```python
from langchain_core.messages import AIMessage

msg = AIMessage(content="You can use list.reverse() or list[::-1]")
print(msg)
# AIMessage(content='You can use list.reverse() or list[::-1]')
```

### **4. FunctionMessage / ToolMessage** - Function/Tool results
```python
from langchain_core.messages import ToolMessage

msg = ToolMessage(
    content="Temperature: 72°F, Condition: Sunny",
    tool_call_id="call_123"
)
print(msg)
```

---

## 💻 Example 1: Building Conversations with Messages

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# Create a conversation history
messages = [
    SystemMessage(content="You are a concise Python tutor."),
    HumanMessage(content="What is a list comprehension?"),
    AIMessage(content="A list comprehension is a compact way to create lists: [x*2 for x in range(5)]"),
    HumanMessage(content="Can you show me filtering with it?"),
]

# Send to LLM
llm = ChatOpenAI(model="gpt-4")
response = llm.invoke(messages)

print(response.content)
# Output: "Sure! To filter, use a condition: [x for x in range(10) if x % 2 == 0] gives even numbers."
```

---

## 🔍 Example 2: Message Attributes

```python
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
response = llm.invoke([HumanMessage(content="What is 2+2?")])

# Access different attributes
print("Content:", response.content)
print("Type:", type(response))
print("Response metadata:", response.response_metadata)

# Example output:
# Content: 2+2 equals 4.
# Type: <class 'langchain_core.messages.ai.AIMessage'>
# Response metadata: {
#     'token_usage': {'completion_tokens': 7, 'prompt_tokens': 12, 'total_tokens': 19},
#     'model_name': 'gpt-4',
#     'finish_reason': 'stop'
# }

# Check token usage
tokens = response.response_metadata.get('token_usage', {})
print(f"Tokens used: {tokens.get('total_tokens', 0)}")
```

---

## 🎨 Example 3: Messages with Metadata

```python
from langchain_core.messages import HumanMessage

# Add custom metadata
msg = HumanMessage(
    content="Analyze this user behavior",
    additional_kwargs={
        "user_id": "12345",
        "session_id": "abc-xyz",
        "timestamp": "2024-01-15T10:30:00"
    }
)

print(msg.content)
print(msg.additional_kwargs)
```

---

## 📋 Part B: Output Parsers

### **Why Output Parsers?**

LLMs return unstructured text. Output Parsers convert this into:
- ✅ Structured data (JSON, dictionaries, lists)
- ✅ Type-safe Python objects
- ✅ Validated data
- ✅ Easier to work with programmatically

---

## 🔧 Types of Output Parsers

### **1. StrOutputParser** - Extract plain text (most basic)
### **2. JsonOutputParser** - Parse JSON
### **3. PydanticOutputParser** - Parse into Pydantic models (type-safe)
### **4. StructuredOutputParser** - Parse structured data with schema
### **5. CommaSeparatedListOutputParser** - Parse lists
### **6. DatetimeOutputParser** - Parse dates
### **7. EnumOutputParser** - Parse enums

---

## 💻 Example 4: StrOutputParser (Most Basic)

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Setup
llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
parser = StrOutputParser()

# Build chain with LCEL
chain = prompt | llm | parser

# Invoke
result = chain.invoke({"topic": "programming"})
print(type(result))  # <class 'str'>
print(result)  # "Why do programmers prefer dark mode? Because light attracts bugs!"
```

**Without parser:**
```python
chain = prompt | llm
result = chain.invoke({"topic": "programming"})
print(type(result))  # <class 'AIMessage'>
print(result.content)  # Need to access .content
```

---

## 📊 Example 5: JsonOutputParser

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Setup parser
parser = JsonOutputParser()

# Create prompt with format instructions
prompt = ChatPromptTemplate.from_template(
    """Extract information about the person in JSON format.
    
    {format_instructions}
    
    Text: {text}
    """
)

# Add format instructions to prompt
prompt = prompt.partial(format_instructions=parser.get_format_instructions())

# Build chain
llm = ChatOpenAI(model="gpt-4", temperature=0)
chain = prompt | llm | parser

# Test
result = chain.invoke({
    "text": "John Doe is a 30-year-old software engineer from San Francisco who loves Python."
})

print(type(result))  # <class 'dict'>
print(result)
# Output: {
#     "name": "John Doe",
#     "age": 30,
#     "occupation": "software engineer",
#     "location": "San Francisco",
#     "interests": ["Python"]
# }
```

---

## 🎯 Example 6: PydanticOutputParser (Type-Safe, Recommended!)

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List

# Define the output schema
class Person(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age in years")
    occupation: str = Field(description="Person's job title")
    location: str = Field(description="City where person lives")
    skills: List[str] = Field(description="List of technical skills")

# Create parser
parser = PydanticOutputParser(pydantic_object=Person)

# Create prompt
prompt = ChatPromptTemplate.from_template(
    """Extract person information from the text below.
    
    {format_instructions}
    
    Text: {text}
    """
)

# Add format instructions
prompt = prompt.partial(format_instructions=parser.get_format_instructions())

# Build chain
llm = ChatOpenAI(model="gpt-4", temperature=0)
chain = prompt | llm | parser

# Test
result = chain.invoke({
    "text": "Sarah Chen is 28 years old and works as a Machine Learning Engineer in Seattle. She specializes in PyTorch, TensorFlow, and NLP."
})

print(type(result))  # <class '__main__.Person'>
print(result)
# Person(
#     name='Sarah Chen',
#     age=28,
#     occupation='Machine Learning Engineer',
#     location='Seattle',
#     skills=['PyTorch', 'TensorFlow', 'NLP']
# )

# Access fields with type safety
print(result.name)  # 'Sarah Chen'
print(result.age)   # 28
print(result.skills)  # ['PyTorch', 'TensorFlow', 'NLP']
```

---

## 📝 Example 7: StructuredOutputParser

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Define schema
response_schemas = [
    ResponseSchema(name="sentiment", description="The sentiment: positive, negative, or neutral"),
    ResponseSchema(name="confidence", description="Confidence score from 0 to 1"),
    ResponseSchema(name="summary", description="A brief summary of the text"),
    ResponseSchema(name="key_points", description="List of key points (comma-separated)")
]

# Create parser
parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Create prompt
prompt = ChatPromptTemplate.from_template(
    """Analyze the following text:
    
    {format_instructions}
    
    Text: {text}
    """
)

prompt = prompt.partial(format_instructions=parser.get_format_instructions())

# Build chain
llm = ChatOpenAI(model="gpt-4", temperature=0)
chain = prompt | llm | parser

# Test
result = chain.invoke({
    "text": "I absolutely love the new Python 3.12 features! The performance improvements are incredible and the syntax enhancements make code so much cleaner."
})

print(result)
# Output: {
#     'sentiment': 'positive',
#     'confidence': 0.95,
#     'summary': 'Enthusiastic review of Python 3.12 features',
#     'key_points': 'performance improvements, syntax enhancements, cleaner code'
# }
```

---

## 📋 Example 8: CommaSeparatedListOutputParser

```python
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Create parser
parser = CommaSeparatedListOutputParser()

# Create prompt
prompt = ChatPromptTemplate.from_template(
    """List 5 popular Python web frameworks.
    
    {format_instructions}
    """
)

prompt = prompt.partial(format_instructions=parser.get_format_instructions())

# Build chain
llm = ChatOpenAI(model="gpt-4", temperature=0)
chain = prompt | llm | parser

# Test
result = chain.invoke({})

print(type(result))  # <class 'list'>
print(result)
# Output: ['Django', 'Flask', 'FastAPI', 'Pyramid', 'Tornado']
```

---

## 📅 Example 9: DatetimeOutputParser

```python
from langchain.output_parsers import DatetimeOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Create parser
parser = DatetimeOutputParser()

# Create prompt
prompt = ChatPromptTemplate.from_template(
    """Extract the date from the following text.
    
    {format_instructions}
    
    Text: {text}
    """
)

prompt = prompt.partial(format_instructions=parser.get_format_instructions())

# Build chain
llm = ChatOpenAI(model="gpt-4", temperature=0)
chain = prompt | llm | parser

# Test
result = chain.invoke({
    "text": "The conference will be held on March 15th, 2024."
})

print(type(result))  # <class 'datetime.datetime'>
print(result)  # 2024-03-15 00:00:00
print(result.strftime("%Y-%m-%d"))  # 2024-03-15
```

---

## 🎲 Example 10: EnumOutputParser

```python
from langchain.output_parsers import EnumOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from enum import Enum

# Define enum
class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

# Create parser
parser = EnumOutputParser(enum=Sentiment)

# Create prompt
prompt = ChatPromptTemplate.from_template(
    """Determine the sentiment of the text.
    
    {format_instructions}
    
    Text: {text}
    """
)

prompt = prompt.partial(format_instructions=parser.get_format_instructions())

# Build chain
llm = ChatOpenAI(model="gpt-4", temperature=0)
chain = prompt | llm | parser

# Test
result = chain.invoke({
    "text": "This product is terrible and doesn't work at all!"
})

print(type(result))  # <class '__main__.Sentiment'>
print(result)  # Sentiment.NEGATIVE
print(result.value)  # 'negative'
```

---

## 🔄 Example 11: Custom Output Parser

```python
from langchain_core.output_parsers import BaseOutputParser
from typing import List

class BulletPointParser(BaseOutputParser[List[str]]):
    """Parse output into bullet points."""
    
    def parse(self, text: str) -> List[str]:
        """Parse the output into a list of bullet points."""
        lines = text.strip().split('\n')
        bullet_points = []
        
        for line in lines:
            line = line.strip()
            # Remove bullet point markers
            if line.startswith(('- ', '* ', '• ')):
                line = line[2:].strip()
            elif line.startswith(tuple(f"{i}. " for i in range(10))):
                line = line[3:].strip()
            
            if line:
                bullet_points.append(line)
        
        return bullet_points

# Use custom parser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

parser = BulletPointParser()
prompt = ChatPromptTemplate.from_template(
    "List 5 benefits of using Python for data science."
)

llm = ChatOpenAI(model="gpt-4", temperature=0)
chain = prompt | llm | parser

result = chain.invoke({})
print(type(result))  # <class 'list'>
print(result)
# Output: [
#     'Rich ecosystem of libraries',
#     'Easy to learn and read',
#     'Strong community support',
#     'Excellent for prototyping',
#     'Integration with other languages'
# ]
```

---

## 🔥 Example 12: Complex Nested Pydantic Model

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List
from enum import Enum

# Define enums
class ExperienceLevel(str, Enum):
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"

# Define nested models
class Project(BaseModel):
    name: str = Field(description="Project name")
    description: str = Field(description="Brief project description")
    technologies: List[str] = Field(description="Technologies used")

class Developer(BaseModel):
    name: str = Field(description="Developer's name")
    level: ExperienceLevel = Field(description="Experience level")
    primary_language: str = Field(description="Primary programming language")
    years_experience: int = Field(description="Years of experience")
    projects: List[Project] = Field(description="Notable projects")
    certifications: List[str] = Field(description="Professional certifications")

# Create parser
parser = PydanticOutputParser(pydantic_object=Developer)

# Create prompt
prompt = ChatPromptTemplate.from_template(
    """Extract developer information from the text.
    
    {format_instructions}
    
    Text: {text}
    """
)

prompt = prompt.partial(format_instructions=parser.get_format_instructions())

# Build chain
llm = ChatOpenAI(model="gpt-4", temperature=0)
chain = prompt | llm | parser

# Test
result = chain.invoke({
    "text": """
    Alex Kumar is a senior software engineer with 8 years of experience, primarily working in Python.
    He's worked on two major projects: an e-commerce platform using Django, PostgreSQL, and Redis,
    and a machine learning pipeline using PyTorch, Airflow, and Kubernetes.
    He holds AWS Solutions Architect and Google Cloud Professional certifications.
    """
})

print(result)
# Developer(
#     name='Alex Kumar',
#     level=<ExperienceLevel.SENIOR: 'senior'>,
#     primary_language='Python',
#     years_experience=8,
#     projects=[
#         Project(name='E-commerce Platform', description='...', technologies=['Django', 'PostgreSQL', 'Redis']),
#         Project(name='ML Pipeline', description='...', technologies=['PyTorch', 'Airflow', 'Kubernetes'])
#     ],
#     certifications=['AWS Solutions Architect', 'Google Cloud Professional']
# )

# Access nested data
print(f"Developer: {result.name}")
print(f"Level: {result.level.value}")
print(f"First project: {result.projects[0].name}")
print(f"Technologies: {result.projects[0].technologies}")
```

---

## 🎯 Example 13: Fixing Failed Parses

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: float = Field(description="Rating from 1-10")
    review: str = Field(description="Review text")

# Base parser
base_parser = PydanticOutputParser(pydantic_object=MovieReview)

# Fixing parser - automatically fixes malformed output
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI(model="gpt-4", temperature=0)
)

# If LLM returns malformed JSON, fixing_parser will try to fix it
malformed_output = """{
    "title": "Inception",
    "rating": "9.5",  // Should be float, not string
    "review": "Mind-bending thriller"
"""  # Missing closing brace

try:
    result = fixing_parser.parse(malformed_output)
    print(result)
except Exception as e:
    print(f"Error: {e}")
```

---

## 📊 Output Parser Comparison

| Parser | Input | Output | Use Case |
|--------|-------|--------|----------|
| StrOutputParser | Text | String | Simple text extraction |
| JsonOutputParser | JSON text | Dict | Flexible JSON parsing |
| PydanticOutputParser | JSON text | Pydantic Model | Type-safe, validated data |
| StructuredOutputParser | Text | Dict | Schema-based parsing |
| CommaSeparatedListOutputParser | CSV text | List | Simple lists |
| DatetimeOutputParser | Date text | datetime | Date extraction |
| EnumOutputParser | Enum value | Enum | Limited choices |

---

## 🔥 Best Practices

### **1. Use PydanticOutputParser for Production**
```python
# ✅ Type-safe, validated, IDE support
parser = PydanticOutputParser(pydantic_object=MyModel)
```

### **2. Always Include Format Instructions**
```python
prompt = prompt.partial(format_instructions=parser.get_format_instructions())
```

### **3. Set temperature=0 for Structured Output**
```python
llm = ChatOpenAI(model="gpt-4", temperature=0)  # More consistent
```

### **4. Use OutputFixingParser in Production**
```python
fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
```

### **5. Validate Complex Schemas**
```python
class MyModel(BaseModel):
    age: int = Field(ge=0, le=150)  # Age between 0-150
    email: str = Field(pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')  # Valid email
```

---

## 🎯 Practical Exercise

```python
"""
Exercise: Build a Resume Parser

Create a chain that:
1. Takes resume text as input
2. Extracts structured information using PydanticOutputParser
3. Returns a validated Resume object

Required fields:
- name, email, phone
- skills (list)
- experience (list of jobs with: company, role, duration, description)
- education (list with: degree, institution, year)

Test with sample resume text.
"""

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# Define your models
class Job(BaseModel):
    # TODO: Define fields
    pass

class Education(BaseModel):
    # TODO: Define fields
    pass

class Resume(BaseModel):
    # TODO: Define all fields
    pass

# TODO: Create parser and chain
# TODO: Test with sample resume

sample_resume = """
John Smith
john.smith@email.com | 555-0123

SKILLS: Python, Machine Learning, TensorFlow, AWS

EXPERIENCE:
Senior ML Engineer at TechCorp (2020-2023)
- Built recommendation systems serving 1M+ users
- Reduced model inference time by 40%

Data Scientist at StartupXYZ (2018-2020)
- Developed predictive models for customer churn
- Improved retention by 15%

EDUCATION:
M.S. Computer Science, MIT, 2018
B.S. Mathematics, UC Berkeley, 2016
"""
```

---

## ✅ Key Takeaways

1. **Messages structure conversations** - SystemMessage, HumanMessage, AIMessage
2. **AIMessage has response_metadata** - access token usage, model info
3. **Output Parsers convert text to structured data**
4. **PydanticOutputParser is best for production** - type-safe, validated
5. **Always include format instructions** in prompts
6. **Use temperature=0** for consistent structured output
7. **OutputFixingParser handles malformed output**
8. **Custom parsers can handle specialized formats**

---

## 📝 Understanding Check

1. What's the difference between HumanMessage and AIMessage?
2. Why is PydanticOutputParser preferred over JsonOutputParser?
3. When would you use OutputFixingParser?

**Ready for the next section on LCEL (Runnable Interface)?** This is where things get really powerful! Or would you like to:
- See the exercise solution?
- Practice more with parsers?
- Ask questions?