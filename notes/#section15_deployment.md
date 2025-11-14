# Section 15: Deployment & Production Best Practices 🏭

Learn how to deploy LangChain applications to production with FastAPI, handle errors gracefully, optimize performance, and follow security best practices.

---

## 🎯 Production Readiness Checklist

```python
# ✅ Before going to production:

# 1. API Design
# - RESTful endpoints
# - Input validation
# - Error handling
# - Rate limiting

# 2. Security
# - API key management
# - Input sanitization
# - CORS configuration
# - Authentication/Authorization

# 3. Performance
# - Caching
# - Async operations
# - Connection pooling
# - Resource limits

# 4. Monitoring
# - Logging
# - Metrics
# - Tracing (LangSmith)
# - Alerting

# 5. Reliability
# - Error handling
# - Retry logic
# - Fallbacks
# - Circuit breakers

# 6. Cost Management
# - Token tracking
# - Budget alerts
# - Provider optimization
# - Usage analytics
```

---

## 🚀 Example 1: Basic FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# FastAPI app
app = FastAPI(title="LangChain API", version="1.0.0")

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    temperature: float = 0.7
    max_tokens: int = 500

class QueryResponse(BaseModel):
    answer: str
    model_used: str
    tokens_used: int

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

prompt = ChatPromptTemplate.from_template("""
Answer the following question clearly and concisely:

Question: {question}

Answer:
""")

chain = prompt | llm | StrOutputParser()

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "langchain-api"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the LLM."""
    try:
        # Validate input
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if len(request.question) > 1000:
            raise HTTPException(status_code=400, detail="Question too long (max 1000 chars)")
        
        # Invoke chain
        answer = chain.invoke({"question": request.question})
        
        return QueryResponse(
            answer=answer,
            model_used="gpt-4",
            tokens_used=0  # Would track with callback
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn main:app --reload
# Test with: curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d '{"question":"What is AI?"}'
```

---

## 🔒 Example 2: Authentication and Security

```python
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
import secrets
import hashlib

app = FastAPI()
security = HTTPBearer()

# Simulated API key database
VALID_API_KEYS = {
    "user1": hashlib.sha256("secret_key_1".encode()).hexdigest(),
    "user2": hashlib.sha256("secret_key_2".encode()).hexdigest(),
}

class QueryRequest(BaseModel):
    question: str
    
    class Config:
        # Prevent additional fields
        extra = "forbid"

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify API key."""
    api_key = credentials.credentials
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    
    # Check if valid
    for user_id, stored_hash in VALID_API_KEYS.items():
        if secrets.compare_digest(key_hash, stored_hash):
            return user_id
    
    raise HTTPException(
        status_code=401,
        detail="Invalid API key"
    )

def sanitize_input(text: str) -> str:
    """Sanitize user input."""
    # Remove potential injection attempts
    dangerous_patterns = ["<script>", "javascript:", "DROP TABLE"]
    
    for pattern in dangerous_patterns:
        if pattern.lower() in text.lower():
            raise HTTPException(
                status_code=400,
                detail="Invalid input detected"
            )
    
    return text.strip()

@app.post("/secure-query")
async def secure_query(
    request: QueryRequest,
    user_id: str = Depends(verify_api_key)
):
    """Secured endpoint with authentication."""
    try:
        # Sanitize input
        clean_question = sanitize_input(request.question)
        
        # Rate limiting check (implement with Redis in production)
        # if check_rate_limit(user_id):
        #     raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Process query
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4", temperature=0.7)
        answer = llm.invoke(clean_question)
        
        # Log usage
        print(f"User {user_id} query: {clean_question[:50]}...")
        
        return {
            "answer": answer.content,
            "user_id": user_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

# CORS configuration
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific origins only
    allow_credentials=True,
    allow_methods=["POST"],  # Only needed methods
    allow_headers=["*"],
)
```

---

## ⚡ Example 3: Async Operations and Performance

```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import asyncio
from typing import List
import time

app = FastAPI()

class BatchQueryRequest(BaseModel):
    questions: List[str]

class BatchQueryResponse(BaseModel):
    answers: List[str]
    total_time: float

# Async LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.7)
prompt = ChatPromptTemplate.from_template("Answer: {question}")
chain = prompt | llm

async def process_single_query(question: str) -> str:
    """Process a single query asynchronously."""
    # Use ainvoke for async
    result = await chain.ainvoke({"question": question})
    return result.content

@app.post("/batch-query", response_model=BatchQueryResponse)
async def batch_query(request: BatchQueryRequest):
    """Process multiple queries in parallel."""
    start_time = time.time()
    
    # Validate
    if len(request.questions) > 10:
        raise HTTPException(status_code=400, detail="Max 10 questions per batch")
    
    # Process in parallel using asyncio.gather
    tasks = [process_single_query(q) for q in request.questions]
    answers = await asyncio.gather(*tasks)
    
    elapsed_time = time.time() - start_time
    
    return BatchQueryResponse(
        answers=answers,
        total_time=elapsed_time
    )

# Background task example
async def process_long_task(task_id: str, question: str):
    """Long-running task processed in background."""
    await asyncio.sleep(5)  # Simulate long processing
    result = await chain.ainvoke({"question": question})
    
    # Store result (use Redis/database in production)
    print(f"Task {task_id} completed: {result.content[:50]}...")

@app.post("/async-query")
async def async_query(question: str, background_tasks: BackgroundTasks):
    """Start a long-running task in the background."""
    task_id = f"task_{int(time.time())}"
    
    # Add task to background
    background_tasks.add_task(process_long_task, task_id, question)
    
    return {
        "task_id": task_id,
        "status": "processing",
        "message": "Query is being processed in the background"
    }

# Streaming endpoint
from fastapi.responses import StreamingResponse

@app.post("/stream-query")
async def stream_query(question: str):
    """Stream LLM response token by token."""
    
    async def generate():
        async for chunk in chain.astream({"question": question}):
            yield f"data: {chunk.content}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

---

## 💾 Example 4: Caching for Performance

```python
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from functools import lru_cache
import hashlib
import json
import time

app = FastAPI()

# In-memory cache (use Redis in production)
response_cache = {}

class QueryRequest(BaseModel):
    question: str
    use_cache: bool = True

def get_cache_key(question: str, model: str) -> str:
    """Generate cache key."""
    key_string = f"{question}:{model}"
    return hashlib.md5(key_string.encode()).hexdigest()

def get_cached_response(cache_key: str):
    """Get response from cache."""
    if cache_key in response_cache:
        cached_data = response_cache[cache_key]
        # Check if cache is still valid (1 hour TTL)
        if time.time() - cached_data["timestamp"] < 3600:
            return cached_data["response"]
    return None

def set_cached_response(cache_key: str, response: str):
    """Store response in cache."""
    response_cache[cache_key] = {
        "response": response,
        "timestamp": time.time()
    }

llm = ChatOpenAI(model="gpt-4", temperature=0.7)
prompt = ChatPromptTemplate.from_template("Answer: {question}")
chain = prompt | llm

@app.post("/cached-query")
async def cached_query(request: QueryRequest):
    """Query with caching support."""
    cache_key = get_cache_key(request.question, "gpt-4")
    
    # Check cache
    if request.use_cache:
        cached_response = get_cached_response(cache_key)
        if cached_response:
            return {
                "answer": cached_response,
                "cached": True,
                "cache_key": cache_key
            }
    
    # Not in cache, query LLM
    start_time = time.time()
    result = await chain.ainvoke({"question": request.question})
    elapsed_time = time.time() - start_time
    
    # Store in cache
    set_cached_response(cache_key, result.content)
    
    return {
        "answer": result.content,
        "cached": False,
        "processing_time": elapsed_time
    }

# LRU cache for expensive computations
@lru_cache(maxsize=100)
def expensive_preprocessing(text: str) -> str:
    """Cache expensive preprocessing."""
    # Simulate expensive operation
    time.sleep(0.1)
    return text.lower().strip()

@app.post("/preprocessed-query")
async def preprocessed_query(question: str):
    """Query with cached preprocessing."""
    # This will be cached
    processed_question = expensive_preprocessing(question)
    
    result = await chain.ainvoke({"question": processed_question})
    
    return {"answer": result.content}
```

---

## 📊 Example 5: Monitoring and Logging

```python
from fastapi import FastAPI, Request
import logging
import time
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Structured logging
class StructuredLogger:
    """Structured JSON logging."""
    
    @staticmethod
    def log_request(
        method: str,
        path: str,
        status_code: int,
        duration: float,
        user_id: str = None,
        error: str = None
    ):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": round(duration * 1000, 2),
            "user_id": user_id,
            "error": error
        }
        
        if status_code >= 400:
            logger.error(json.dumps(log_data))
        else:
            logger.info(json.dumps(log_data))

structured_logger = StructuredLogger()

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    start_time = time.time()
    
    # Get user ID from header (if authenticated)
    user_id = request.headers.get("X-User-ID")
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        structured_logger.log_request(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration=duration,
            user_id=user_id
        )
        
        # Add custom headers
        response.headers["X-Process-Time"] = str(duration)
        
        return response
    
    except Exception as e:
        duration = time.time() - start_time
        
        structured_logger.log_request(
            method=request.method,
            path=request.url.path,
            status_code=500,
            duration=duration,
            user_id=user_id,
            error=str(e)
        )
        
        raise

# Metrics collection
from collections import defaultdict

class Metrics:
    """Simple metrics collector."""
    
    def __init__(self):
        self.request_count = defaultdict(int)
        self.error_count = defaultdict(int)
        self.total_latency = defaultdict(float)
    
    def record_request(self, endpoint: str, latency: float, success: bool):
        self.request_count[endpoint] += 1
        self.total_latency[endpoint] += latency
        
        if not success:
            self.error_count[endpoint] += 1
    
    def get_stats(self):
        stats = {}
        for endpoint in self.request_count:
            stats[endpoint] = {
                "total_requests": self.request_count[endpoint],
                "errors": self.error_count[endpoint],
                "avg_latency": self.total_latency[endpoint] / self.request_count[endpoint],
                "error_rate": self.error_count[endpoint] / self.request_count[endpoint]
            }
        return stats

metrics = Metrics()

@app.get("/metrics")
async def get_metrics():
    """Get service metrics."""
    return metrics.get_stats()

@app.post("/query")
async def query(question: str):
    """Monitored query endpoint."""
    start_time = time.time()
    success = True
    
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4")
        result = await llm.ainvoke(question)
        
        return {"answer": result.content}
    
    except Exception as e:
        success = False
        logger.error(f"Query failed: {str(e)}")
        raise
    
    finally:
        latency = time.time() - start_time
        metrics.record_request("/query", latency, success)
```

---

## 🔄 Example 6: Error Handling and Retry Logic

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    question: str

# Retry configuration
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    before_sleep=lambda retry_state: logger.warning(
        f"Retrying after error: {retry_state.outcome.exception()}"
    )
)
async def call_llm_with_retry(question: str):
    """Call LLM with automatic retry."""
    llm = ChatOpenAI(model="gpt-4", timeout=10)
    return await llm.ainvoke(question)

# Fallback chain
async def query_with_fallback(question: str):
    """Query with provider fallback."""
    providers = [
        ("gpt-4", ChatOpenAI(model="gpt-4")),
        ("gpt-3.5-turbo", ChatOpenAI(model="gpt-3.5-turbo")),
    ]
    
    last_error = None
    
    for name, llm in providers:
        try:
            logger.info(f"Trying {name}...")
            result = await llm.ainvoke(question)
            logger.info(f"Success with {name}")
            return result.content, name
        
        except Exception as e:
            logger.warning(f"{name} failed: {str(e)}")
            last_error = e
            continue
    
    raise Exception(f"All providers failed. Last error: {last_error}")

@app.post("/robust-query")
async def robust_query(request: QueryRequest):
    """Query with retry and fallback."""
    try:
        # Try with retry
        result = await call_llm_with_retry(request.question)
        
        return {
            "answer": result.content,
            "provider": "gpt-4",
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Retry failed: {str(e)}")
        
        # Fallback to alternative providers
        try:
            answer, provider = await query_with_fallback(request.question)
            
            return {
                "answer": answer,
                "provider": provider,
                "status": "fallback"
            }
        
        except Exception as fallback_error:
            logger.error(f"All attempts failed: {str(fallback_error)}")
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable"
            )

# Circuit breaker pattern
from datetime import datetime, timedelta

class CircuitBreaker:
    """Simple circuit breaker implementation."""
    
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func, *args, **kwargs):
        if self.state == "open":
            # Check if timeout expired
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout):
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            
            if self.state == "half-open":
                self.state = "closed"
                self.failures = 0
            
            return result
        
        except Exception as e:
            self.failures += 1
            self.last_failure_time = datetime.now()
            
            if self.failures >= self.failure_threshold:
                self.state = "open"
            
            raise

circuit_breaker = CircuitBreaker()
```

---

## 🎯 Example 7: Rate Limiting

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict
from datetime import datetime, timedelta
import time

app = FastAPI()

# In-memory rate limiter (use Redis in production)
class RateLimiter:
    """Simple token bucket rate limiter."""
    
    def __init__(self, requests_per_minute: int = 10):
        self.requests_per_minute = requests_per_minute
        self.users: Dict[str, dict] = {}
    
    def is_allowed(self, user_id: str) -> tuple[bool, dict]:
        """Check if request is allowed."""
        now = time.time()
        
        if user_id not in self.users:
            self.users[user_id] = {
                "tokens": self.requests_per_minute,
                "last_update": now
            }
        
        user_data = self.users[user_id]
        
        # Refill tokens based on time passed
        time_passed = now - user_data["last_update"]
        tokens_to_add = time_passed * (self.requests_per_minute / 60)
        user_data["tokens"] = min(
            self.requests_per_minute,
            user_data["tokens"] + tokens_to_add
        )
        user_data["last_update"] = now
        
        # Check if request allowed
        if user_data["tokens"] >= 1:
            user_data["tokens"] -= 1
            return True, {
                "remaining": int(user_data["tokens"]),
                "limit": self.requests_per_minute
            }
        else:
            retry_after = int((1 - user_data["tokens"]) * 60 / self.requests_per_minute)
            return False, {
                "remaining": 0,
                "limit": self.requests_per_minute,
                "retry_after": retry_after
            }

rate_limiter = RateLimiter(requests_per_minute=10)

async def check_rate_limit(user_id: str):
    """Dependency to check rate limit."""
    allowed, info = rate_limiter.is_allowed(user_id)
    
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {info['retry_after']} seconds",
            headers={
                "X-RateLimit-Limit": str(info["limit"]),
                "X-RateLimit-Remaining": str(info["remaining"]),
                "Retry-After": str(info["retry_after"])
            }
        )
    
    return info

@app.post("/rate-limited-query")
async def rate_limited_query(
    question: str,
    user_id: str,
    rate_info: dict = Depends(lambda: check_rate_limit("test_user"))
):
    """Rate-limited query endpoint."""
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(model="gpt-4")
    result = await llm.ainvoke(question)
    
    return {
        "answer": result.content,
        "rate_limit": rate_info
    }

# Tiered rate limiting
class TieredRateLimiter:
    """Rate limiter with different tiers."""
    
    def __init__(self):
        self.tiers = {
            "free": {"requests_per_minute": 5, "requests_per_day": 100},
            "basic": {"requests_per_minute": 20, "requests_per_day": 1000},
            "premium": {"requests_per_minute": 100, "requests_per_day": 10000}
        }
        self.user_tiers = {}  # user_id -> tier
        self.usage = {}  # user_id -> usage data
    
    def check_limit(self, user_id: str) -> bool:
        """Check if user is within limits."""
        tier = self.user_tiers.get(user_id, "free")
        limits = self.tiers[tier]
        
        # Check minute limit
        # Check daily limit
        # Return True if allowed
        return True

tiered_limiter = TieredRateLimiter()
```

---

## 🔐 Example 8: Cost Tracking and Budget Management

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
from typing import Dict
from datetime import datetime, date

app = FastAPI()

class CostTracker:
    """Track LLM usage costs."""
    
    def __init__(self, daily_budget: float = 100.0):
        self.daily_budget = daily_budget
        self.costs: Dict[str, dict] = {}  # user_id -> costs
    
    def record_usage(self, user_id: str, tokens: int, cost: float):
        """Record user usage."""
        today = date.today().isoformat()
        
        if user_id not in self.costs:
            self.costs[user_id] = {}
        
        if today not in self.costs[user_id]:
            self.costs[user_id][today] = {
                "tokens": 0,
                "cost": 0.0,
                "requests": 0
            }
        
        self.costs[user_id][today]["tokens"] += tokens
        self.costs[user_id][today]["cost"] += cost
        self.costs[user_id][today]["requests"] += 1
    
    def check_budget(self, user_id: str) -> tuple[bool, dict]:
        """Check if user is within budget."""
        today = date.today().isoformat()
        
        if user_id not in self.costs or today not in self.costs[user_id]:
            return True, {
                "spent": 0.0,
                "budget": self.daily_budget,
                "remaining": self.daily_budget
            }
        
        spent = self.costs[user_id][today]["cost"]
        remaining = self.daily_budget - spent
        
        return remaining > 0, {
            "spent": spent,
            "budget": self.daily_budget,
            "remaining": remaining
        }
    
    def get_usage_report(self, user_id: str, days: int = 7):
        """Get usage report for user."""
        if user_id not in self.costs:
            return {}
        
        return self.costs[user_id]

cost_tracker = CostTracker(daily_budget=10.0)

@app.post("/budget-aware-query")
async def budget_aware_query(question: str, user_id: str):
    """Query with budget tracking."""
    
    # Check budget
    within_budget, budget_info = cost_tracker.check_budget(user_id)
    
    if not within_budget:
        raise HTTPException(
            status_code=429,
            detail=f"Daily budget exceeded. Spent: ${budget_info['spent']:.4f}"
        )
    
    # Query with cost tracking
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    with get_openai_callback() as cb:
        result = await llm.ainvoke(question)
        
        # Record usage
        cost_tracker.record_usage(
            user_id=user_id,
            tokens=cb.total_tokens,
            cost=cb.total_cost
        )
    
    return {
        "answer": result.content,
        "usage": {
            "tokens": cb.total_tokens,
            "cost": cb.total_cost,
            "budget_remaining": budget_info["remaining"] - cb.total_cost
        }
    }

@app.get("/usage-report/{user_id}")
async def usage_report(user_id: str, days: int = 7):
    """Get usage report."""
    report = cost_tracker.get_usage_report(user_id, days)
    
    total_cost = sum(day_data["cost"] for day_data in report.values())
    total_tokens = sum(day_data["tokens"] for day_data in report.values())
    total_requests = sum(day_data["requests"] for day_data in report.values())
    
    return {
        "user_id": user_id,
        "period_days": days,
        "total_cost": total_cost,
        "total_tokens": total_tokens,
        "total_requests": total_requests,
        "daily_breakdown": report
    }
```

---

## 🎨 Example 9: Streamlit UI

```python
"""
streamlit_app.py - Simple UI for LangChain app

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
import os

# Page config
st.set_page_config(
    page_title="LangChain Chat",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 LangChain Chat Assistant")

# Sidebar configuration
with st.sidebar:
    st.header("Settings")
    
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    model = st.selectbox(
        "Model",
        ["gpt-4", "gpt-3.5-turbo"]
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )
    
    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                llm = ChatOpenAI(
                    model=model,
                    temperature=temperature
                )
                
                response = llm.invoke(prompt)
                st.markdown(response.content)
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.content
                })
            
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Display stats
with st.sidebar:
    st.markdown("---")
    st.subheader("Stats")
    st.metric("Messages", len(st.session_state.messages))
```

---

## 🐳 Example 10: Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  langchain-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LANGCHAIN_TRACING_V2=true
      - LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}
      - LANGCHAIN_PROJECT=production
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

volumes:
  redis-data:
```

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f langchain-api

# Scale
docker-compose up -d --scale langchain-api=3
```

---

## 🔥 Production Checklist

### **1. Security**
```python
# ✅ Checklist:
# - Use HTTPS in production
# - Implement authentication (OAuth2, JWT)
# - Sanitize all inputs
# - Rate limiting per user
# - API key rotation
# - CORS configuration
# - Environment variables for secrets
# - Input validation with Pydantic
```

### **2. Performance**
```python
# ✅ Checklist:
# - Async operations (ainvoke, astream)
# - Caching (Redis)
# - Connection pooling
# - Batch processing
# - CDN for static assets
# - Database indexing
# - Query optimization
```

### **3. Monitoring**
```python
# ✅ Checklist:
# - Structured logging
# - Error tracking (Sentry)
# - LangSmith tracing
# - Metrics (Prometheus)
# - Alerting (PagerDuty)
# - Health checks
# - Uptime monitoring
```

### **4. Reliability**
```python
# ✅ Checklist:
# - Retry logic with exponential backoff
# - Circuit breakers
# - Fallback providers
# - Graceful degradation
# - Database backups
# - Disaster recovery plan
```

### **5. Cost Management**
```python
# ✅ Checklist:
# - Token usage tracking
# - Daily/monthly budgets
# - Cost per user
# - Provider optimization
# - Budget alerts
# - Usage analytics
```

---

## 🎯 Final Exercise: Production-Ready API

```python
"""
Exercise: Build a Complete Production API

Create a production-ready LangChain API with:

1. FastAPI Setup
   - Multiple endpoints (query, batch, stream)
   - Request/response validation
   - API documentation

2. Security
   - API key authentication
   - Input sanitization
   - Rate limiting
   - CORS configuration

3. Performance
   - Async operations
   - Caching (in-memory or Redis)
   - Batch processing
   - Connection pooling

4. Monitoring
   - Structured logging
   - Metrics collection
   - Health checks
   - LangSmith integration

5. Error Handling
   - Retry logic
   - Provider fallback
   - Circuit breaker
   - Graceful errors

6. Cost Management
   - Token tracking
   - Budget limits
   - Usage reports

7. Deployment
   - Dockerfile
   - docker-compose.yml
   - Environment configuration
   - Health checks

Requirements:
- All endpoints must be async
- Comprehensive error handling
- Full test coverage
- Production-ready logging
- Docker deployment
- Complete documentation

Test scenarios:
- Normal operation
- Rate limit exceeded
- Budget exceeded
- Provider failure
- Invalid inputs
- Concurrent requests
"""

# TODO: Implement complete production API
# Hint: Combine concepts from all examples above
```

---

## ✅ Key Takeaways

1. **FastAPI for REST APIs** - async, fast, well-documented
2. **Security first** - authentication, sanitization, rate limiting
3. **Async everywhere** - ainvoke, astream, abatch
4. **Cache aggressively** - reduce costs and latency
5. **Monitor everything** - logs, metrics, traces
6. **Handle failures gracefully** - retries, fallbacks, circuit breakers
7. **Track costs** - tokens, budgets, usage
8. **Use Docker** - consistent deployments
9. **LangSmith in production** - observability and debugging
10. **Test thoroughly** - unit, integration, load tests

---

## 📝 Congratulations! 🎉

You've completed the comprehensive LangChain course covering:

1. ✅ Models (LLMs, Chat Models)
2. ✅ Prompts and Prompt Templates
3. ✅ Messages and Output Parsers
4. ✅ LCEL (Runnable Interface)
5. ✅ Memory Systems
6. ✅ Document Loaders and Text Splitters
7. ✅ Vector Stores and Embeddings
8. ✅ RAG (Retrieval-Augmented Generation)
9. ✅ Tools and Function Calling
10. ✅ Agents
11. ✅ Advanced RAG Techniques
12. ✅ LangGraph
13. ✅ LangSmith (Monitoring & Debugging)
14. ✅ API Integration
15. ✅ Deployment & Production Best Practices

**Next Steps:**
- Build a real project using these concepts
- Explore LangChain documentation for advanced features
- Join LangChain community (Discord, GitHub)
- Stay updated with new releases
- Contribute to open source

**Resources:**
- Official Docs: https://python.langchain.com/
- LangSmith: https://smith.langchain.com/
- GitHub: https://github.com/langchain-ai/langchain
- Discord: https://discord.gg/langchain

Happy building! 🚀