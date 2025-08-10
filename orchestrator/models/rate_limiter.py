"""
Rate limiting and retry logic for model interactions.

Provides rate limiting, exponential backoff, and retry mechanisms
for different model providers.
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Callable, Any, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

from ..core.exceptions import ModelError
from .types import ModelProvider

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategies for failed requests."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    
    requests_per_minute: int = 60
    requests_per_hour: int = 3600
    tokens_per_minute: int = 90000
    tokens_per_hour: int = 1000000
    concurrent_requests: int = 10
    
    def __post_init__(self):
        """Validate configuration."""
        if self.requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        if self.tokens_per_minute <= 0:
            raise ValueError("tokens_per_minute must be positive")


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retry_on_errors: list = field(default_factory=lambda: [
        "rate_limit_exceeded",
        "timeout",
        "connection_error",
        "server_error"
    ])
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.base_delay <= 0:
            raise ValueError("base_delay must be positive")


@dataclass
class RequestRecord:
    """Record of a request for rate limiting."""
    
    timestamp: float
    tokens_used: int = 0
    
    @property
    def age_seconds(self) -> float:
        """Get age of request in seconds."""
        return time.time() - self.timestamp


class RateLimiter:
    """Rate limiter for model API requests."""
    
    def __init__(self, config: RateLimitConfig):
        """Initialize rate limiter with configuration."""
        self.config = config
        self.request_history: deque = deque()
        self.token_history: deque = deque()
        self.concurrent_requests = 0
        self.semaphore = asyncio.Semaphore(config.concurrent_requests)
        self._lock = asyncio.Lock()
    
    async def acquire(self, estimated_tokens: int = 0) -> None:
        """
        Acquire permission to make a request.
        
        Args:
            estimated_tokens: Estimated tokens for the request
            
        Raises:
            ModelError: If rate limit cannot be satisfied
        """
        async with self._lock:
            await self._wait_for_rate_limit(estimated_tokens)
            await self.semaphore.acquire()
            
            # Record the request
            now = time.time()
            self.request_history.append(RequestRecord(now, estimated_tokens))
            self.concurrent_requests += 1
    
    async def release(self, actual_tokens: int = 0) -> None:
        """
        Release a request slot.
        
        Args:
            actual_tokens: Actual tokens used in the request
        """
        async with self._lock:
            self.concurrent_requests -= 1
            self.semaphore.release()
            
            # Update token usage if we have actual count
            if actual_tokens > 0 and self.request_history:
                # Update the most recent request
                self.request_history[-1].tokens_used = actual_tokens
    
    async def _wait_for_rate_limit(self, estimated_tokens: int) -> None:
        """Wait until rate limits allow the request."""
        max_wait_time = 300  # 5 minutes max wait
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            self._cleanup_old_records()
            
            # Check request rate limits
            requests_last_minute = self._count_requests_in_window(60)
            requests_last_hour = self._count_requests_in_window(3600)
            
            # Check token rate limits
            tokens_last_minute = self._count_tokens_in_window(60)
            tokens_last_hour = self._count_tokens_in_window(3600)
            
            # Check if we can proceed
            can_proceed = (
                requests_last_minute < self.config.requests_per_minute and
                requests_last_hour < self.config.requests_per_hour and
                tokens_last_minute + estimated_tokens <= self.config.tokens_per_minute and
                tokens_last_hour + estimated_tokens <= self.config.tokens_per_hour
            )
            
            if can_proceed:
                return
            
            # Calculate wait time
            wait_time = self._calculate_wait_time(
                requests_last_minute, requests_last_hour,
                tokens_last_minute, tokens_last_hour,
                estimated_tokens
            )
            
            logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
            await asyncio.sleep(min(wait_time, 10))  # Max 10s per iteration
        
        raise ModelError("Rate limit wait timeout exceeded")
    
    def _cleanup_old_records(self) -> None:
        """Remove old records outside the tracking window."""
        now = time.time()
        cutoff_time = now - 3600  # Keep 1 hour of history
        
        while self.request_history and self.request_history[0].timestamp < cutoff_time:
            self.request_history.popleft()
    
    def _count_requests_in_window(self, window_seconds: int) -> int:
        """Count requests in the given time window."""
        now = time.time()
        cutoff_time = now - window_seconds
        
        return sum(1 for record in self.request_history 
                  if record.timestamp >= cutoff_time)
    
    def _count_tokens_in_window(self, window_seconds: int) -> int:
        """Count tokens used in the given time window."""
        now = time.time()
        cutoff_time = now - window_seconds
        
        return sum(record.tokens_used for record in self.request_history 
                  if record.timestamp >= cutoff_time)
    
    def _calculate_wait_time(
        self, 
        requests_minute: int, 
        requests_hour: int,
        tokens_minute: int, 
        tokens_hour: int, 
        estimated_tokens: int
    ) -> float:
        """Calculate how long to wait before next request."""
        wait_times = []
        
        # Request rate waits
        if requests_minute >= self.config.requests_per_minute:
            # Wait until oldest request in minute window expires
            oldest_in_minute = min(
                record.timestamp for record in self.request_history
                if record.age_seconds <= 60
            )
            wait_times.append(60 - (time.time() - oldest_in_minute))
        
        if requests_hour >= self.config.requests_per_hour:
            oldest_in_hour = min(
                record.timestamp for record in self.request_history
                if record.age_seconds <= 3600
            )
            wait_times.append(3600 - (time.time() - oldest_in_hour))
        
        # Token rate waits
        if tokens_minute + estimated_tokens > self.config.tokens_per_minute:
            # Find when enough tokens will be available
            sorted_records = sorted(
                [r for r in self.request_history if r.age_seconds <= 60],
                key=lambda x: x.timestamp
            )
            
            cumulative_tokens = 0
            for record in sorted_records:
                cumulative_tokens += record.tokens_used
                if tokens_minute - cumulative_tokens + estimated_tokens <= self.config.tokens_per_minute:
                    wait_times.append(60 - record.age_seconds)
                    break
        
        return max(wait_times) if wait_times else 1.0
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current rate limit usage statistics."""
        self._cleanup_old_records()
        
        return {
            "requests_last_minute": self._count_requests_in_window(60),
            "requests_last_hour": self._count_requests_in_window(3600),
            "tokens_last_minute": self._count_tokens_in_window(60),
            "tokens_last_hour": self._count_tokens_in_window(3600),
            "concurrent_requests": self.concurrent_requests,
            "limits": {
                "requests_per_minute": self.config.requests_per_minute,
                "requests_per_hour": self.config.requests_per_hour,
                "tokens_per_minute": self.config.tokens_per_minute,
                "tokens_per_hour": self.config.tokens_per_hour,
                "concurrent_requests": self.config.concurrent_requests
            }
        }


class RetryHandler:
    """Handles retry logic for failed requests."""
    
    def __init__(self, config: RetryConfig):
        """Initialize retry handler with configuration."""
        self.config = config
    
    async def execute_with_retry(
        self,
        func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a function with retry logic.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
            
        Raises:
            ModelError: If all retries are exhausted
        """
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return await func(*args, **kwargs)
                
            except Exception as e:
                last_error = e
                error_type = self._classify_error(e)
                
                if not self._should_retry(error_type, attempt):
                    logger.error(f"Not retrying error type '{error_type}' on attempt {attempt + 1}")
                    raise e
                
                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt, error_type)
                    logger.warning(
                        f"Attempt {attempt + 1} failed with {error_type}, "
                        f"retrying in {delay:.1f}s: {str(e)[:100]}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_retries + 1} attempts failed")
        
        raise ModelError(f"Request failed after {self.config.max_retries + 1} attempts: {last_error}")
    
    def _classify_error(self, error: Exception) -> str:
        """Classify error type for retry decisions."""
        error_str = str(error).lower()
        
        if "rate limit" in error_str or "429" in error_str:
            return "rate_limit_exceeded"
        elif "timeout" in error_str:
            return "timeout"
        elif "connection" in error_str or "network" in error_str:
            return "connection_error"
        elif "500" in error_str or "502" in error_str or "503" in error_str:
            return "server_error"
        elif "401" in error_str or "403" in error_str:
            return "auth_error"
        elif "400" in error_str:
            return "client_error"
        else:
            return "unknown_error"
    
    def _should_retry(self, error_type: str, attempt: int) -> bool:
        """Determine if we should retry based on error type and attempt."""
        if attempt >= self.config.max_retries:
            return False
        
        return error_type in self.config.retry_on_errors
    
    def _calculate_delay(self, attempt: int, error_type: str) -> float:
        """Calculate delay before next retry."""
        if self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (2 ** attempt)
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * (attempt + 1)
        else:  # FIXED_DELAY
            delay = self.config.base_delay
        
        # Special handling for rate limits
        if error_type == "rate_limit_exceeded":
            delay = max(delay, 60)  # At least 1 minute for rate limits
        
        return min(delay, self.config.max_delay)


class ModelRateLimitManager:
    """Manages rate limiting for different model providers."""
    
    def __init__(self):
        """Initialize rate limit manager."""
        self.rate_limiters: Dict[ModelProvider, RateLimiter] = {}
        self.retry_handlers: Dict[ModelProvider, RetryHandler] = {}
        self._setup_default_configs()
    
    def _setup_default_configs(self) -> None:
        """Set up default rate limiting configurations."""
        # OpenAI rate limits (conservative estimates)
        openai_config = RateLimitConfig(
            requests_per_minute=50,
            requests_per_hour=3000,
            tokens_per_minute=80000,
            tokens_per_hour=800000,
            concurrent_requests=8
        )
        
        # Ollama rate limits (more generous for local)
        ollama_config = RateLimitConfig(
            requests_per_minute=120,
            requests_per_hour=7200,
            tokens_per_minute=200000,
            tokens_per_hour=2000000,
            concurrent_requests=4  # Limited by local resources
        )
        
        # Retry configurations
        openai_retry = RetryConfig(
            max_retries=3,
            base_delay=1.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        )
        
        ollama_retry = RetryConfig(
            max_retries=2,
            base_delay=0.5,
            strategy=RetryStrategy.LINEAR_BACKOFF
        )
        
        # Set up rate limiters and retry handlers
        self.rate_limiters[ModelProvider.OPENAI] = RateLimiter(openai_config)
        self.rate_limiters[ModelProvider.OLLAMA] = RateLimiter(ollama_config)
        
        self.retry_handlers[ModelProvider.OPENAI] = RetryHandler(openai_retry)
        self.retry_handlers[ModelProvider.OLLAMA] = RetryHandler(ollama_retry)
    
    def get_rate_limiter(self, provider: ModelProvider) -> RateLimiter:
        """Get rate limiter for a provider."""
        return self.rate_limiters[provider]
    
    def get_retry_handler(self, provider: ModelProvider) -> RetryHandler:
        """Get retry handler for a provider."""
        return self.retry_handlers[provider]
    
    async def execute_with_limits(
        self,
        provider: ModelProvider,
        func: Callable[..., Awaitable[Any]],
        estimated_tokens: int = 0,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a function with rate limiting and retry logic.
        
        Args:
            provider: Model provider
            func: Function to execute
            estimated_tokens: Estimated tokens for rate limiting
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        rate_limiter = self.get_rate_limiter(provider)
        retry_handler = self.get_retry_handler(provider)
        
        async def wrapped_func():
            await rate_limiter.acquire(estimated_tokens)
            try:
                result = await func(*args, **kwargs)
                # Extract actual token usage if available
                actual_tokens = 0
                if hasattr(result, 'usage') and result.usage:
                    actual_tokens = getattr(result.usage, 'total_tokens', 0)
                await rate_limiter.release(actual_tokens)
                return result
            except Exception as e:
                await rate_limiter.release()
                raise e
        
        return await retry_handler.execute_with_retry(wrapped_func)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all providers."""
        stats = {}
        for provider, limiter in self.rate_limiters.items():
            stats[provider.value] = limiter.get_current_usage()
        return stats