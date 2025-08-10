"""
Context management utilities for handling large inputs in model interactions.

Provides context truncation, summarization, and chunking capabilities.
"""

import logging
import tiktoken
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .types import TaskType

logger = logging.getLogger(__name__)


class ContextStrategy(Enum):
    """Strategies for handling large context."""
    TRUNCATE = "truncate"
    SUMMARIZE = "summarize"
    CHUNK = "chunk"
    PRIORITIZE = "prioritize"


@dataclass
class ContextWindow:
    """Represents a context window with size limits."""
    
    max_tokens: int
    reserved_tokens: int = 500  # Reserved for system prompt and response
    
    @property
    def available_tokens(self) -> int:
        """Get available tokens for user content."""
        return self.max_tokens - self.reserved_tokens


@dataclass
class ContextChunk:
    """A chunk of context with metadata."""
    
    content: str
    token_count: int
    priority: int = 1
    chunk_type: str = "general"
    metadata: Optional[Dict[str, Any]] = None


class ContextManager:
    """Manages context for model interactions with size constraints."""
    
    def __init__(self, model_name: str = "gpt-4"):
        """
        Initialize context manager.
        
        Args:
            model_name: Name of the model for token counting
        """
        self.model_name = model_name
        self.encoding = self._get_encoding(model_name)
        
        # Context windows for different models
        self.context_windows = {
            "gpt-4": ContextWindow(max_tokens=8192),
            "gpt-4-turbo": ContextWindow(max_tokens=128000),
            "gpt-3.5-turbo": ContextWindow(max_tokens=4096),
            "qwen2.5:7b": ContextWindow(max_tokens=32768),
            "default": ContextWindow(max_tokens=4096)
        }
    
    def _get_encoding(self, model_name: str) -> tiktoken.Encoding:
        """Get the appropriate encoding for token counting."""
        try:
            if "gpt-4" in model_name.lower():
                return tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in model_name.lower():
                return tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                # Default to cl100k_base for most modern models
                return tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to get encoding for {model_name}, using default: {e}")
            return tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.warning(f"Failed to count tokens, using character approximation: {e}")
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4
    
    def get_context_window(self, model_name: Optional[str] = None) -> ContextWindow:
        """Get context window for a model."""
        model = model_name or self.model_name
        return self.context_windows.get(model, self.context_windows["default"])
    
    def prepare_context(
        self,
        context: Dict[str, Any],
        task_type: TaskType,
        model_name: Optional[str] = None,
        strategy: ContextStrategy = ContextStrategy.PRIORITIZE
    ) -> Dict[str, Any]:
        """
        Prepare context for model interaction, handling size constraints.
        
        Args:
            context: Original context dictionary
            task_type: Type of task for prioritization
            model_name: Target model name
            strategy: Strategy for handling large context
            
        Returns:
            Prepared context that fits within model limits
        """
        window = self.get_context_window(model_name)
        
        # Convert context to chunks
        chunks = self._create_context_chunks(context, task_type)
        
        # Calculate total tokens
        total_tokens = sum(chunk.token_count for chunk in chunks)
        
        if total_tokens <= window.available_tokens:
            # Context fits, return as-is
            return context
        
        logger.info(f"Context too large ({total_tokens} tokens), applying {strategy.value} strategy")
        
        # Apply strategy
        if strategy == ContextStrategy.TRUNCATE:
            return self._truncate_context(chunks, window)
        elif strategy == ContextStrategy.SUMMARIZE:
            return self._summarize_context(chunks, window, task_type)
        elif strategy == ContextStrategy.CHUNK:
            return self._chunk_context(chunks, window)
        elif strategy == ContextStrategy.PRIORITIZE:
            return self._prioritize_context(chunks, window, task_type)
        else:
            return self._truncate_context(chunks, window)
    
    def _create_context_chunks(
        self, 
        context: Dict[str, Any], 
        task_type: TaskType
    ) -> List[ContextChunk]:
        """Create context chunks with priorities based on task type."""
        chunks = []
        
        # Priority mapping for different task types
        priority_maps = {
            TaskType.PLANNING: {
                "specification": 10,
                "existing_tests": 8,
                "requirements": 9,
                "constraints": 7
            },
            TaskType.DEBUGGING: {
                "error_message": 10,
                "stack_trace": 9,
                "test_code": 8,
                "test_name": 7,
                "additional_context": 6
            },
            TaskType.DRAFTING: {
                "test_steps": 10,
                "expected_results": 9,
                "test_case_name": 8,
                "selectors": 7,
                "requirements": 6
            },
            TaskType.ANALYSIS: {
                "test_files": 10,
                "test_results": 9,
                "focus_areas": 8,
                "current_issues": 7
            },
            TaskType.GENERATION: {
                "requirements": 10,
                "specifications": 9,
                "component_type": 8,
                "constraints": 7
            }
        }
        
        priority_map = priority_maps.get(task_type, {})
        
        for key, value in context.items():
            if value is None:
                continue
            
            content = str(value)
            token_count = self.count_tokens(content)
            priority = priority_map.get(key, 5)  # Default priority
            
            chunks.append(ContextChunk(
                content=content,
                token_count=token_count,
                priority=priority,
                chunk_type=key,
                metadata={"original_key": key}
            ))
        
        # Sort by priority (highest first)
        chunks.sort(key=lambda x: x.priority, reverse=True)
        return chunks
    
    def _truncate_context(
        self, 
        chunks: List[ContextChunk], 
        window: ContextWindow
    ) -> Dict[str, Any]:
        """Truncate context to fit within window."""
        result = {}
        remaining_tokens = window.available_tokens
        
        for chunk in chunks:
            if chunk.token_count <= remaining_tokens:
                result[chunk.metadata["original_key"]] = chunk.content
                remaining_tokens -= chunk.token_count
            else:
                # Truncate this chunk to fit
                if remaining_tokens > 100:  # Only if we have reasonable space
                    truncated_content = self._truncate_text(chunk.content, remaining_tokens)
                    result[chunk.metadata["original_key"]] = truncated_content
                break
        
        return result
    
    def _prioritize_context(
        self, 
        chunks: List[ContextChunk], 
        window: ContextWindow,
        task_type: TaskType
    ) -> Dict[str, Any]:
        """Prioritize context based on task type and importance."""
        result = {}
        remaining_tokens = window.available_tokens
        
        # First pass: include high-priority chunks
        for chunk in chunks:
            if chunk.priority >= 8 and chunk.token_count <= remaining_tokens:
                result[chunk.metadata["original_key"]] = chunk.content
                remaining_tokens -= chunk.token_count
        
        # Second pass: include medium-priority chunks if space allows
        for chunk in chunks:
            if chunk.priority >= 6 and chunk.priority < 8:
                if chunk.token_count <= remaining_tokens:
                    result[chunk.metadata["original_key"]] = chunk.content
                    remaining_tokens -= chunk.token_count
                elif remaining_tokens > 200:
                    # Truncate medium-priority content
                    truncated = self._truncate_text(chunk.content, remaining_tokens)
                    result[chunk.metadata["original_key"]] = truncated
                    break
        
        # Third pass: include low-priority chunks if space allows
        for chunk in chunks:
            if chunk.priority < 6 and chunk.token_count <= remaining_tokens:
                result[chunk.metadata["original_key"]] = chunk.content
                remaining_tokens -= chunk.token_count
        
        return result
    
    def _summarize_context(
        self, 
        chunks: List[ContextChunk], 
        window: ContextWindow,
        task_type: TaskType
    ) -> Dict[str, Any]:
        """Summarize large context chunks."""
        result = {}
        remaining_tokens = window.available_tokens
        
        for chunk in chunks:
            if chunk.token_count <= remaining_tokens:
                result[chunk.metadata["original_key"]] = chunk.content
                remaining_tokens -= chunk.token_count
            else:
                # Summarize large chunks
                if chunk.priority >= 7:  # Only summarize important content
                    summary = self._create_summary(chunk.content, chunk.chunk_type)
                    summary_tokens = self.count_tokens(summary)
                    
                    if summary_tokens <= remaining_tokens:
                        result[chunk.metadata["original_key"]] = f"[SUMMARIZED] {summary}"
                        remaining_tokens -= summary_tokens
        
        return result
    
    def _chunk_context(
        self, 
        chunks: List[ContextChunk], 
        window: ContextWindow
    ) -> Dict[str, Any]:
        """Split context into multiple chunks for processing."""
        # For now, return the first chunk that fits
        # In a full implementation, this would return multiple contexts
        # for separate model calls
        return self._truncate_context(chunks, window)
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        if max_tokens <= 0:
            return ""
        
        # Encode and truncate
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate tokens and decode
        truncated_tokens = tokens[:max_tokens - 10]  # Leave some buffer
        truncated_text = self.encoding.decode(truncated_tokens)
        
        return truncated_text + "... [TRUNCATED]"
    
    def _create_summary(self, content: str, content_type: str) -> str:
        """Create a summary of content."""
        # Simple summarization - in practice, this could use another model
        lines = content.split('\n')
        
        if content_type == "test_code":
            # Extract key parts of test code
            important_lines = []
            for line in lines[:20]:  # First 20 lines
                if any(keyword in line.lower() for keyword in ['test(', 'describe(', 'expect(', 'await']):
                    important_lines.append(line.strip())
            return '\n'.join(important_lines)
        
        elif content_type == "error_message" or content_type == "stack_trace":
            # Keep first and last few lines of errors
            if len(lines) > 10:
                return '\n'.join(lines[:5] + ['...'] + lines[-3:])
            return content
        
        else:
            # Generic summarization - first paragraph or first few sentences
            if len(content) > 500:
                sentences = content.split('. ')
                summary_sentences = []
                char_count = 0
                
                for sentence in sentences:
                    if char_count + len(sentence) > 400:
                        break
                    summary_sentences.append(sentence)
                    char_count += len(sentence)
                
                return '. '.join(summary_sentences) + '.'
            
            return content
    
    def estimate_response_tokens(self, task_type: TaskType) -> int:
        """Estimate tokens needed for response based on task type."""
        estimates = {
            TaskType.PLANNING: 800,
            TaskType.DEBUGGING: 600,
            TaskType.DRAFTING: 1200,
            TaskType.ANALYSIS: 700,
            TaskType.GENERATION: 1000
        }
        
        return estimates.get(task_type, 600)
    
    def validate_context_size(
        self, 
        messages: List[Dict[str, str]], 
        model_name: Optional[str] = None
    ) -> Tuple[bool, int, int]:
        """
        Validate that messages fit within context window.
        
        Returns:
            Tuple of (fits, total_tokens, max_tokens)
        """
        window = self.get_context_window(model_name)
        
        total_tokens = 0
        for message in messages:
            total_tokens += self.count_tokens(message.get("content", ""))
        
        fits = total_tokens <= window.available_tokens
        return fits, total_tokens, window.available_tokens