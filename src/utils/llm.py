"""LLM client utilities for OpenAI and Anthropic."""
from typing import Optional, List, Dict, Any
import openai
from anthropic import Anthropic
from src.config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    MODEL_NAME,
)


class LLMClient:
    """Unified LLM client for OpenAI and Anthropic."""
    
    def __init__(self):
        self.provider = LLM_PROVIDER
        self.model = MODEL_NAME
        
        if self.provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set")
            self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        elif self.provider == "anthropic":
            if not ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")
    
    def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
    ) -> str:
        """Make an LLM call and return the response."""
        if self.provider == "openai":
            # OpenAI format
            msgs = messages.copy()
            if system:
                msgs.insert(0, {"role": "system", "content": system})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        
        elif self.provider == "anthropic":
            # Anthropic format
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or 4096,
                temperature=temperature,
                system=system or "",
                messages=messages,
            )
            return response.content[0].text
    
    def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
    ):
        """Stream LLM response (generator)."""
        if self.provider == "openai":
            msgs = messages.copy()
            if system:
                msgs.insert(0, {"role": "system", "content": system})
            
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        elif self.provider == "anthropic":
            stream = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or 4096,
                temperature=temperature,
                system=system or "",
                messages=messages,
                stream=True,
            )
            for event in stream:
                if event.type == "content_block_delta":
                    yield event.delta.text


# Global singleton
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the global LLM client."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client

