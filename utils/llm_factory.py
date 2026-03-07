"""
War-Room Bot -- LLM Factory with 7-Provider Failover
Priority: Puter → OpenAI (paid) → Cerebras → OpenRouter → NVIDIA NIM → Gemini → Groq

Rate Limits (documented):
  - Puter:      $0.50 lifetime quota (gpt-4o-mini)
  - OpenAI:     Paid tier (user's account)
  - Cerebras:   30 req/min, free tier (Llama 3.1 8B)
  - OpenRouter:  20 req/min, 50 req/day free (Llama 3.3 70B :free)
  - NVIDIA NIM: 40 req/min, free (various open models)
  - Gemini:     Free tier (gemini-2.0-flash-lite)
  - Groq:       100K TPD free (llama-3.3-70b-versatile)
"""

import os
import time
from config import (
    LLM_TEMPERATURE,
    LLM_MODEL_GROQ, LLM_MODEL_OPENAI, LLM_MODEL_GEMMA,
    OPENAI_API_KEY, GROQ_API_KEY, GEMINI_API_KEY, PUTER_API_TOKEN,
)
from utils.logger import get_logger

logger = get_logger("llm_factory")

# ============================================
# Provider configs (all OpenAI-compatible except Gemini)
# ============================================
PUTER_BASE_URL = "https://api.puter.com/puterai/openai/v1/"
PUTER_MODEL = "gpt-4o-mini"

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
CEREBRAS_MODEL = "llama3.1-8b"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_NIM_MODEL = "meta/llama-3.3-70b-instruct"

# Read from .env
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
NVIDIA_NIM_API_KEY = os.getenv("NVIDIA_NIM_API_KEY", "")

# Minimum delay between LLM calls (seconds) to avoid 429
MIN_CALL_INTERVAL = 3.5


class FailoverLLM:
    """Wraps 7 LLM providers with automatic failover on rate limits / errors."""

    def __init__(self, role="default"):
        self.role = role
        self.providers = self._build_provider_chain()
        self.current_provider_name = "none"
        self._error_counts = {}
        self._last_call_time = 0  # Track last call for rate limiting

    def _build_provider_chain(self) -> list[tuple[str, object]]:
        """Build chain based on role."""
        chain = []

        if self.role == "parsing":
            # Prioritize free/fast models for data extraction (News, Sentiment)
            order = ["groq", "gemini", "puter", "openrouter", "cerebras", "nvidia", "openai"]
        else:
            # Prioritize heavy reasoning models for logic (Research, Risk)
            order = ["openai", "puter", "groq", "openrouter", "cerebras", "gemini", "nvidia"]

        for provider in order:
            llm = self._create_provider(provider)
            if llm:
                chain.append((provider, llm))

        if not chain:
            raise ValueError(
                "No LLM providers available. Set at least one API key."
            )
        return chain

    def _create_provider(self, name: str):
        try:
            if name == "puter" and PUTER_API_TOKEN:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=PUTER_MODEL,
                    temperature=LLM_TEMPERATURE,
                    api_key=PUTER_API_TOKEN,
                    base_url=PUTER_BASE_URL,
                )
            elif name == "openai" and OPENAI_API_KEY:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=LLM_MODEL_OPENAI,
                    temperature=LLM_TEMPERATURE,
                    api_key=OPENAI_API_KEY,
                )
            elif name == "cerebras" and CEREBRAS_API_KEY:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=CEREBRAS_MODEL,
                    temperature=LLM_TEMPERATURE,
                    api_key=CEREBRAS_API_KEY,
                    base_url=CEREBRAS_BASE_URL,
                )
            elif name == "openrouter" and OPENROUTER_API_KEY:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=OPENROUTER_MODEL,
                    temperature=LLM_TEMPERATURE,
                    api_key=OPENROUTER_API_KEY,
                    base_url=OPENROUTER_BASE_URL,
                    default_headers={
                        "HTTP-Referer": "https://war-room-bot.local",
                        "X-Title": "War-Room Day Trader",
                    },
                )
            elif name == "nvidia" and NVIDIA_NIM_API_KEY:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=NVIDIA_NIM_MODEL,
                    temperature=LLM_TEMPERATURE,
                    api_key=NVIDIA_NIM_API_KEY,
                    base_url=NVIDIA_NIM_BASE_URL,
                )
            elif name == "gemini" and GEMINI_API_KEY:
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=LLM_MODEL_GEMMA,
                    temperature=LLM_TEMPERATURE,
                    google_api_key=GEMINI_API_KEY,
                    convert_system_message_to_human=True,
                )
            elif name == "groq" and GROQ_API_KEY:
                from langchain_groq import ChatGroq
                return ChatGroq(
                    model=LLM_MODEL_GROQ,
                    temperature=LLM_TEMPERATURE,
                    api_key=GROQ_API_KEY,
                )
        except Exception as e:
            logger.warning(f"Failed to initialize {name}: {e}")
        return None

    def invoke(self, messages, **kwargs):
        """Call LLM with automatic failover + rate limiting between calls."""
        # Enforce minimum delay between calls to avoid 429
        elapsed = time.time() - self._last_call_time
        if elapsed < MIN_CALL_INTERVAL:
            wait = MIN_CALL_INTERVAL - elapsed
            time.sleep(wait)

        self._last_call_time = time.time()
        last_error = None

        for name, llm in self.providers:
            try:
                result = llm.invoke(messages, **kwargs)
                if self.current_provider_name != name:
                    logger.info(f"LLM provider: {name}")
                    self.current_provider_name = name
                self._error_counts[name] = 0
                return result
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = any(kw in error_str for kw in [
                    "rate limit", "429", "quota", "insufficient_quota",
                    "too many requests", "tokens per day", "resource_exhausted",
                    "credit", "exceeded", "throttl",
                ])
                self._error_counts[name] = self._error_counts.get(name, 0) + 1

                if is_rate_limit:
                    logger.warning(f"{name} rate limited, trying next provider...")
                else:
                    logger.warning(f"{name} error: {str(e)[:120]}, trying next...")
                last_error = e

                # Small delay before trying next provider
                time.sleep(1)
                continue

        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    def check_health(self) -> dict:
        """Quick health check -- returns which providers are available."""
        return {
            name: self._error_counts.get(name, 0)
            for name, _ in self.providers
        }


_llm_instances = {}


def get_llm(role="default"):
    if role not in _llm_instances:
        _llm_instances[role] = FailoverLLM(role=role)
        providers = [name for name, _ in _llm_instances[role].providers]
        logger.info(f"LLM failover chain ({role}): {' -> '.join(providers)}")
    return _llm_instances[role]
