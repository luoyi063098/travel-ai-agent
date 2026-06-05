from __future__ import annotations

import asyncio
import logging

from openai import (
    AsyncOpenAI,
    APIError,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT,
)

logger = logging.getLogger("travel_agent.llm")

llm_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    timeout=LLM_TIMEOUT,
    max_retries=0,
)


async def chat(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    last_error = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            resp = await llm_client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except RateLimitError as e:
            last_error = e
            wait = 2 ** attempt
            logger.warning("Rate limited, retrying in %ds (attempt %d/%d)", wait, attempt + 1, LLM_MAX_RETRIES)
            await asyncio.sleep(wait)
        except (APIConnectionError, APITimeoutError) as e:
            last_error = e
            wait = 2 ** attempt
            logger.warning("Connection/timeout error, retrying in %ds (attempt %d/%d)", wait, attempt + 1, LLM_MAX_RETRIES)
            await asyncio.sleep(wait)
        except APIError as e:
            last_error = e
            if e.status_code and e.status_code >= 500:
                wait = 2 ** attempt
                logger.warning("Server error %s, retrying in %ds", e.status_code, wait)
                await asyncio.sleep(wait)
            else:
                raise

    logger.error("LLM call failed after %d retries: %s", LLM_MAX_RETRIES, last_error)
    raise last_error


async def chat_stream(messages: list[dict], temperature: float = 0.7, max_tokens: int = 4096):
    stream = await llm_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
