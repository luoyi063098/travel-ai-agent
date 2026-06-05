"""
大语言模型调用封装模块。
提供带指数退避重试的 chat 方法和流式输出的 chat_stream 方法。
"""

from __future__ import annotations                      # 启用类型注解的延迟求值

import asyncio
import logging

from openai import (                                     # OpenAI 兼容 SDK
    AsyncOpenAI,                                         # 异步 OpenAI 客户端
    APIError,                                            # API 通用错误
    APIConnectionError,                                  # 网络连接错误
    RateLimitError,                                      # 速率限制错误（请求过频）
    APITimeoutError,                                     # 请求超时错误
)

from config import (                                     # 从全局配置导入
    DEEPSEEK_API_KEY,                                    # API 密钥
    DEEPSEEK_BASE_URL,                                   # API 基础地址
    DEEPSEEK_MODEL,                                      # 模型名称
    LLM_MAX_RETRIES,                                     # 最大重试次数
    LLM_TIMEOUT,                                         # 请求超时秒数
)

logger = logging.getLogger("travel_agent.llm")           # 获取本模块专用的日志记录器

# 创建异步 OpenAI 客户端实例（线程安全，可复用）
llm_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,                            # 设置 API 密钥
    base_url=DEEPSEEK_BASE_URL,                          # 设置 API 基础地址
    timeout=LLM_TIMEOUT,                                 # 设置请求超时时间
    max_retries=0,                                       # 关闭 SDK 内置重试，由本模块自行控制重试逻辑
)


async def chat(
    messages: list[dict],                                # 对话历史消息列表，格式为 [{"role": "...", "content": "..."}]
    temperature: float = 0.7,                            # 生成多样性参数，值越高输出越随机
    max_tokens: int = 4096,                              # 最大生成 Token 数
) -> str:
    """
    调用大模型进行非流式对话，带指数退避重试机制。

    返回模型生成的文本内容（字符串）。
    """
    last_error = None                                    # 记录最后一次异常，所有重试用尽后抛出
    for attempt in range(LLM_MAX_RETRIES):               # 按配置的最大重试次数进行循环
        try:
            # 发起异步 API 调用，请求模型生成回复
            resp = await llm_client.chat.completions.create(
                model=DEEPSEEK_MODEL,                    # 使用的模型名称
                messages=messages,                        # 完整的对话消息历史
                temperature=temperature,                  # 生成温度参数
                max_tokens=max_tokens,                    # 最大 Token 上限
            )
            # 提取并返回首个候选回复的文本内容
            return resp.choices[0].message.content or ""

        except RateLimitError as e:
            # 触发速率限制（通常由于 API 调用频率过高）
            last_error = e
            wait = 2 ** attempt                           # 指数退避：第 0 次等 1s，第 1 次等 2s，第 2 次等 4s ...
            logger.warning("Rate limited, retrying in %ds (attempt %d/%d)", wait, attempt + 1, LLM_MAX_RETRIES)
            await asyncio.sleep(wait)                     # 异步等待，不阻塞事件循环

        except (APIConnectionError, APITimeoutError) as e:
            # 网络连接失败或请求超时（可重试的网络级错误）
            last_error = e
            wait = 2 ** attempt
            logger.warning("Connection/timeout error, retrying in %ds (attempt %d/%d)", wait, attempt + 1, LLM_MAX_RETRIES)
            await asyncio.sleep(wait)

        except APIError as e:
            # API 服务端返回的错误（如 500 Internal Server Error）
            last_error = e
            if e.status_code and e.status_code >= 500:
                # 仅对 5xx 服务端错误进行重试，4xx 客户端错误不重试
                wait = 2 ** attempt
                logger.warning("Server error %s, retrying in %ds", e.status_code, wait)
                await asyncio.sleep(wait)
            else:
                # 4xx 错误（如 400 参数错误、401 认证失败）直接抛出，不重试
                raise

    # 所有重试均失败，记录错误日志并抛出最后一次异常
    logger.error("LLM call failed after %d retries: %s", LLM_MAX_RETRIES, last_error)
    raise last_error


async def chat_stream(messages: list[dict], temperature: float = 0.7, max_tokens: int = 4096):
    """
    调用大模型进行流式对话，逐 token 产出内容。

    使用 yield 关键字实现异步生成器，调用方可用 async for 逐块消费。
    """
    # 发起带 stream=True 的 API 调用，启用流式传输
    stream = await llm_client.chat.completions.create(
        model=DEEPSEEK_MODEL,                            # 使用的模型名称
        messages=messages,                                # 完整的对话消息历史
        temperature=temperature,                          # 生成温度参数
        max_tokens=max_tokens,                            # 最大 Token 上限
        stream=True,                                      # 启用流式模式，逐块返回
    )
    # 异步迭代流式响应，每次收到一个增量块
    async for chunk in stream:
        # 仅当当前块包含有效文本增量时才产出（跳过空块和特殊控制块）
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content          # 产出文本片段，由调用方拼接
