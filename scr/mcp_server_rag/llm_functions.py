"""
LLM, Vision, and Embedding Functions for RAG Anything

This module provides factory functions for creating LLM, vision, and embedding
functions compatible with the RAG Anything framework. These functions handle
communication with LightRAG and external LLM/Vision APIs.

Functions:
    create_llm_model_func: Create LLM model function factory
    create_vision_model_func: Create vision model function factory
    create_embedding_func: Create embedding function factory
"""


import logging
from typing import Callable, Optional

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc

from .config import MCSConfig

logger = logging.getLogger(__name__)


def create_llm_model_func(config: MCSConfig) -> Callable:
    """
    Create LLM model function for RAGAnything using OpenAI API.

    This factory function creates a callable that can be used by RAGAnything
    to make LLM API calls. The returned function handles:
    - API communication with configured LLM provider
    - Conversation history management
    - System prompts
    - Temperature and token limits
    - Error handling and logging

    Note: The function returns the result directly from openai_complete_if_cache,
    which can be either a string or a coroutine. The RAGAnything framework
    handles both cases automatically.

    Args:
        config: MCSConfig instance with LLM configuration

    Returns:
        Callable: LLM model function with signature:
            (prompt: str, system_prompt: Optional[str] = None,
             history_messages: Optional[list] = None, **kwargs) -> str or coroutine

    Example:
        # >>> config = MCSConfig.from_env()
        # >>> llm_func = create_llm_model_func(config)
        # >>> response = llm_func("What is AI?")
        # >>> print(response)
    """
    def llm_model_func(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[list] = None,
        **kwargs
    ):
        """
        Call LLM API with the configured provider.

        Args:
            prompt: The main prompt/query
            system_prompt: Optional system prompt
            history_messages: Optional conversation history
            **kwargs: Additional arguments passed to the LLM

        Returns:
            str or coroutine: LLM response text (can be string or coroutine)

        Raises:
            Exception: If LLM API call fails
        """
        if history_messages is None:
            history_messages = []

        try:
            # Return the result directly - let RAGAnything framework handle coroutines
            return openai_complete_if_cache(
                config.llm.model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=config.llm.api_key,
                base_url=config.llm.base_url,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise

    return llm_model_func


def create_vision_model_func(config: MCSConfig) -> Callable:
    """
    Create vision model function for multimodal processing.

    This factory function creates a callable for processing multimodal content
    including images, tables, and equations. The returned function supports:
    - Pre-formatted messages for VLM enhanced queries
    - Single image processing with base64 encoding
    - Text fallback to LLM when no images provided
    - Proper message formatting for vision models
    - Error handling and logging

    Note: The function returns the result directly from openai_complete_if_cache,
    which can be either a string or a coroutine. The RAGAnything framework
    handles both cases automatically.

    Args:
        config: MCSConfig instance with vision configuration

    Returns:
        Callable: Vision model function with signature:
            (prompt: str, system_prompt: Optional[str] = None,
             history_messages: Optional[list] = None,
             image_data: Optional[str] = None,
             messages: Optional[list] = None, **kwargs) -> str or coroutine

    Example:
        # >>> config = MCSConfig.from_env()
        # >>> vision_func = create_vision_model_func(config)
        # >>> response = vision_func("Describe this image", image_data=base64_image)
        # >>> print(response)
    """
    def vision_model_func(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[list] = None,
        image_data: Optional[str] = None,
        messages: Optional[list] = None,
        **kwargs
    ):
        """
        Call vision model API for image/multimodal processing.

        Args:
            prompt: The main prompt/query
            system_prompt: Optional system prompt
            history_messages: Optional conversation history
            image_data: Base64 encoded image data
            messages: Pre-formatted messages for multimodal VLM
            **kwargs: Additional arguments

        Returns:
            str or coroutine: Vision model response (can be string or coroutine)

        Raises:
            Exception: If vision model API call fails
        """
        if history_messages is None:
            history_messages = []

        try:
            # If messages format is provided (for multimodal VLM enhanced query), use it directly
            if messages:
                return openai_complete_if_cache(
                    config.vision.model or config.llm.model,
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=config.vision.api_key or config.llm.api_key,
                    base_url=config.vision.base_url or config.llm.base_url,
                    **kwargs,
                )

            # Traditional single image format
            elif image_data:
                # Build messages list, filtering out None values
                messages_list = []
                if system_prompt:
                    messages_list.append({"role": "system", "content": system_prompt})

                messages_list.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                        },
                    ],
                })

                return openai_complete_if_cache(
                    config.vision.model or config.llm.model,
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages_list,
                    api_key=config.vision.api_key or config.llm.api_key,
                    base_url=config.vision.base_url or config.llm.base_url,
                    **kwargs,
                )

            # Pure text format - fallback to LLM
            else:
                llm_func = create_llm_model_func(config)
                return llm_func(prompt, system_prompt, history_messages, **kwargs)

        except Exception as e:
            logger.error(f"Error calling vision model: {e}")
            raise

    return vision_model_func


def create_embedding_func(config: MCSConfig) -> EmbeddingFunc:
    """
    Create embedding function for RAGAnything.

    This factory function creates an embedding function wrapper that can be
    used by RAGAnything for text vectorization. The function:
    - Uses Ollama or OpenAI-compatible embedding API
    - Handles configurable embedding dimensions
    - Manages token size limits
    - Provides proper error handling

    Note: Uses LightRAG's ollama_embed or openai_embed functions which properly
    handle async/sync contexts, similar to the LLM functions.

    Args:
        config: MCSConfig instance with embedding configuration

    Returns:
        EmbeddingFunc: Embedding function wrapper with configured dimensions

    Raises:
        Exception: If embedding function creation fails

    Example:
        # >>> config = MCSConfig.from_env()
        # >>> embedding_func = create_embedding_func(config)
        # >>> embeddings = embedding_func(["Hello", "World"])
    """
    try:
        # Use Ollama embedding if configured
        if config.embedding.use_ollama:
            embedding_func = EmbeddingFunc(
                embedding_dim=config.embedding.dimension,
                max_token_size=8192,
                func=lambda texts: ollama_embed(
                    texts,
                    embed_model=config.embedding.model,
                    host=config.embedding.host,
                ),
            )
            logger.info(f"Ollama embedding function created: {config.embedding.model} (dim={config.embedding.dimension})")
        else:
            # Use OpenAI-compatible embedding
            embedding_func = EmbeddingFunc(
                embedding_dim=config.embedding.dimension,
                max_token_size=8192,
                func=lambda texts: openai_embed(
                    texts,
                    model=config.embedding.model,
                    api_key=config.llm.api_key,
                    base_url=config.llm.base_url,
                ),
            )
            logger.info(f"OpenAI embedding function created: {config.embedding.model} (dim={config.embedding.dimension})")

        return embedding_func
    except Exception as e:
        logger.error(f"Error creating embedding function: {e}")
        raise

