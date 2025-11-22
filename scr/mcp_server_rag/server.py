"""
Professional RAG Anything MCP Server

A comprehensive MCP server implementation for RAG Anything framework,
providing document processing and multimodal querying capabilities.
"""


import argparse
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastmcp import FastMCP, Context
from dotenv import load_dotenv

from mcp_server_rag.config import MCSConfig
from mcp_server_rag.connector import RAGAnythingConnector
from mcp_server_rag.llm_functions import (
    create_llm_model_func,
    create_vision_model_func,
    create_embedding_func,
)
from mcp_server_rag.tools import (
    process_document_tool,
    process_folder_tool,
    query_tool,
    query_with_multimodal_tool,
    get_config_tool,
    get_processor_status_tool,
    get_knowledge_base_stats_tool,
    list_supported_formats_tool,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Context Management
# ============================================================================


class RAGAnythingContext:
    """Context for RAG Anything operations"""

    def __init__(self, connector: RAGAnythingConnector):
        self.connector = connector


@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[RAGAnythingContext]:
    """Manage server lifecycle"""
    logger.info("Starting RAG Anything MCP Server...")

    # Load configuration
    config = MCSConfig.from_env()
    logger.info(f"Configuration loaded")
    logger.info(f"  LLM Provider: {config.llm.provider}")
    logger.info(f"  LLM Model: {config.llm.model}")
    logger.info(f"  RAG Parser: {config.rag.parser}")

    try:
        # Create LLM function
        logger.info("Creating LLM model function...")
        llm_model_func = create_llm_model_func(config)

        # Create vision model function
        logger.info("Creating vision model function...")
        vision_model_func = create_vision_model_func(config)

        # Create embedding function
        logger.info("Creating embedding function...")
        embedding_func = create_embedding_func(config)

        # Create connector with all functions
        logger.info("Creating RAGAnything connector...")
        connector = RAGAnythingConnector(
            config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        # Initialize RAGAnything
        logger.info("Initializing RAGAnything...")
        init_result = await connector.initialize()
        if not init_result.get("success"):
            error_msg = init_result.get('error', 'Unknown error')
            logger.error(f"Failed to initialize: {error_msg}")
            raise RuntimeError(f"Initialization failed: {error_msg}")

        logger.info("RAG Anything initialized successfully")
        logger.info(f"  Working Directory: {init_result.get('working_dir')}")
        logger.info(f"  Parser: {init_result.get('parser')}")

        try:
            yield RAGAnythingContext(connector)
        finally:
            logger.info("Shutting down RAG Anything MCP Server...")
            await connector.close()
            logger.info("Server shutdown complete")

    except Exception as e:
        logger.error(f"Error during server lifecycle: {e}", exc_info=True)
        raise


# ============================================================================
# MCP Server Setup
# ============================================================================

mcp = FastMCP(name="API企业级RAG", lifespan=server_lifespan)


# ============================================================================
# MCP Tools
# ============================================================================


@mcp.tool()
async def process_document(
    file_path: str,
    parse_method: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """
    Process a single document and insert into knowledge base.

    Args:
        file_path: Path to the document file
        parse_method: Parse method (auto, ocr, txt)

    Returns:
        Processing result with status and details
    """
    connector = ctx.request_context.lifespan_context.connector
    return await process_document_tool(connector, file_path, parse_method)

@mcp.tool()
async def process_folder(
    folder_path: str,
    recursive: bool = True,
    parse_method: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """
    Process all documents in a folder.

    Args:
        folder_path: Path to the folder
        recursive: Whether to process subfolders
        parse_method: Parse method

    Returns:
        Processing result with status and details
    """
    connector = ctx.request_context.lifespan_context.connector
    return await process_folder_tool(connector, folder_path, recursive, parse_method)


@mcp.tool()
async def query(
    query_text: str,
    mode: str = "hybrid",
    top_k: int = 10,
    ctx: Context = None,
) -> str:
    """
    Execute a text query on the knowledge base.

    Args:
        query_text: The query text
        mode: Query mode (local, global, hybrid, naive, mix, bypass)
        top_k: Number of top results to return

    Returns:
        Query result with answer and sources
    """
    connector = ctx.request_context.lifespan_context.connector
    return await query_tool(connector, query_text, mode, top_k)


@mcp.tool()
async def query_with_multimodal(
    query_text: str,
    multimodal_content: Optional[str] = None,
    mode: str = "hybrid",
    ctx: Context = None,
) -> str:
    """
    Execute a multimodal query (text + images/tables/equations).

    Args:
        query_text: The query text
        multimodal_content: JSON string of multimodal content list
        mode: Query mode

    Returns:
        Query result with answer and sources
    """
    connector = ctx.request_context.lifespan_context.connector

    # Parse multimodal content if provided
    multimodal_list = None
    if multimodal_content:
        try:
            multimodal_list = json.loads(multimodal_content)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in multimodal_content: {e}")
            return json.dumps({
                "success": False,
                "error": "Invalid JSON in multimodal_content",
                "details": str(e)
            })

    return await query_with_multimodal_tool(
        connector, query_text, multimodal_list, mode
    )


@mcp.tool()
async def get_config(ctx: Context = None) -> str:
    """
    Get current configuration.

    Returns:
        Configuration details as JSON
    """
    connector = ctx.request_context.lifespan_context.connector
    return await get_config_tool(connector)


@mcp.tool()
async def get_processor_status(ctx: Context = None) -> str:
    """
    Get processor status and information.

    Returns:
        Processor status and available processors
    """
    connector = ctx.request_context.lifespan_context.connector
    return await get_processor_status_tool(connector)


@mcp.tool()
async def get_knowledge_base_stats(ctx: Context = None) -> str:
    """
    Get knowledge base statistics.

    Returns:
        Knowledge base statistics
    """
    connector = ctx.request_context.lifespan_context.connector
    return await get_knowledge_base_stats_tool(connector)


@mcp.tool()
async def list_supported_formats(ctx: Context = None) -> str:
    """
    List supported file formats.

    Returns:
        List of supported formats and multimodal capabilities
    """
    connector = ctx.request_context.lifespan_context.connector
    return await list_supported_formats_tool(connector)


# ============================================================================
# Main Entry Point
# ============================================================================


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="RAG Anything MCP Server")
    parser.add_argument(
        "--sse", action="store_true", default=True, help="Enable SSE mode"
    )
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    return parser.parse_args()


def main():
    """Main entry point"""
    load_dotenv()
    args = parse_arguments()

    logger.info(f"Starting server on {args.host}:{args.port}")

    if args.sse:
        mcp.run(transport="sse", port=args.port, host=args.host)
    else:
        mcp.run()


if __name__ == "__main__":
    main()

