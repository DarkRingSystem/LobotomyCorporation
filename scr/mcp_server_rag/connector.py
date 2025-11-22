import os
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging
from raganything import RAGAnything
from raganything.config import RAGAnythingConfig
from mcp_server_rag.config import MCSConfig


logger = logging.getLogger(__name__)


class RAGAnythingConnector:
    """Connector for managing RAGAnything instance    """

    def __init__(
        self,
        config: MCSConfig,
        llm_model_func=None,
        vision_model_func=None,
        embedding_func=None,
    ):
        """
        Initialize RAGAnything Connector

        Args:
            config: MCSConfig instance
            llm_model_func: LLM model function
            vision_model_func: Vision model function (optional)
            embedding_func: Embedding function
        """
        self.config = config
        self.llm_model_func = llm_model_func
        self.vision_model_func = vision_model_func
        self.embedding_func = embedding_func
        self.rag_anything: Optional[RAGAnything] = None
        self._initialized = False

    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize RAGAnything instance

        Returns:
            Dict with initialization status and details
        """
        try:
            if self._initialized and self.rag_anything:
                logger.info("RAGAnything 初始化完成")
                return {"success": True, "message": "Already initialized"}

            # Create RAGAnything configuration
            rag_config = RAGAnythingConfig(
                working_dir=self.config.rag.working_dir,
                parser=self.config.rag.parser,
                parse_method=self.config.rag.parse_method,
                enable_image_processing=self.config.rag.enable_image,
                enable_table_processing=self.config.rag.enable_table,
                enable_equation_processing=self.config.rag.enable_equation,
                max_concurrent_files=self.config.rag.max_concurrent_files,
            )

            # Create RAGAnything instance
            self.rag_anything = RAGAnything(
                config=rag_config,
                llm_model_func=self.llm_model_func,
                vision_model_func=self.vision_model_func,
                embedding_func=self.embedding_func,
            )

            # Initialize LightRAG
            init_result = await self.rag_anything._ensure_lightrag_initialized()

            if not init_result.get("success"):
                error_msg = init_result.get("error", "Unknown error")
                logger.error(f"Failed to initialize LightRAG: {error_msg}")
                return {"success": False, "error": error_msg}

            self._initialized = True
            logger.info("RAGAnything initialized successfully")

            return {
                "success": True,
                "message": "RAGAnything initialized successfully",
                "working_dir": self.config.rag.working_dir,
                "parser": self.config.rag.parser,
            }

        except Exception as e:
            logger.error(f"Error initializing RAGAnything: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def process_document(
        self,
        file_path: str,
        parse_method: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a single document

        Args:
            file_path: Path to document
            parse_method: Parse method (optional)
            **kwargs: Additional arguments

        Returns:
            Processing result
        """
        if not self._initialized or not self.rag_anything:
            return {"success": False, "error": "RAGAnything not initialized"}

        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return {"success": False, "error": f"File not found: {file_path}"}

            logger.info(f"Processing document: {file_path}")

            # Process document using the correct method from RAGAnything
            result = await self.rag_anything.process_document_complete(
                str(file_path),
                parse_method=parse_method or self.config.rag.parse_method,
                **kwargs
            )

            logger.info(f"Document processed successfully: {file_path}")
            return {"success": True, "result": result, "file_path": str(file_path)}

        except Exception as e:
            logger.error(f"Error processing document: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def query(
        self,
        query_text: str,
        mode: str = "hybrid",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a query

        Args:
            query_text: Query text
            mode: Query mode (local, global, hybrid, naive, mix, bypass)
            **kwargs: Additional arguments

        Returns:
            Query result
        """
        if not self._initialized or not self.rag_anything:
            return {"success": False, "error": "RAGAnything not initialized"}

        try:
            logger.info(f"Executing query: {query_text[:100]}... (mode: {mode})")

            result = await self.rag_anything.aquery(query_text, mode=mode, **kwargs)

            logger.info("Query completed successfully")
            return {"success": True, "result": result}

        except Exception as e:
            logger.error(f"Error executing query: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def get_status(self) -> Dict[str, Any]:
        """Get connector status"""
        return {
            "initialized": self._initialized,
            "rag_anything_available": self.rag_anything is not None,
            "config": self.config.to_dict(),
        }

    async def close(self):
        """Close and cleanup resources"""
        try:
            if self.rag_anything:
                await self.rag_anything.finalize_storages()
                logger.info("RAGAnything storages finalized")
        except Exception as e:
            logger.error(f"Error closing RAGAnything: {e}")