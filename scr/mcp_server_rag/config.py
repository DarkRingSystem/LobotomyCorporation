import os
from dataclasses import dataclass, field
from typing import Optional, List
from dotenv import load_dotenv


@dataclass
class LLMConfig:
    """LLM 配置"""
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "deepseek"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "deepseek-chat"))
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("LLM_API_KEY"))
    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"))
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "4096")))


@dataclass
class VisionConfig:
    """视觉模型配置 (选配)"""
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("VISION_API_KEY") or os.getenv("LLM_API_KEY"))
    base_url: Optional[str] = field(default_factory=lambda: os.getenv("VISION_BASE_URL") or os.getenv("LLM_BASE_URL"))
    model: Optional[str] = field(default_factory=lambda: os.getenv("VISION_MODEL"))
    provider: Optional[str] = field(default_factory=lambda: os.getenv("VISION_PROVIDER"))
# noqa  MS80OmFIVnBZMlhsc0xubHVKM21vYVk2VGxWU1pRPT06NDRkMmQ3YTA=


@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""
    use_ollama: bool = field(default_factory=lambda: os.getenv("USE_OLLAMA_EMBEDDING", "true").lower() == "true")
    model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b"))
    host: str = field(default_factory=lambda: os.getenv("EMBEDDING_HOST", "http://localhost:11434"))
    dimension: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024")))


@dataclass
class RAGConfig:
    """RAG 配置"""
    working_dir: str = field(default_factory=lambda: os.getenv("RAG_WORKING_DIR", "./rag_storage"))

    parser: str = field(default_factory=lambda: os.getenv("RAG_PARSER", "docling"))
    parse_method: str = field(default_factory=lambda: os.getenv("RAG_PARSE_METHOD", "auto"))
    enable_image: bool = field(default_factory=lambda: os.getenv("RAG_ENABLE_IMAGE", "true").lower() == "true")
    enable_table: bool = field(default_factory=lambda: os.getenv("RAG_ENABLE_TABLE", "true").lower() == "true")
    enable_equation: bool = field(default_factory=lambda: os.getenv("RAG_ENABLE_EQUATION", "true").lower() == "true")
    load_existing: bool = field(default_factory=lambda: os.getenv("RAG_LOAD_EXISTING", "true").lower() == "true")
    max_concurrent_files: int = field(default_factory=lambda: int(os.getenv("RAG_MAX_CONCURRENT", "2")))


@dataclass
class ServerConfig:
    """服务配置"""
    host: str = field(default_factory=lambda: os.getenv("SERVER_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("SERVER_PORT", "8001")))
    sse_mode: bool = field(default_factory=lambda: os.getenv("SERVER_SSE_MODE", "true").lower() == "true")
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")


@dataclass
class MCSConfig:
    """完整 MCP 服务配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @classmethod
    def from_env(cls) -> "MCSConfig":
        """Load configuration from environment variables"""
        load_dotenv()
        return cls()

    def to_dict(self) -> dict:
        """转换配置为字典"""
        return {
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "base_url": self.llm.base_url,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
            },
            "vision": {
                "api_key": self.vision.api_key,
                "base_url": self.vision.base_url,
                "model": self.vision.model,
                "provider": self.vision.provider,
            },
            "embedding": {
                "use_ollama": self.embedding.use_ollama,
                "model": self.embedding.model,
                "host": self.embedding.host,
                "dimension": self.embedding.dimension,
            },
            "rag": {
                "working_dir": self.rag.working_dir,
                "parser": self.rag.parser,
                "parse_method": self.rag.parse_method,
                "enable_image": self.rag.enable_image,
                "enable_table": self.rag.enable_table,
                "enable_equation": self.rag.enable_equation,
                "load_existing": self.rag.load_existing,
                "max_concurrent_files": self.rag.max_concurrent_files,
            },
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "sse_mode": self.server.sse_mode,
                "debug": self.server.debug,
            },
        }

