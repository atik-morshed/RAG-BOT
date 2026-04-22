from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    rag_api_key: str = Field(default="dev-secret", alias="RAG_API_KEY")

    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.1:8b", alias="OLLAMA_MODEL")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")
    openrouter_model: str = Field(
        default="meta-llama/llama-3.1-8b-instruct:free",
        alias="OPENROUTER_MODEL",
    )
    openrouter_site_url: str = Field(default="", alias="OPENROUTER_SITE_URL")
    openrouter_app_name: str = Field(default="rag-chatbot-portfolio", alias="OPENROUTER_APP_NAME")

    chroma_mode: str = Field(default="http", alias="CHROMA_MODE")
    chroma_host: str = Field(default="localhost", alias="CHROMA_HOST")
    chroma_port: int = Field(default=8000, alias="CHROMA_PORT")
    chroma_collection: str = Field(default="documents", alias="CHROMA_COLLECTION")
    chroma_persist_dir: str = Field(default="./chroma_data", alias="CHROMA_PERSIST_DIR")

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        alias="RERANK_MODEL",
    )

    top_k: int = Field(default=5, alias="TOP_K")
    use_hybrid: bool = Field(default=True, alias="USE_HYBRID")
    use_rerank: bool = Field(default=True, alias="USE_RERANK")

    query_log_path: str = Field(default="logs/query_log.jsonl", alias="QUERY_LOG_PATH")


@lru_cache
def get_settings() -> Settings:
    return Settings()
