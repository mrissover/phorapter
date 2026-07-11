"""Server settings and the factories that build a store adapter and an embedder from them.

Settings are loaded (in precedence order) from process environment variables
prefixed ``PHOROPTER_`` — nested sections use a double underscore, e.g.
``PHOROPTER_STORE__URL`` — then from an optional ``phoropter.toml`` in the
working directory, then from the defaults below. The environment always wins.

This module is part of the server layer, not the core: it may import
pydantic-settings and the framework-bound adapters. The core purity contracts
forbid the reverse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from phoropter.grid import DEFAULT_GRID
from phoropter.tokens import DEFAULT_COUNTER_ID

if TYPE_CHECKING:
    from phoropter.embed import Embedder
    from phoropter.stores import VectorStoreAdapter

__all__ = [
    "DefaultsSettings",
    "EmbedderSettings",
    "LimitsSettings",
    "LoggingSettings",
    "McpSettings",
    "ServerSettings",
    "Settings",
    "StoreSettings",
    "build_embedder",
    "build_store",
]


class ServerSettings(BaseModel):
    """HTTP server settings. ``api_key`` enables optional static bearer auth."""

    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None


class StoreSettings(BaseModel):
    """Vector store settings. ``kind`` is a name registered under ``phoropter.stores``."""

    kind: str = "qdrant"
    url: str = "http://localhost:6333"
    api_key: str | None = None
    prefix: str = "phoropter"
    search_timeout_s: float = 5.0


class EmbedderSettings(BaseModel):
    """Embedder settings. ``provider`` is a name registered under ``phoropter.embedders``."""

    provider: str = "ollama"
    model: str = "nomic-embed-text"
    base_url: str | None = None
    api_key: str | None = None
    batch_size: int = 32
    max_concurrency: int = 4


class DefaultsSettings(BaseModel):
    """Per-corpus defaults, applied exactly once at corpus creation."""

    grid_sizes: tuple[int, ...] = DEFAULT_GRID.sizes
    tokenizer: str = DEFAULT_COUNTER_ID
    top_k_per_size: int = 10
    strategy: str = "greedy_upward"


class LimitsSettings(BaseModel):
    """Hard bounds enforced at the trust boundary."""

    max_document_codepoints: int = 500_000
    max_batch_documents: int = 100
    max_token_budget: int = 32_000


class McpSettings(BaseModel):
    """MCP surface settings."""

    enable_document_tools: bool = False
    default_token_budget: int = 4000


class LoggingSettings(BaseModel):
    """Structured-logging settings. ``json`` renders logs as JSON lines when true."""

    model_config = {"populate_by_name": True}

    level: str = "INFO"
    json_output: bool = Field(default=True, alias="json")


class Settings(BaseSettings):
    """Top-level server configuration.

    Instances are cheap value objects; construct one per process at startup and
    thread it through :class:`~phoropter.service.core.ServiceCore`.
    """

    model_config = SettingsConfigDict(
        env_prefix="PHOROPTER_",
        env_nested_delimiter="__",
        toml_file="phoropter.toml",
        extra="ignore",
    )

    server: ServerSettings = Field(default_factory=ServerSettings)
    store: StoreSettings = Field(default_factory=StoreSettings)
    embedder: EmbedderSettings = Field(default_factory=EmbedderSettings)
    defaults: DefaultsSettings = Field(default_factory=DefaultsSettings)
    limits: LimitsSettings = Field(default_factory=LimitsSettings)
    mcp: McpSettings = Field(default_factory=McpSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Precedence (first wins): explicit init kwargs, then environment, then
        # an optional phoropter.toml. The environment always beats the file.
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls),
        )


def build_store(settings: Settings) -> VectorStoreAdapter:
    """Construct the configured vector store adapter.

    ``memory`` is built directly (it takes no arguments); ``qdrant`` is built
    from the store URL/credentials; any other kind is resolved through the
    ``phoropter.stores`` entry-point group and constructed with no arguments.
    """
    from phoropter.stores import load_store_class

    kind = settings.store.kind
    if kind == "memory":
        from phoropter.stores.memory import InMemoryStore

        return InMemoryStore()
    if kind == "qdrant":
        from phoropter.stores.qdrant import QdrantStore

        return QdrantStore(
            url=settings.store.url,
            api_key=settings.store.api_key,
            prefix=settings.store.prefix,
            timeout_s=settings.store.search_timeout_s,
        )
    return load_store_class(kind)()


def build_embedder(settings: Settings) -> Embedder:
    """Construct the configured embedder from the ``phoropter.embedders`` registry.

    The ``fake`` provider is built for tests and offline use; the shipped HTTP
    providers receive their model, base URL, credentials, and concurrency knobs.
    Unknown providers raise :class:`~phoropter.errors.EmbedderError`.
    """
    from phoropter.embed import create_embedder

    e = settings.embedder
    if e.provider == "fake":
        return create_embedder("fake")
    kwargs: dict[str, Any] = {
        "model": e.model,
        "batch_size": e.batch_size,
        "max_concurrency": e.max_concurrency,
    }
    if e.base_url is not None:
        kwargs["base_url"] = e.base_url
    # Ollama does not take an api_key; only pass it to providers that accept one.
    if e.api_key is not None and e.provider != "ollama":
        kwargs["api_key"] = e.api_key
    return create_embedder(e.provider, **kwargs)
