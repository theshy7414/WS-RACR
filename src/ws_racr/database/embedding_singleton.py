"""Shared sentence-embedding model (singleton).

``TaskManager`` (plan RAG), ``PerceptionLogger`` and the knowledge importer all need the same
``paraphrase-multilingual-MiniLM-L12-v2`` encoder; loading it once avoids repeated start-up cost.
``sentence-transformers`` is imported lazily, so modules that merely import this one (and the
offline tests) work without the ``llm`` extra installed.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingSingleton:
    """Process-wide holder of one SentenceTransformer instance."""

    _instance: Optional["EmbeddingSingleton"] = None
    _model: Optional["SentenceTransformer"] = None
    _model_name: str = DEFAULT_MODEL_NAME
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_model(cls, model_name: str = DEFAULT_MODEL_NAME) -> "SentenceTransformer":
        """Return the shared model, loading it on first use (or when ``model_name`` changes)."""
        if cls._model is None or model_name != cls._model_name:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError("Install the 'llm' extra: pip install -e .[llm]") from exc
            logger.info("Loading embedding model %s ...", model_name)
            start = time.time()
            cls._model = SentenceTransformer(model_name)
            cls._model_name = model_name
            cls._initialized = True
            logger.info("Embedding model loaded (%.2fs)", time.time() - start)
        return cls._model

    @classmethod
    def is_initialized(cls) -> bool:
        return cls._initialized

    @classmethod
    def reset(cls) -> None:
        """Drop the cached model (used in tests)."""
        cls._model = None
        cls._initialized = False
        logger.info("Embedding model reset")


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> "SentenceTransformer":
    """Return the shared embedding model."""
    return EmbeddingSingleton.get_model(model_name)
