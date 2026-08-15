import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from src.models import Chunk, SearchResult


class IndexFormatError(ValueError):
    pass


class VectorStore:
    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2-D matrix")

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                "chunks and embeddings must have the same number of items"
            )

        self.chunks = chunks
        self.embeddings = embeddings.astype(
            np.float32,
            copy=False,
        )

    def search(
        self,
        query: np.ndarray,
        top_k: int,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if query.ndim != 1 or query.shape[0] != self.embeddings.shape[1]:
            raise ValueError("query dimension must match embedding dimension")

        matrix_norms = np.linalg.norm(
            self.embeddings,
            axis=1,
        )
        query_norm = np.linalg.norm(query)

        denominators = matrix_norms * query_norm

        scores = np.divide(
            self.embeddings @ query,
            denominators,
            out=np.zeros_like(matrix_norms),
            where=denominators != 0,
        )

        indexes = np.argsort(
            -scores,
            kind="stable",
        )[: min(top_k, len(self.chunks))]

        return [
            SearchResult(
                chunk=self.chunks[index],
                score=float(scores[index]),
            )
            for index in indexes
        ]

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        chunks_target = directory / "chunks.json"
        vectors_target = directory / "embeddings.npy"

        chunks_temp: Path | None = None
        vectors_temp: Path | None = None

        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=directory,
                delete=False,
            ) as file:
                chunks_temp = Path(file.name)
                json.dump(
                    [chunk.to_dict() for chunk in self.chunks],
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            with NamedTemporaryFile(
                mode="wb",
                dir=directory,
                delete=False,
            ) as file:
                vectors_temp = Path(file.name)
                np.save(file, self.embeddings)

            os.replace(chunks_temp, chunks_target)
            os.replace(vectors_temp, vectors_target)
        finally:
            if chunks_temp is not None:
                chunks_temp.unlink(missing_ok=True)
            if vectors_temp is not None:
                vectors_temp.unlink(missing_ok=True)

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        directory = Path(directory)

        try:
            metadata = json.loads(
                (directory / "chunks.json").read_text(
                    encoding="utf-8",
                )
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise IndexFormatError(
                "invalid vector index: cannot read chunk metadata"
            ) from exc
        try:
            if not isinstance(metadata, list):
                raise TypeError("chunk metadata must be a list")
            chunks = [
                Chunk.from_dict(item)
                for item in metadata
            ]
        except (TypeError, KeyError, ValueError) as exc:
            raise IndexFormatError(
                "invalid vector index: invalid chunk metadata"
            ) from exc

        try:
            embeddings = np.load(
                directory / "embeddings.npy",
                allow_pickle=False,
            )
        except (OSError, ValueError, EOFError) as exc:
            raise IndexFormatError(
                "invalid vector index: cannot read embeddings"
            ) from exc

        try:
            return cls(chunks, embeddings)
        except ValueError as exc:
            raise IndexFormatError(
                "invalid vector index: inconsistent chunks and embeddings"
            ) from exc
