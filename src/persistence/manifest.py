"""Persisted index metadata and compatibility validation."""

import json

from dataclasses import asdict, dataclass
from pathlib import Path

class ManifestError(ValueError):
    pass

@dataclass(frozen=True)
class IndexManifest:
    schema_version: int
    backend: str
    embedding_model: str
    vector_dimension: int
    chunk_size: int
    chunk_overlap: int
    document_count: int
    chunk_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

def write_manifest(
    directory: Path,
    manifest: IndexManifest,
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    target = directory / "manifest.json"
    temporary = directory / "manifest.json.tmp"

    temporary.write_text(
        json.dumps(
            manifest.to_dict(),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(target)

def load_manifest(directory: Path) -> IndexManifest:
    path = Path(directory) / "manifest.json"
    try:
        value = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(
            "invalid index manifest: cannot read manifest"
        ) from exc

    if not isinstance(value, dict):
        raise ManifestError(
            "invalid index manifest: root must be an object"
        )

    try:
        manifest = IndexManifest(
            schema_version=int(value["schema_version"]),
            backend=str(value["backend"]),
            embedding_model=str(value["embedding_model"]),
            vector_dimension=int(value["vector_dimension"]),
            chunk_size=int(value["chunk_size"]),
            chunk_overlap=int(value["chunk_overlap"]),
            document_count=int(value["document_count"]),
            chunk_count=int(value["chunk_count"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError(
            "invalid index manifest: missing or invalid fields"
        ) from exc

    validate_manifest(manifest)
    return manifest  

def validate_manifest(manifest: IndexManifest) -> None:
    if manifest.schema_version != 1:
        raise ManifestError(
            f"unsupported index schema version: "
            f"{manifest.schema_version}"
        )

    if manifest.backend not in {"numpy", "qdrant"}:
        raise ManifestError(
            f"unsupported index backend: {manifest.backend}"
        )

    if not manifest.embedding_model.strip():
        raise ManifestError(
            "embedding model cannot be empty"
        )

    if manifest.vector_dimension <= 0:
        raise ManifestError(
            "vector dimension must be positive"
        )

    if manifest.chunk_size <= 0:
        raise ManifestError(
            "chunk size must be positive"
        )

    if not 0 <= manifest.chunk_overlap < manifest.chunk_size:
        raise ManifestError(
            "chunk overlap must be non-negative and smaller than chunk size"
        )

    if manifest.document_count < 0:
        raise ManifestError(
            "document count cannot be negative"
        )

    if manifest.chunk_count <= 0:
        raise ManifestError(
            "chunk count must be positive"
        )

def qdrant_manifest_directory(
    qdrant_path: Path,
    collection_name: str,
) -> Path:
    if not collection_name.strip():
        raise ValueError("collection_name cannot be empty")

    qdrant_path = Path(qdrant_path)

    return (
        qdrant_path.parent
        / f"{qdrant_path.name}-manifests"
        / collection_name
    )

def validate_compatibility(
    manifest: IndexManifest,
    *,
    backend: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    if manifest.backend != backend:
        raise ManifestError(
            "index backend does not match current configuration"
        )

    if manifest.embedding_model != embedding_model:
        raise ManifestError(
            "embedding model does not match index manifest"
        )

    if manifest.chunk_size != chunk_size:
        raise ManifestError(
            "chunk size does not match index manifest"
        )

    if manifest.chunk_overlap != chunk_overlap:
        raise ManifestError(
            "chunk overlap does not match index manifest"
        )

def validate_vector_dimension(
    manifest: IndexManifest,
    actual_dimension: int,
) -> None:
    if actual_dimension <= 0:
        raise ManifestError(
            "actual vector dimension must be positive"
        )

    if manifest.vector_dimension != actual_dimension:
        raise ManifestError(
            "vector dimension does not match index manifest"
        )
