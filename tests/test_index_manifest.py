import json

import pytest

from src.index_manifest import (
    IndexManifest,
    ManifestError,
    load_manifest,
    write_manifest,
)


def manifest() -> IndexManifest:
    return IndexManifest(
        schema_version=1,
        backend="numpy",
        embedding_model="text-embedding-3-small",
        vector_dimension=1536,
        chunk_size=700,
        chunk_overlap=100,
        document_count=3,
        chunk_count=12,
    )


def test_manifest_round_trips(tmp_path) -> None:
    write_manifest(tmp_path, manifest())

    restored = load_manifest(tmp_path)

    assert restored == manifest()


def test_manifest_is_human_readable_json(tmp_path) -> None:
    write_manifest(tmp_path, manifest())

    value = json.loads(
        (tmp_path / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert value["backend"] == "numpy"
    assert value["embedding_model"] == "text-embedding-3-small"
    assert value["vector_dimension"] == 1536


def test_manifest_rejects_unsupported_schema(tmp_path) -> None:
    value = manifest().to_dict()
    value["schema_version"] = 2

    (tmp_path / "manifest.json").write_text(
        json.dumps(value),
        encoding="utf-8",
    )

    with pytest.raises(
        ManifestError,
        match="unsupported index schema version",
    ):
        load_manifest(tmp_path)


def test_manifest_rejects_invalid_chunk_overlap(tmp_path) -> None:
    value = manifest().to_dict()
    value["chunk_overlap"] = 700

    (tmp_path / "manifest.json").write_text(
        json.dumps(value),
        encoding="utf-8",
    )

    with pytest.raises(
        ManifestError,
        match="chunk overlap",
    ):
        load_manifest(tmp_path)


def test_manifest_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(
        ManifestError,
        match="cannot read manifest",
    ):
        load_manifest(tmp_path)