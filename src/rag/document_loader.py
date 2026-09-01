from pathlib import Path
import re
from zipfile import BadZipFile

from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from lxml.etree import XMLSyntaxError

from src.rag.models import Paragraph


class DocumentLoadError(ValueError):
    pass


def _normalize(text: str) -> str:
    """Normalize repeated whitespace inside one Word paragraph."""
    return re.sub(r"\s+", " ", text).strip()


def load_docx(path: Path) -> list[Paragraph]:
    if path.suffix.lower() != ".docx":
        raise DocumentLoadError(
            f"unsupported file type: {path.suffix}"
        )
    try:
        document = Document(path)
    except (
        BadZipFile,
        PackageNotFoundError,
        XMLSyntaxError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        raise DocumentLoadError(
            f"cannot read {path.name}: {exc}"
        ) from exc

    paragraphs: list[Paragraph] = []
    for position, paragraph in enumerate(
        document.paragraphs,
        start=1,
    ):
        text = _normalize(paragraph.text)

        if not text:
            continue

        style_name = (
            paragraph.style.name
            if paragraph.style
            else ""
        )
        paragraphs.append(
            Paragraph(
                text=text,
                source=path.name,
                position=position,
                is_heading=(
                    style_name.lower().startswith("heading")
                    or style_name.startswith("标题")
                ),
            )
        )

    if not paragraphs:
        raise DocumentLoadError(
            f"empty document: {path.name}"
        )

    return paragraphs


def load_directory(
    path: Path,
) -> tuple[dict[str, list[Paragraph]], dict[str, str]]:
    documents: dict[str, list[Paragraph]] = {}
    errors: dict[str, str] = {}

    docx_files = sorted(
        path.glob("*.docx"),
        key=lambda item: item.name.lower(),
    )

    for file_path in docx_files:
        try:
            documents[file_path.name] = load_docx(file_path)
        except DocumentLoadError as exc:
            errors[file_path.name] = str(exc)

    return documents, errors
