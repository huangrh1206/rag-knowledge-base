from zipfile import BadZipFile
from docx.opc.exceptions import PackageNotFoundError

from pathlib import Path
import re

from docx import Document

from src.models import Paragraph

class DocumentLoadError(ValueError):
    pass

def _normalize(text: str) -> str:
    "清除空格和换行符"
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
        ValueError, 
        KeyError
    ) as e:
        raise DocumentLoadError(
            f"cannot read: {path.name}, error: {e}"
        ) from e
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
                text = text,
                source = path.name,
                position = position,
                is_heading = (
                    style_name.lower().startswith("heading")
                    or style_name.lower().startswith("标题")
                ),
            )
        )
    
    if not paragraphs:
        raise DocumentLoadError(
            f"empty document: {path.name}"
        )

    return paragraphs

def load_directory(path: Path
                   ) -> tuple[dict[str, list[Paragraph]], dict[str, str]]:

    documents: dict[str, list[Paragraph]] = {}
    errors: dict[str, str] = {}

    docx_files = sorted(
        path.glob("*.docx"),
        key = lambda item: item.name.lower(),
    )

    for file_path in docx_files:
        try:
            documents[file_path.name] = load_docx(file_path)
        except DocumentLoadError as e:
            errors[file_path.name] = str(e)

    return documents, errors
