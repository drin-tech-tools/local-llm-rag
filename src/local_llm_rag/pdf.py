"""PDF text extraction, one string per page so answers can cite page numbers."""

from io import BytesIO

from pypdf import PdfReader


def extract_pages(data: bytes) -> list[str]:
    reader = PdfReader(BytesIO(data))
    return [page.extract_text() or "" for page in reader.pages]
