"""Document loader for HR policy documents."""

import os
from pathlib import Path
from typing import List, Dict, Any
import re


class DocumentLoader:
    """Load and parse HR policy documents from various formats."""

    def __init__(self, policies_dir: str = "policies"):
        self.policies_dir = Path(policies_dir)

    def load_documents(self) -> List[Dict[str, Any]]:
        """Load all documents from the policies directory."""
        documents = []

        if not self.policies_dir.exists():
            raise FileNotFoundError(f"Policies directory not found: {self.policies_dir}")

        for file_path in self.policies_dir.rglob("*"):
            if file_path.is_file() and self._is_supported_format(file_path):
                doc = self._load_file(file_path)
                if doc:
                    documents.append(doc)

        return documents

    def _is_supported_format(self, path: Path) -> bool:
        """Check if file format is supported."""
        supported = {".md", ".txt", ".html", ".htm", ".pdf"}
        return path.suffix.lower() in supported

    def _load_file(self, file_path: Path) -> Dict[str, Any] | None:
        """Load a single file and return document metadata."""
        try:
            # Handle PDF files
            if file_path.suffix.lower() == ".pdf":
                return self._load_pdf(file_path)

            # Handle text-based files (md, txt, html)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract title from first heading or filename
            title = self._extract_title(content, file_path.stem)

            # Extract sections using markdown headings
            sections = self._extract_sections(content)

            return {
                "id": file_path.stem,
                "title": title,
                "source": str(file_path),
                "content": content,
                "sections": sections,
                "metadata": {
                    "format": file_path.suffix.lower(),
                    "filename": file_path.name,
                    "size": len(content),
                }
            }
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def _load_pdf(self, file_path: Path) -> Dict[str, Any] | None:
        """Load and parse a PDF file."""
        try:
            from pypdf import PdfReader
        except ImportError:
            try:
                from PyPDF2 import PdfReader
            except ImportError:
                print(f"Error loading {file_path}: pypdf or PyPDF2 is required for PDF support")
                return None

        try:
            reader = PdfReader(file_path)
            pages_text = []

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages_text.append(f"[Page {page_num + 1}]\n{text}")

            content = "\n\n".join(pages_text)

            # Extract title from metadata or first line
            title = self._extract_pdf_title(reader, file_path.stem)

            # Extract sections from PDF content
            sections = self._extract_pdf_sections(content)

            return {
                "id": file_path.stem,
                "title": title,
                "source": str(file_path),
                "content": content,
                "sections": sections,
                "metadata": {
                    "format": ".pdf",
                    "filename": file_path.name,
                    "size": len(content),
                    "page_count": len(reader.pages),
                }
            }
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
            return None

    def _extract_pdf_title(self, reader: "PdfReader", default: str) -> str:
        """Extract title from PDF metadata or first line."""
        # Try to get title from PDF metadata
        if hasattr(reader, 'metadata') and reader.metadata:
            title = reader.metadata.get("/Title", "")
            if title:
                return title.strip()

        # Try to extract title from the first page content
        if reader.pages and len(reader.pages) > 0:
            try:
                first_page_text = reader.pages[0].extract_text()
                if first_page_text:
                    # Look for an all-caps line near the top (likely the title)
                    for line in first_page_text.split("\n")[:20]:
                        stripped = line.strip()
                        # Title is usually the first non-empty all-caps line
                        if stripped and stripped.isupper() and 5 < len(stripped) < 100:
                            return stripped
            except Exception:
                pass

        return default.replace("-", " ").replace("_", " ").title()

    def _extract_pdf_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract sections from PDF text, looking for page markers and patterns."""
        sections = []
        lines = content.split("\n")
        current_section = {"heading": "Introduction", "level": 1, "content": ""}

        for line in lines:
            stripped = line.strip()

            # Check for page markers
            page_match = re.match(r'^\[Page\s+(\d+)\]$', stripped)
            if page_match:
                if current_section["content"].strip():
                    sections.append(current_section)
                current_section = {"heading": f"Page {page_match.group(1)}", "level": 1, "content": ""}
                continue

            # Check if line looks like a section header (all caps, short, ends with colon or is numbered)
            if self._is_pdf_section_header(stripped):
                if current_section["content"].strip():
                    sections.append(current_section)
                current_section = {"heading": stripped, "level": 2, "content": ""}
                continue

            current_section["content"] += line + "\n"

        if current_section["content"].strip():
            sections.append(current_section)

        return sections

    def _is_pdf_section_header(self, line: str) -> bool:
        """Check if a line looks like a section header in PDF text."""
        if not line:
            return False

        # Skip if too long or too short
        if len(line) > 100 or len(line) < 3:
            return False

        # Check for common section header patterns
        patterns = [
            r'^[A-Z][A-Z\s\d\-:]+$',  # ALL CAPS
            r'^\d+\.\s+[A-Z]',  # Numbered sections like "1. INTRODUCTION"
            r'^[A-Z]\.\s+[A-Z]',  # Lettered sections like "A. OVERVIEW"
            r'^(Section|Chapter)\s+\d+',  # Section/Chapter prefixes
        ]

        for pattern in patterns:
            if re.match(pattern, line):
                return True

        return False

    def _extract_title(self, content: str, default: str, format_suffix: str = "") -> str:
        """Extract title from first heading or use default."""
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                title = stripped.lstrip("#").strip()
                if title:
                    return title
        return default.replace("-", " ").replace("_", " ").title()

    def _extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract sections based on markdown or HTML headings."""
        sections = []
        lines = content.split("\n")
        current_section = {"heading": "Introduction", "level": 1, "content": ""}
        current_level = 1

        for line in lines:
            # Check if line is a markdown heading (e.g., # Title)
            md_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if md_match:
                # Save previous section if it has content
                if current_section["content"].strip():
                    sections.append(current_section)

                level = len(md_match.group(1))
                heading = md_match.group(2).strip()
                current_section = {"heading": heading, "level": level, "content": ""}
                current_level = level
                continue

            # Check if line is an HTML heading (e.g., <h1>Title</h1>)
            html_match = re.match(r'<h([1-6])[^>]*>(.+?)</h\1>', line, re.IGNORECASE)
            if html_match:
                # Save previous section if it has content
                if current_section["content"].strip():
                    sections.append(current_section)

                level = int(html_match.group(1))
                heading = re.sub(r'<[^>]+>', '', html_match.group(2)).strip()
                current_section = {"heading": heading, "level": level, "content": ""}
                current_level = level
                continue

            current_section["content"] += line + "\n"

        # Add last section
        if current_section["content"].strip():
            sections.append(current_section)

        return sections

    def load_document_by_id(self, doc_id: str) -> Dict[str, Any] | None:
        """Load a specific document by its ID."""
        for doc in self.load_documents():
            if doc["id"] == doc_id:
                return doc
        return None

    def get_document_count(self) -> int:
        """Get the total number of loaded documents."""
        return len(self.load_documents())
