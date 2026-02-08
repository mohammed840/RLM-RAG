"""
Reducto OCR Parser
------------------
Handles PDF parsing via Reducto's Parse API:
  - OCR + layout detection (no vision model)
  - Text extraction with structure preserved
  - Table extraction (HTML format)
  - Figure/image extraction (URLs)
  - Variable chunking for RAG indexing

Returns structured content that the RAG pipelines consume.
"""

import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field

from reducto import Reducto

logger = logging.getLogger(__name__)


# =====================================================================
# Data classes for structured parser output
# =====================================================================
@dataclass
class ParsedBlock:
    """A single layout element from a page."""
    block_type: str          # Title, Text, Table, Figure, Section Header, etc.
    content: str             # The text/markdown content
    page: int                # 1-indexed page number
    confidence: str          # "high" or "low"
    image_url: str | None = None   # URL for figure/table images
    bbox: dict = field(default_factory=dict)  # {left, top, width, height, page}


@dataclass
class ParsedChunk:
    """A semantic chunk from Reducto (ready for embedding)."""
    content: str             # Full markdown content of the chunk
    embed: str               # Embedding-optimized version
    blocks: list[ParsedBlock] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """Complete parsed document output."""
    chunks: list[ParsedChunk]
    full_text: str           # All chunk content concatenated
    tables: list[ParsedBlock]  # Just the table blocks
    figures: list[ParsedBlock]  # Just the figure blocks
    num_pages: int
    credits_used: float
    parse_time: float
    job_id: str
    studio_link: str


# =====================================================================
# Reducto Parser
# =====================================================================
class ReductoParser:
    """Parse PDFs using Reducto's OCR engine (no vision model)."""

    def __init__(self):
        api_key = os.environ.get("REDUCTO_API_KEY", "")
        if not api_key:
            raise ValueError(
                "REDUCTO_API_KEY environment variable is required. "
                "Get your key at https://studio.reducto.ai"
            )
        self.client = Reducto(api_key=api_key)

    def parse(self, filepath: str) -> ParsedDocument:
        """
        Upload and parse a PDF via Reducto.

        Uses:
        - Variable chunking (semantic boundaries — best for RAG)
        - HTML table output (preserves merged cells, complex layouts)
        - Figure + table image extraction (URLs)
        - NO figure summarization (no vision model)
        - Filters out headers/footers/page numbers (noise for RAG)

        Returns a ParsedDocument with structured content.
        """
        t0 = time.time()

        # ── Upload ────────────────────────────────────────────────────
        logger.info(f"Uploading {filepath} to Reducto...")
        upload = self.client.upload(file=Path(filepath))
        logger.info(f"Upload complete: {upload.file_id}")

        # ── Parse with optimal settings ───────────────────────────────
        logger.info("Parsing with Reducto OCR...")
        result = self.client.parse.run(
            input=upload.file_id,
            # Variable chunking: splits at semantic boundaries
            # (sections, tables, figures stay intact). Best for RAG.
            retrieval={
                "chunking": {"chunk_mode": "variable"},
                "filter_blocks": ["Header", "Footer", "Page Number"],
            },
            # HTML tables preserve complex layouts better
            formatting={
                "table_output_format": "html",
            },
            # Return image URLs for figures and tables
            settings={
                "return_images": ["figure", "table"],
            },
            # No vision model — just pure OCR
            enhance={
                "summarize_figures": False,
            },
        )

        elapsed = round(time.time() - t0, 2)

        # ── Process the response ──────────────────────────────────────
        chunks: list[ParsedChunk] = []
        all_tables: list[ParsedBlock] = []
        all_figures: list[ParsedBlock] = []
        full_text_parts: list[str] = []

        for chunk in result.result.chunks:
            parsed_blocks: list[ParsedBlock] = []

            for block in chunk.blocks:
                pb = ParsedBlock(
                    block_type=block.type,
                    content=block.content,
                    page=block.bbox.page if block.bbox else 0,
                    confidence=getattr(block, "confidence", "unknown"),
                    image_url=getattr(block, "image_url", None),
                    bbox={
                        "left": block.bbox.left,
                        "top": block.bbox.top,
                        "width": block.bbox.width,
                        "height": block.bbox.height,
                        "page": block.bbox.page,
                    } if block.bbox else {},
                )
                parsed_blocks.append(pb)

                if block.type == "Table":
                    all_tables.append(pb)
                elif block.type == "Figure":
                    all_figures.append(pb)

            pc = ParsedChunk(
                content=chunk.content,
                embed=getattr(chunk, "embed", chunk.content),
                blocks=parsed_blocks,
            )
            chunks.append(pc)
            full_text_parts.append(chunk.content)

        full_text = "\n\n".join(full_text_parts)

        doc = ParsedDocument(
            chunks=chunks,
            full_text=full_text,
            tables=all_tables,
            figures=all_figures,
            num_pages=result.usage.num_pages,
            credits_used=result.usage.credits,
            parse_time=elapsed,
            job_id=result.job_id,
            studio_link=getattr(result, "studio_link", ""),
        )

        logger.info(
            f"Reducto parse complete: {doc.num_pages} pages, "
            f"{len(chunks)} chunks, {len(all_tables)} tables, "
            f"{len(all_figures)} figures in {elapsed}s"
        )

        return doc
