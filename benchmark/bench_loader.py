"""
Open RAG Bench Loader
---------------------
Loads the Vectara Open RAG Benchmark dataset:
  - 3045 queries (abstractive / extractive)
  - Ground truth answers
  - Query-document relevance labels (qrels)
  - PDF URLs from arXiv

Supports browsing, filtering, and downloading benchmark PDFs.
Source: https://github.com/vectara/open-rag-bench
Dataset: https://huggingface.co/datasets/vectara/open_ragbench
"""

import json
import os
import random
import logging
import ssl
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@dataclass
class BenchmarkQuestion:
    """A single benchmark question with ground truth."""
    query_id: str
    query: str
    query_type: str        # "abstractive" or "extractive"
    source: str            # "text", "text-image", "text-table", "text-table-image"
    answer: str            # ground truth answer
    doc_id: str            # arxiv paper ID
    section_id: int        # section index in the document
    pdf_url: str           # URL to download the PDF


class BenchmarkLoader:
    """Load and serve Open RAG Bench questions."""

    def __init__(self):
        self.queries: dict = {}
        self.answers: dict = {}
        self.qrels: dict = {}
        self.pdf_urls: dict = {}
        self._loaded = False

    def load(self):
        """Load all benchmark data files."""
        if self._loaded:
            return

        logger.info("Loading Open RAG Bench dataset...")

        with open(os.path.join(DATA_DIR, "queries.json")) as f:
            self.queries = json.load(f)

        with open(os.path.join(DATA_DIR, "answers.json")) as f:
            self.answers = json.load(f)

        with open(os.path.join(DATA_DIR, "qrels.json")) as f:
            self.qrels = json.load(f)

        with open(os.path.join(DATA_DIR, "pdf_urls.json")) as f:
            self.pdf_urls = json.load(f)

        self._loaded = True
        logger.info(
            f"Loaded {len(self.queries)} queries, "
            f"{len(self.pdf_urls)} PDFs"
        )

    def get_stats(self) -> dict:
        """Return dataset statistics."""
        self.load()

        type_counts = {}
        source_counts = {}
        for q in self.queries.values():
            t = q.get("type", "unknown")
            s = q.get("source", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
            source_counts[s] = source_counts.get(s, 0) + 1

        return {
            "total_queries": len(self.queries),
            "total_pdfs": len(self.pdf_urls),
            "by_type": type_counts,
            "by_source": source_counts,
        }

    def get_question(self, query_id: str) -> BenchmarkQuestion | None:
        """Get a specific benchmark question by ID."""
        self.load()

        if query_id not in self.queries:
            return None

        q = self.queries[query_id]
        qrel = self.qrels.get(query_id, {})
        doc_id = qrel.get("doc_id", "")

        return BenchmarkQuestion(
            query_id=query_id,
            query=q["query"],
            query_type=q.get("type", "unknown"),
            source=q.get("source", "unknown"),
            answer=self.answers.get(query_id, ""),
            doc_id=doc_id,
            section_id=qrel.get("section_id", 0),
            pdf_url=self.pdf_urls.get(doc_id, ""),
        )

    def list_questions(
        self,
        query_type: str | None = None,
        source: str | None = None,
        limit: int = 20,
        offset: int = 0,
        shuffle: bool = False,
    ) -> tuple[list[dict], int]:
        """
        List benchmark questions with optional filtering.

        Returns (questions_list, total_matching_count).
        """
        self.load()

        # Build filtered list
        items = []
        for qid, q in self.queries.items():
            if query_type and q.get("type") != query_type:
                continue
            if source and q.get("source") != source:
                continue

            qrel = self.qrels.get(qid, {})
            doc_id = qrel.get("doc_id", "")

            items.append({
                "query_id": qid,
                "query": q["query"],
                "type": q.get("type", "unknown"),
                "source": q.get("source", "unknown"),
                "doc_id": doc_id,
                "has_answer": qid in self.answers,
            })

        total = len(items)

        if shuffle:
            random.shuffle(items)

        # Paginate
        page = items[offset : offset + limit]
        return page, total

    def get_random_questions(
        self,
        count: int = 5,
        query_type: str | None = None,
        source: str | None = None,
    ) -> list[dict]:
        """Get random benchmark questions."""
        items, _ = self.list_questions(
            query_type=query_type,
            source=source,
            limit=999999,
        )
        return random.sample(items, min(count, len(items)))

    def download_pdf(self, doc_id: str, dest_dir: str) -> str | None:
        """
        Download a benchmark PDF from arXiv.

        Returns the local file path, or None on failure.
        """
        self.load()

        url = self.pdf_urls.get(doc_id)
        if not url:
            logger.error(f"No PDF URL for doc_id: {doc_id}")
            return None

        os.makedirs(dest_dir, exist_ok=True)
        filename = f"{doc_id}.pdf"
        filepath = os.path.join(dest_dir, filename)

        # Skip if already downloaded
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            logger.info(f"PDF already cached: {filepath}")
            return filepath

        logger.info(f"Downloading {url} → {filepath}")
        try:
            # macOS Python doesn't use system certs by default.
            # Try certifi first, fall back to unverified context.
            try:
                import certifi
                ctx = ssl.create_default_context(cafile=certifi.where())
            except ImportError:
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Deep Doc RAG Benchmark)"}
            )
            with urllib.request.urlopen(req, context=ctx) as resp:
                with open(filepath, "wb") as out:
                    out.write(resp.read())

            logger.info(f"Downloaded: {filepath} ({os.path.getsize(filepath)} bytes)")
            return filepath
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            # Clean up partial downloads
            if os.path.exists(filepath):
                os.remove(filepath)
            return None
