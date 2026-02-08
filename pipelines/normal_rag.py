"""
Normal RAG Pipeline
-------------------
Traditional Retrieval-Augmented Generation:
1. Accepts Reducto-parsed chunks (text, tables, images already extracted)
2. Embeds chunks with sentence-transformers
3. Retrieves top-K similar chunks via cosine similarity
4. Generates answer with Qwen3 via OpenRouter
"""

import os
import time
import numpy as np
from openai import OpenAI


class NormalRAG:
    def __init__(self, model_name: str = "qwen/qwen3-30b-a3b"):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        )
        self.model = model_name
        self.chunks: list[str] = []
        self.chunk_metadata: list[dict] = []  # block types, pages, etc.
        self.embeddings: np.ndarray | None = None
        self._embed_model = None
        self.doc_name: str = ""

    # ------------------------------------------------------------------
    # Embedding model (lazy-loaded)
    # ------------------------------------------------------------------
    def _get_embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embed_model

    # ------------------------------------------------------------------
    # Document processing — accepts Reducto ParsedDocument
    # ------------------------------------------------------------------
    def process_document(self, parsed_doc, doc_name: str = "") -> dict:
        """
        Index a Reducto-parsed document.

        Uses Reducto's semantic chunks directly (no re-chunking needed).
        Each chunk's `embed` field is optimized for embedding by Reducto.

        Args:
            parsed_doc: ParsedDocument from ReductoParser
            doc_name: display name of the document
        """
        self.doc_name = doc_name
        t0 = time.time()

        # ── Use Reducto's pre-built chunks for indexing ───────────────
        self.chunks = []
        self.chunk_metadata = []

        for chunk in parsed_doc.chunks:
            # Use the embed-optimized text for the embedding
            embed_text = chunk.embed if chunk.embed else chunk.content
            if embed_text.strip():
                self.chunks.append(embed_text.strip())

                # Collect metadata about what's in this chunk
                block_types = [b.block_type for b in chunk.blocks]
                pages = sorted(set(b.page for b in chunk.blocks if b.page > 0))
                has_table = any(b.block_type == "Table" for b in chunk.blocks)
                has_figure = any(b.block_type == "Figure" for b in chunk.blocks)
                image_urls = [
                    b.image_url for b in chunk.blocks
                    if b.image_url
                ]

                self.chunk_metadata.append({
                    "block_types": block_types,
                    "pages": pages,
                    "has_table": has_table,
                    "has_figure": has_figure,
                    "image_urls": image_urls,
                    "content": chunk.content,  # full content for retrieval display
                })

        # ── Embed the chunks ──────────────────────────────────────────
        model = self._get_embed_model()
        self.embeddings = model.encode(self.chunks, normalize_embeddings=True)

        elapsed = time.time() - t0
        return {
            "num_chunks": len(self.chunks),
            "embed_dim": self.embeddings.shape[1],
            "time_s": round(elapsed, 2),
        }

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(self, question: str, top_k: int = 5) -> tuple[str, list[dict]]:
        """
        Run the full RAG pipeline.
        Returns (answer, steps) where steps is a list of
        {"title": str, "content": str} dicts for the chain-of-thought.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return "⚠️ No document loaded.", []

        steps: list[dict] = []
        t0 = time.time()

        # ── Step 1: Embed the query ──────────────────────────────────
        model = self._get_embed_model()
        query_emb = model.encode([question], normalize_embeddings=True)
        steps.append({
            "title": "🔤 Step 1 — Embed Query",
            "content": (
                f"Encoded the question into a **{query_emb.shape[1]}-dimensional** "
                f"vector using `all-MiniLM-L6-v2`."
            ),
        })

        # ── Step 2: Retrieve top-K chunks ────────────────────────────
        scores = np.dot(self.embeddings, query_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        retrieval_details = []
        for rank, idx in enumerate(top_indices, 1):
            meta = self.chunk_metadata[idx] if idx < len(self.chunk_metadata) else {}
            pages_str = f" (pages {meta.get('pages', [])})" if meta.get("pages") else ""
            type_tags = ""
            if meta.get("has_table"):
                type_tags += " 📊 Table"
            if meta.get("has_figure"):
                type_tags += " 🖼️ Figure"

            preview = self.chunks[idx][:250].replace("\n", " ")
            retrieval_details.append(
                f"**#{rank}** — Chunk {idx}{pages_str}{type_tags}  "
                f"(score **{scores[idx]:.4f}**)\n"
                f"> {preview}…"
            )

        steps.append({
            "title": f"📄 Step 2 — Retrieve Top-{top_k} Chunks",
            "content": "\n\n".join(retrieval_details),
        })

        # ── Step 3: Build prompt & generate ──────────────────────────
        # Use the full content (not embed text) for the LLM context
        context_parts = []
        for idx in top_indices:
            if idx < len(self.chunk_metadata):
                content = self.chunk_metadata[idx].get("content", self.chunks[idx])
            else:
                content = self.chunks[idx]
            context_parts.append(content)

        context = "\n\n---\n\n".join(context_parts)

        prompt_preview = (
            f"System: Answer based on the context. Be thorough.\n\n"
            f"Context ({len(context)} chars from {top_k} chunks) + Question"
        )
        steps.append({
            "title": "🤖 Step 3 — Generate Answer",
            "content": (
                f"Sending **{len(context):,}** characters of context to "
                f"`{self.model}` via OpenRouter.\n\n"
                f"Context includes text, tables (HTML), and figure descriptions "
                f"extracted by **Reducto OCR**.\n\n"
                f"```\n{prompt_preview}\n```"
            ),
        })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer the question based "
                        "only on the provided context. The context may include "
                        "text, HTML tables, and figure descriptions from a PDF "
                        "document parsed with OCR. Be thorough, accurate, "
                        "and cite specific parts of the context."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
        )

        answer = response.choices[0].message.content
        elapsed = round(time.time() - t0, 2)

        # ── Step 4: Final answer ─────────────────────────────────────
        usage = response.usage
        token_info = ""
        if usage:
            token_info = (
                f"  \n**Tokens** — prompt: {usage.prompt_tokens:,}, "
                f"completion: {usage.completion_tokens:,}, "
                f"total: {usage.total_tokens:,}"
            )

        steps.append({
            "title": f"✅ Step 4 — Answer  ({elapsed}s)",
            "content": answer + "\n" + token_info,
        })

        return answer, steps
