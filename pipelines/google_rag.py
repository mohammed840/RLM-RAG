"""
Google RAG Pipeline
-------------------
Uses Google's File Search (genai SDK) to:
1. Upload the PDF to a File Search Store
2. Query using Gemini with the file_search tool
3. Return grounding sources and the answer
"""

import os
import time
from google import genai
from google.genai import types


class GoogleRAG:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        self.client = genai.Client(api_key=api_key)
        self.model = model_name
        self.store = None
        self.store_name: str = ""
        self.doc_name: str = ""

    # ------------------------------------------------------------------
    # Document upload
    # ------------------------------------------------------------------
    def process_document(self, filepath: str, doc_name: str = "") -> dict:
        """Upload PDF to Google File Search Store. Returns processing info."""
        self.doc_name = doc_name
        t0 = time.time()
        steps_log: list[str] = []

        # Create a new store
        self.store = self.client.file_search_stores.create()
        self.store_name = self.store.name
        steps_log.append(f"Created store: {self.store_name}")

        # Upload document
        upload_op = self.client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=self.store_name,
            file=filepath,
        )
        steps_log.append("Upload started…")

        # Poll until done
        poll_count = 0
        while not upload_op.done:
            time.sleep(3)
            upload_op = self.client.operations.get(upload_op)
            poll_count += 1
            if poll_count > 60:  # 3-min timeout
                raise TimeoutError("Google upload timed out after 3 minutes")

        elapsed = time.time() - t0
        steps_log.append(f"Upload completed in {elapsed:.1f}s")

        return {
            "store_name": self.store_name,
            "time_s": round(elapsed, 2),
            "log": steps_log,
        }

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(self, question: str) -> tuple[str, list[dict]]:
        """
        Query the uploaded document via Gemini + File Search.
        Returns (answer, steps).
        """
        if not self.store_name:
            return "⚠️ No document uploaded to Google.", []

        steps: list[dict] = []
        t0 = time.time()

        # ── Step 1: Prepare file_search tool ─────────────────────────
        steps.append({
            "title": "🔧 Step 1 — Configure File Search Tool",
            "content": (
                f"Using Google File Search store `{self.store_name}` "
                f"with model `{self.model}`.\n\n"
                "The file has been chunked and indexed by Google's "
                "internal retrieval system."
            ),
        })

        # ── Step 2: Generate with file_search ────────────────────────
        steps.append({
            "title": "🔍 Step 2 — Query with Grounding",
            "content": (
                f"Sending question to `{self.model}` with `file_search` tool.\n\n"
                f"Google will internally retrieve relevant passages and "
                f"ground the response."
            ),
        })

        response = self.client.models.generate_content(
            model=self.model,
            contents=question,
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[self.store_name]
                        )
                    )
                ]
            ),
        )

        answer = response.text or "(No response text)"

        # ── Step 3: Extract grounding sources ────────────────────────
        grounding_info = "No grounding metadata found."
        sources = []

        candidate = response.candidates[0] if response.candidates else None
        if candidate and candidate.grounding_metadata:
            gm = candidate.grounding_metadata
            if gm.grounding_chunks:
                for chunk in gm.grounding_chunks:
                    ctx = chunk.retrieved_context
                    title = getattr(ctx, "title", "Unknown") if ctx else "Unknown"
                    text = getattr(ctx, "text", "") if ctx else ""
                    sources.append({"title": title, "text": text[:300]})

                source_titles = {s["title"] for s in sources}
                grounding_info = (
                    f"**{len(sources)} grounding chunks** from "
                    f"{len(source_titles)} source(s):\n\n"
                )
                for i, s in enumerate(sources, 1):
                    preview = s["text"][:200].replace("\n", " ")
                    grounding_info += (
                        f"**Chunk {i}** — *{s['title']}*\n"
                        f"> {preview}…\n\n"
                    )

        steps.append({
            "title": "📚 Step 3 — Grounding Sources",
            "content": grounding_info,
        })

        # ── Step 4: Final answer ─────────────────────────────────────
        elapsed = round(time.time() - t0, 2)

        usage = response.usage_metadata
        token_info = ""
        if usage:
            token_info = (
                f"  \n**Tokens** — prompt: {usage.prompt_token_count:,}, "
                f"completion: {usage.candidates_token_count:,}, "
                f"total: {usage.total_token_count:,}"
            )

        steps.append({
            "title": f"✅ Step 4 — Answer  ({elapsed}s)",
            "content": answer + "\n" + token_info,
        })

        return answer, steps

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def cleanup(self):
        """Delete the file search store to avoid orphaned resources."""
        if self.store_name:
            try:
                self.client.file_search_stores.delete(name=self.store_name)
                self.store_name = ""
            except Exception:
                pass
