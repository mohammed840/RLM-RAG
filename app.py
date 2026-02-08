"""
Deep Doc — RAG vs Google RAG vs RLM
====================================
Flask backend serving three document intelligence pipelines side-by-side.
Uses Reducto OCR for PDF parsing (text, tables, images — no vision model).

Run:  python app.py
"""

import os
import json
import traceback
import logging
import httpx
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
JUDGE_MODEL = "anthropic/claude-sonnet-4"

from pipelines.reducto_parser import ReductoParser
from pipelines.normal_rag import NormalRAG
from pipelines.google_rag import GoogleRAG
from pipelines.rlm_pipeline import RLMPipeline
from benchmark.bench_loader import BenchmarkLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# Config
# =====================================================================
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen/qwen3-30b-a3b")
# Paper's approach: GPT-5 (root) + GPT-5-mini (sub) — use a cheaper sub-model
QWEN_SUB_MODEL = os.environ.get("QWEN_SUB_MODEL", "qwen/qwen3-8b")
GOOGLE_MODEL = os.environ.get("GOOGLE_MODEL", "gemini-2.5-flash")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =====================================================================
# Flask app
# =====================================================================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# =====================================================================
# Pipeline singletons
# =====================================================================
reducto_parser = ReductoParser()
normal_rag = NormalRAG(model_name=QWEN_MODEL)
google_rag = GoogleRAG(model_name=GOOGLE_MODEL)
rlm_pipe = RLMPipeline(model_name=QWEN_MODEL, sub_model_name=QWEN_SUB_MODEL)
bench = BenchmarkLoader()

# Track state
doc_state = {"loaded": False, "name": ""}


# =====================================================================
# Routes
# =====================================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Upload PDF → Reducto OCR → feed all three pipelines."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    # Save
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    doc_name = file.filename

    results = {"filename": doc_name, "pipelines": {}}

    # ── Reducto OCR Parse ──────────────────────────────────────────
    try:
        logger.info(f"Parsing {doc_name} with Reducto OCR...")
        parsed = reducto_parser.parse(filepath)

        results["ocr"] = {
            "engine": "Reducto",
            "num_pages": parsed.num_pages,
            "num_chunks": len(parsed.chunks),
            "num_tables": len(parsed.tables),
            "num_figures": len(parsed.figures),
            "text_length": len(parsed.full_text),
            "credits_used": parsed.credits_used,
            "parse_time_s": parsed.parse_time,
            "job_id": parsed.job_id,
            "studio_link": parsed.studio_link,
        }
        logger.info(
            f"Reducto OCR: {parsed.num_pages} pages, "
            f"{len(parsed.chunks)} chunks, "
            f"{len(parsed.tables)} tables, "
            f"{len(parsed.figures)} figures"
        )
    except Exception as e:
        logger.error(f"Reducto parse failed: {e}")
        return jsonify({"error": f"Reducto OCR failed: {e}"}), 500

    # --- Normal RAG (uses Reducto-parsed chunks for indexing) ------
    try:
        info = normal_rag.process_document(parsed, doc_name=doc_name)
        results["pipelines"]["normal_rag"] = {
            "status": "ok",
            "num_chunks": info["num_chunks"],
            "embed_dim": info["embed_dim"],
            "time_s": info["time_s"],
        }
    except Exception as e:
        results["pipelines"]["normal_rag"] = {"status": "error", "error": str(e)}

    # --- Google RAG (uploads raw PDF to Google File Search) --------
    try:
        info = google_rag.process_document(filepath, doc_name=doc_name)
        results["pipelines"]["google_rag"] = {
            "status": "ok",
            "store_name": info["store_name"],
            "time_s": info["time_s"],
        }
    except Exception as e:
        results["pipelines"]["google_rag"] = {"status": "error", "error": str(e)}

    # --- RLM (uses Reducto-parsed full text) -----------------------
    try:
        info = rlm_pipe.process_document(parsed.full_text, doc_name=doc_name)
        results["pipelines"]["rlm"] = {
            "status": "ok",
            "doc_length": info["doc_length"],
        }
    except Exception as e:
        results["pipelines"]["rlm"] = {"status": "error", "error": str(e)}

    doc_state["loaded"] = True
    doc_state["name"] = doc_name
    return jsonify(results)


@app.route("/query", methods=["POST"])
def query():
    """Run a question through a specific pipeline."""
    data = request.get_json()
    question = data.get("question", "").strip()
    pipeline = data.get("pipeline", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        if pipeline == "normal_rag":
            answer, steps = normal_rag.query(question)
        elif pipeline == "google_rag":
            answer, steps = google_rag.query(question)
        elif pipeline == "rlm":
            answer, steps = rlm_pipe.query(question)
        else:
            return jsonify({"error": f"Unknown pipeline: {pipeline}"}), 400

        return jsonify({"answer": answer, "steps": steps})

    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error": str(e), "traceback": tb}), 500


# =====================================================================
# Benchmark routes
# =====================================================================
@app.route("/benchmark/stats")
def benchmark_stats():
    """Return dataset statistics."""
    try:
        stats = bench.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/benchmark/questions")
def benchmark_questions():
    """List benchmark questions with optional filtering."""
    query_type = request.args.get("type")       # abstractive / extractive
    source = request.args.get("source")          # text / text-image / text-table / text-table-image
    limit = int(request.args.get("limit", 20))
    offset = int(request.args.get("offset", 0))
    shuffle = request.args.get("shuffle", "false").lower() == "true"

    try:
        questions, total = bench.list_questions(
            query_type=query_type,
            source=source,
            limit=limit,
            offset=offset,
            shuffle=shuffle,
        )
        return jsonify({"questions": questions, "total": total})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/benchmark/load", methods=["POST"])
def benchmark_load():
    """
    Load a benchmark question:
    1. Download PDF from arXiv
    2. Parse with Reducto OCR
    3. Feed all three pipelines
    4. Return ground truth + OCR stats
    """
    data = request.get_json()
    query_id = data.get("query_id", "").strip()

    if not query_id:
        return jsonify({"error": "No query_id provided"}), 400

    # Look up the question
    question = bench.get_question(query_id)
    if not question:
        return jsonify({"error": f"Unknown query_id: {query_id}"}), 404

    results = {
        "query_id": question.query_id,
        "query": question.query,
        "query_type": question.query_type,
        "source": question.source,
        "ground_truth": question.answer,
        "doc_id": question.doc_id,
        "pdf_url": question.pdf_url,
        "pipelines": {},
    }

    # ── Download PDF ──────────────────────────────────────────────
    bench_pdf_dir = os.path.join(UPLOAD_FOLDER, "benchmark")
    filepath = bench.download_pdf(question.doc_id, bench_pdf_dir)
    if not filepath:
        return jsonify({"error": f"Failed to download PDF: {question.pdf_url}"}), 500

    doc_name = f"{question.doc_id}.pdf"

    # ── Reducto OCR Parse ─────────────────────────────────────────
    try:
        logger.info(f"Parsing benchmark PDF {doc_name} with Reducto OCR...")
        parsed = reducto_parser.parse(filepath)

        results["ocr"] = {
            "engine": "Reducto",
            "num_pages": parsed.num_pages,
            "num_chunks": len(parsed.chunks),
            "num_tables": len(parsed.tables),
            "num_figures": len(parsed.figures),
            "text_length": len(parsed.full_text),
            "credits_used": parsed.credits_used,
            "parse_time_s": parsed.parse_time,
            "job_id": parsed.job_id,
            "studio_link": parsed.studio_link,
        }
    except Exception as e:
        logger.error(f"Reducto parse failed for benchmark: {e}")
        return jsonify({"error": f"Reducto OCR failed: {e}"}), 500

    # --- Normal RAG ---
    try:
        info = normal_rag.process_document(parsed, doc_name=doc_name)
        results["pipelines"]["normal_rag"] = {
            "status": "ok",
            "num_chunks": info["num_chunks"],
            "embed_dim": info["embed_dim"],
            "time_s": info["time_s"],
        }
    except Exception as e:
        results["pipelines"]["normal_rag"] = {"status": "error", "error": str(e)}

    # --- Google RAG ---
    try:
        info = google_rag.process_document(filepath, doc_name=doc_name)
        results["pipelines"]["google_rag"] = {
            "status": "ok",
            "store_name": info["store_name"],
            "time_s": info["time_s"],
        }
    except Exception as e:
        results["pipelines"]["google_rag"] = {"status": "error", "error": str(e)}

    # --- RLM ---
    try:
        info = rlm_pipe.process_document(parsed.full_text, doc_name=doc_name)
        results["pipelines"]["rlm"] = {
            "status": "ok",
            "doc_length": info["doc_length"],
        }
    except Exception as e:
        results["pipelines"]["rlm"] = {"status": "error", "error": str(e)}

    doc_state["loaded"] = True
    doc_state["name"] = doc_name
    return jsonify(results)


# =====================================================================
# LLM-as-Judge (Claude Sonnet via OpenRouter)
# =====================================================================
@app.route("/benchmark/judge", methods=["POST"])
def benchmark_judge():
    """
    Use Claude Sonnet as an impartial judge to evaluate which pipeline
    answer is closest to the ground truth.
    """
    data = request.get_json()
    ground_truth = data.get("ground_truth", "")
    question = data.get("question", "")
    answers = data.get("answers", {})  # {normal_rag: "...", google_rag: "...", rlm: "..."}

    if not ground_truth or not question or not answers:
        return jsonify({"error": "Missing required fields"}), 400

    # Build the judge prompt
    judge_prompt = f"""You are an impartial judge evaluating RAG (Retrieval-Augmented Generation) pipeline answers.

**Question asked:**
{question}

**Ground Truth Answer (the correct answer):**
{ground_truth}

**Pipeline Answers to evaluate:**

1. **Normal RAG** (Qwen3 + embeddings + top-K retrieval):
{answers.get('normal_rag', 'No answer provided')}

2. **Google RAG** (Gemini + File Search + grounding):
{answers.get('google_rag', 'No answer provided')}

3. **RLM** (DSPy Recursive Language Model + REPL + sub-LM calls):
{answers.get('rlm', 'No answer provided')}

---

**Your task:**
Score each pipeline answer on a scale of 1-10 across three dimensions:
- **Accuracy**: How factually correct is the answer compared to the ground truth?
- **Completeness**: Does it cover all key points from the ground truth?
- **Relevance**: Does it stay on-topic and avoid unnecessary information?

Then determine the **overall winner** — which pipeline answer is closest to the ground truth.

**Respond in this exact JSON format only (no markdown, no code fences):**
{{
  "normal_rag": {{
    "accuracy": <1-10>,
    "completeness": <1-10>,
    "relevance": <1-10>,
    "overall": <1-10>,
    "reasoning": "<1-2 sentence explanation>"
  }},
  "google_rag": {{
    "accuracy": <1-10>,
    "completeness": <1-10>,
    "relevance": <1-10>,
    "overall": <1-10>,
    "reasoning": "<1-2 sentence explanation>"
  }},
  "rlm": {{
    "accuracy": <1-10>,
    "completeness": <1-10>,
    "relevance": <1-10>,
    "overall": <1-10>,
    "reasoning": "<1-2 sentence explanation>"
  }},
  "winner": "<normal_rag|google_rag|rlm>",
  "summary": "<2-3 sentence overall comparison>"
}}"""

    try:
        resp = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": JUDGE_MODEL,
                "messages": [
                    {"role": "user", "content": judge_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1000,
            },
            timeout=60,
        )
        if resp.status_code != 200:
            logger.error(f"Judge API error {resp.status_code}: {resp.text}")
        resp.raise_for_status()
        result = resp.json()

        # Extract the judge's response
        judge_text = result["choices"][0]["message"]["content"].strip()

        # Parse JSON from the response (handle possible markdown fences)
        if judge_text.startswith("```"):
            judge_text = judge_text.split("\n", 1)[1]
            judge_text = judge_text.rsplit("```", 1)[0]

        judge_data = json.loads(judge_text)
        judge_data["model"] = JUDGE_MODEL
        return jsonify(judge_data)

    except json.JSONDecodeError as e:
        logger.error(f"Judge JSON parse error: {e}\nRaw: {judge_text}")
        return jsonify({"error": "Failed to parse judge response", "raw": judge_text}), 500
    except Exception as e:
        logger.error(f"Judge error: {e}")
        return jsonify({"error": str(e)}), 500


# =====================================================================
# Entry point
# =====================================================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
