# Deep Doc: Comparative Analysis of RAG and RLM Approaches

A systematic comparison of three document intelligence paradigms: traditional Retrieval-Augmented Generation (RAG), Google's File Search-based RAG, and Recursive Language Models (RLM).

## Overview

This repository implements three distinct approaches for question-answering over long documents, evaluated on the [Vectara Open RAG Benchmark](https://github.com/vectara/open-rag-bench) (3,045 questions across 1,000 research papers).

### Implemented Approaches

| Approach | Architecture | Context Strategy | Model |
|----------|-------------|------------------|-------|
| **Normal RAG** | Chunking + Embedding + Top-K Retrieval | Pre-indexed chunks (384-dim vectors) | Qwen3-30B via OpenRouter |
| **Google RAG** | File Search + Grounding | Google-managed indexing | Gemini 2.5 Flash |
| **RLM** | REPL + Recursive Sub-LM Calls | Dynamic context mining | Qwen3-30B via OpenRouter |

## Motivation

Recent work on Recursive Language Models [1] has demonstrated a novel paradigm for processing long-context tasks without requiring massive context windows. However, there remains confusion in the community about the relationship between RLMs and traditional RAG systems. This implementation provides:

1. **Direct comparison** of RLM against established RAG baselines
2. **Full transparency** into each system's reasoning process (chain-of-thought traces)
3. **Benchmark evaluation** on realistic document QA tasks

## Key Differences

### Traditional RAG
- **Pre-indexes** documents into fixed-size chunks
- **Retrieves** top-K most similar chunks via embedding similarity
- **Single LLM call** with retrieved context

### Google RAG
- **Managed indexing** via Google File Search
- **Grounding-based retrieval** with source attribution
- **Single LLM call** with grounded context

### RLM (Recursive Language Model)
- **No pre-indexing** — processes documents on-demand
- **Iterative reasoning** via REPL environment (Python sandbox)
- **Multiple sub-LM calls** to extract and verify information
- **Programmatic context navigation** (search, filter, chunk dynamically)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Query                            │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Normal RAG   │   │  Google RAG   │   │      RLM      │
├───────────────┤   ├───────────────┤   ├───────────────┤
│ 1. Embed Q    │   │ 1. Upload PDF │   │ 1. Load doc   │
│ 2. Top-K      │   │ 2. File Search│   │    as variable│
│    retrieval  │   │    grounding  │   │ 2. LLM writes │
│ 3. LLM call   │   │ 3. LLM call   │   │    Python code│
│    (Qwen3)    │   │    (Gemini)   │   │ 3. Sub-LM     │
│               │   │               │   │    calls      │
│               │   │               │   │ 4. Iterate    │
│               │   │               │   │ 5. SUBMIT()   │
└───────────────┘   └───────────────┘   └───────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                    ┌───────────────┐
                    │ Side-by-Side  │
                    │  Comparison   │
                    └───────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- [Deno](https://deno.land) (required for RLM's sandboxed REPL)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/RLM-RAG.git
   cd RLM-RAG
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Deno** (for RLM sandbox)
   ```bash
   curl -fsSL https://deno.land/install.sh | sh
   ```
   Restart your shell after installation.

4. **Configure API keys**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your credentials:
   ```bash
   OPENROUTER_API_KEY=your_openrouter_key
   GOOGLE_API_KEY=your_google_key
   REDUCTO_API_KEY=your_reducto_key
   ```

5. **Run the application**
   ```bash
   python app.py
   ```
   Navigate to `http://localhost:5000`

## API Keys

| Service | Purpose | Obtain Key |
|---------|---------|------------|
| OpenRouter | Normal RAG + RLM (Qwen3) | [openrouter.ai](https://openrouter.ai) |
| Google AI | Google RAG (Gemini) | [ai.google.dev](https://ai.google.dev) |
| Reducto | PDF parsing (OCR) | [reducto.ai](https://reducto.ai) |

## Configuration

Model selection can be configured via environment variables:

```bash
QWEN_MODEL=qwen/qwen3-30b-a3b      # OpenRouter model for RAG + RLM
GOOGLE_MODEL=gemini-2.5-flash       # Google model for File Search
```

## Project Structure

```
RLM-RAG/
├── app.py                      # Flask backend
├── pipelines/
│   ├── normal_rag.py           # Traditional RAG implementation
│   ├── google_rag.py           # Google File Search RAG
│   ├── rlm_pipeline.py         # DSPy RLM implementation
│   └── reducto_parser.py       # PDF parsing with Reducto OCR
├── benchmark/
│   ├── bench_loader.py         # Vectara Open RAG Bench loader
│   └── data/                   # Benchmark questions & answers
├── templates/
│   └── index.html              # Web interface
├── static/
│   ├── style.css               # UI styling
│   └── script.js               # Frontend logic
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Implementation Details

### Normal RAG Pipeline

1. **Document Processing**
   - PDF parsed via Reducto OCR (extracts text, tables, figures)
   - Text chunked into ~1000-character segments (200-char overlap)
   - Chunks embedded using `sentence-transformers/all-MiniLM-L6-v2` (384-dim)

2. **Query Processing**
   - Question embedded using same model
   - Top-5 chunks retrieved via cosine similarity
   - Context + question sent to Qwen3-30B

### Google RAG Pipeline

1. **Document Processing**
   - Raw PDF uploaded to Google File Search Store
   - Google internally indexes and chunks the document

2. **Query Processing**
   - Gemini 2.5 Flash uses `file_search` tool
   - Retrieves relevant passages with grounding metadata
   - Returns answer with source attribution

### RLM Pipeline

1. **Document Processing**
   - Full document text loaded as Python variable in REPL
   - No pre-indexing or chunking

2. **Query Processing**
   - LLM receives question + metadata (doc length, structure)
   - LLM writes Python code to explore document:
     - `peek(start, end)` — view text slice
     - `search(keyword)` — find occurrences
     - `llm_query(text, question)` — spawn sub-LM call
   - Iterates until sufficient information gathered
   - Calls `SUBMIT(answer)` to finalize

3. **Recursive Reasoning**
   - Main LLM never sees full document (avoids context overflow)
   - Sub-LM calls process specific chunks
   - Full trajectory (code + outputs) displayed for transparency

## Benchmark

The application includes the **Vectara Open RAG Benchmark** [2]:

- **3,045 questions** across 1,000 research papers (arXiv)
- **Question types**: Abstractive (1,793), Extractive (1,252)
- **Source types**: Text (1,914), Text+Image (763), Text+Table (148), Text+Table+Image (220)

### Evaluation

Each pipeline can be evaluated on benchmark questions with:
- **Ground truth answers** for comparison
- **LLM-as-Judge** scoring (Claude Sonnet 4 via OpenRouter)
- **Metrics**: Accuracy, Completeness, Relevance (1-10 scale)

## References

[1] Zhang, A., Kraska, T., & Khattab, O. (2025). Recursive Language Models. *arXiv preprint arXiv:2512.24601*. [https://arxiv.org/abs/2512.24601](https://arxiv.org/abs/2512.24601)

[2] Vectara. (2024). Open RAG Bench: A Benchmark for Retrieval-Augmented Generation. [https://github.com/vectara/open-rag-bench](https://github.com/vectara/open-rag-bench)

[3] Khattab, O., et al. (2024). DSPy: Programming—not prompting—Foundation Models. [https://dspy.ai](https://dspy.ai)

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{rlm_rag_comparison,
  title={Deep Doc: Comparative Analysis of RAG and RLM Approaches},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/RLM-RAG}
}
```

## License

MIT License

## Acknowledgments

- **DSPy** team for the RLM implementation
- **Vectara** for the Open RAG Benchmark
- **Reducto** for PDF parsing infrastructure
