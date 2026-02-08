"""
RLM Pipeline (Recursive Language Model)
----------------------------------------
Uses DSPy's dspy.RLM module to recursively process documents:
1. Load the FULL document text as a variable in a sandboxed REPL
2. The LLM writes code to explore, chunk, filter, and sub-query
3. Returns a trajectory showing every step of the reasoning
"""

import os
import time
import dspy


class RLMPipeline:
    def __init__(
        self,
        model_name: str = "qwen/qwen3-30b-a3b",
        sub_model_name: str | None = None,
        max_iterations: int = 15,
        max_llm_calls: int = 50,
    ):
        self.model_name = model_name
        # Paper uses different models: GPT-5 (root) + GPT-5-mini (sub)
        # We mirror this with a bigger root and cheaper sub for cost efficiency
        self.sub_model_name = sub_model_name or "qwen/qwen3-8b"
        self.max_iterations = max_iterations
        self.max_llm_calls = max_llm_calls
        self.document_text: str = ""
        self.doc_name: str = ""

        # Configure DSPy with OpenRouter
        self.main_lm = dspy.LM(
            f"openrouter/{model_name}",
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        )
        self.sub_lm = dspy.LM(
            f"openrouter/{self.sub_model_name}",
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        )
        dspy.configure(lm=self.main_lm)

        # We'll build the RLM module dynamically per-query to inject
        # context metadata (length, etc.) like the paper does.
        self.rlm = None  # built in query()

    def _build_rlm(self, doc_length: int) -> dspy.RLM:
        """Build RLM with document-aware instructions (from the paper's Appendix C)."""
        # ── Paper's Appendix C system prompt (adapted for our variable names) ──
        # Matches the GPT-5 prompt + the Qwen3-Coder cost warning
        instructions = (
            "You are tasked with answering a query with associated context. "
            "You can access, transform, and analyze this context interactively "
            "in a REPL environment that can recursively query sub-LLMs, which you are strongly "
            "encouraged to use as much as possible. "
            "You will be queried iteratively until you provide a final answer.\n\n"

            f"Your context is a string with {doc_length:,} total characters.\n\n"

            "The REPL environment is initialized with:\n"
            "1. A 'document' variable that contains extremely important information about your query. "
            "You should check the content of the 'document' variable to understand what you are working with. "
            "Make sure you look through it sufficiently as you answer your query.\n"
            "2. A 'question' variable containing the query to answer.\n"
            "3. A 'llm_query' function that allows you to query an LLM (that can handle around 500K chars) "
            "inside your REPL environment.\n"
            "4. The ability to use 'print()' statements to view the output of your REPL code "
            "and continue your reasoning.\n\n"

            "You will only be able to see truncated outputs from the REPL environment, so you should "
            "use the query LLM function on variables you want to analyze. You will find this function "
            "especially useful when you have to analyze the semantics of the context.\n\n"

            "Use these variables as buffers to build up your final answer.\n\n"

            "Make sure to explicitly look through the entire context in REPL before answering your query. "
            "An example strategy is to first look at the context and figure out a chunking strategy, "
            "then break up the context into smart chunks, and query an LLM per chunk with a particular "
            "question and save the answers to a buffer, then query an LLM with all the buffers to produce "
            "your final answer.\n\n"

            "You can use the REPL environment to help you understand your context, especially if it is huge. "
            "Remember that your sub LLMs are powerful -- they can fit around 500K characters in their context window, "
            "so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed "
            "10 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit "
            "it in a few sub-LLM calls!\n\n"

            # ── Qwen3 cost warning (from paper's Appendix C, prompt 1b) ──
            "IMPORTANT: Be very careful about using 'llm_query' as it incurs high runtime costs. "
            "Always batch as much information as reasonably possible into each call "
            "(aim for around ~200k characters per call). For example, if you have 1000 lines of "
            "information to process, it's much better to split into chunks of 5 and call 'llm_query' "
            "on each chunk (200 calls total) rather than making 1000 individual calls. "
            "Minimize the number of 'llm_query' calls by batching related information together.\n\n"

            "IMPORTANT: The sub-LLM called via llm_query() CANNOT see the 'document' variable — "
            "you MUST include the relevant text in the prompt string.\n\n"

            # ── Examples from the paper ──
            "Example — iterative chunking with buffers:\n"
            "```\n"
            "buffers = []\n"
            "chunk_size = len(document) // 5\n"
            "for i in range(5):\n"
            "    start = i * chunk_size\n"
            "    end = start + chunk_size if i < 4 else len(document)\n"
            "    chunk = document[start:end]\n"
            "    answer = llm_query(f'You are iteratively looking through a document, section {i} of 5. "
            "Gather information to help answer: {question}\\n\\nSection:\\n{chunk}')\n"
            "    buffers.append(answer)\n"
            "    print(f'After section {i}: {answer[:200]}')\n"
            "final = llm_query(f'Based on these extracts, answer: {question}\\n\\n' + '\\n'.join(buffers))\n"
            "SUBMIT(final)\n"
            "```\n\n"

            "Example — when context isn't too long, combine chunks and batch sub-LLM calls:\n"
            "```\n"
            "# Context is ~1M chars, split into ~200K per call = 5 calls\n"
            "chunk_size = len(document) // 5\n"
            "answers = []\n"
            "for i in range(5):\n"
            "    chunk = document[i*chunk_size : (i+1)*chunk_size if i < 4 else len(document)]\n"
            "    ans = llm_query(f'Answer based on this section: {question}\\n\\nText:\\n{chunk}')\n"
            "    answers.append(ans)\n"
            "final = llm_query(f'Combine these partial answers to produce the final answer for: "
            "{question}\\n\\n' + '\\n---\\n'.join(answers))\n"
            "SUBMIT(final)\n"
            "```\n"
        )

        rlm_signature = dspy.Signature(
            "document, question -> answer",
            instructions=instructions,
        )

        return dspy.RLM(
            rlm_signature,
            max_iterations=self.max_iterations,
            max_llm_calls=self.max_llm_calls,
            sub_lm=self.sub_lm,
            verbose=True,
        )

    # ------------------------------------------------------------------
    # Document processing
    # ------------------------------------------------------------------
    def process_document(self, text: str, doc_name: str = "") -> dict:
        """Store the full document text. RLM doesn't need pre-processing."""
        self.document_text = text
        self.doc_name = doc_name
        return {
            "doc_length": len(text),
            "doc_lines": text.count("\n"),
            "note": "RLM loads the full document into the REPL — no chunking needed.",
        }

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(self, question: str) -> tuple[str, list[dict]]:
        """
        Run the RLM pipeline.
        Returns (answer, steps) where steps show the full trajectory.
        """
        if not self.document_text:
            return "⚠️ No document loaded.", []

        steps: list[dict] = []
        t0 = time.time()

        # ── Step 1: Setup info ───────────────────────────────────────
        steps.append({
            "title": "🧠 Step 1 — Initialize RLM Environment",
            "content": (
                f"Loading **{len(self.document_text):,}** characters into a "
                f"sandboxed Python REPL as the `document` variable.\n\n"
                f"- **Root LM**: `{self.model_name}` (decides strategy, writes code)\n"
                f"- **Sub LM**: `{self.sub_model_name}` (handles `llm_query()` calls)\n"
                f"- **Max iterations**: {self.max_iterations}\n"
                f"- **Max LLM calls**: {self.max_llm_calls}\n\n"
                f"The LLM will now write Python code to explore the document, "
                f"make recursive sub-LLM calls, and build up an answer."
            ),
        })

        # ── Step 2: Run RLM ──────────────────────────────────────────
        try:
            # Build RLM with document-aware instructions (paper's approach)
            rlm = self._build_rlm(doc_length=len(self.document_text))
            result = rlm(
                document=self.document_text,
                question=question,
            )
            answer = result.answer

            # ── Step 3: Parse trajectory ─────────────────────────────
            trajectory = getattr(result, "trajectory", [])
            for i, step in enumerate(trajectory, 1):
                reasoning = step.get("reasoning", "")
                code = step.get("code", "")
                output = step.get("output", "")

                step_content = ""
                if reasoning:
                    step_content += f"**Reasoning:**\n{reasoning}\n\n"
                if code:
                    step_content += f"**Code:**\n```python\n{code}\n```\n\n"
                if output:
                    # Truncate very long outputs for display
                    display_output = output[:2000]
                    if len(output) > 2000:
                        display_output += f"\n... ({len(output) - 2000} more chars)"
                    step_content += f"**Output:**\n```\n{display_output}\n```"

                steps.append({
                    "title": f"🔄 Iteration {i}",
                    "content": step_content or "(empty step)",
                })

        except Exception as e:
            answer = f"⚠️ RLM error: {e}"
            steps.append({
                "title": "❌ Error",
                "content": f"```\n{e}\n```",
            })

        elapsed = round(time.time() - t0, 2)

        # ── Final ────────────────────────────────────────────────────
        steps.append({
            "title": f"✅ Final Answer  ({elapsed}s)",
            "content": answer,
        })

        return answer, steps
