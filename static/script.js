// ═══════════════════════════════════════════════════════
// Deep Doc — Frontend Logic
// ═══════════════════════════════════════════════════════

const PIPELINES = ['normal_rag', 'google_rag', 'rlm'];

// Current mode: 'upload' or 'benchmark'
let currentMode = 'upload';
let benchmarkQuestion = null;  // currently loaded benchmark question
let collectedAnswers = {};     // stores pipeline answers for judge

// ── Tab switching ────────────────────────────────────
function switchTab(mode) {
    currentMode = mode;

    // Toggle tab buttons
    document.getElementById('tab-upload').classList.toggle('active', mode === 'upload');
    document.getElementById('tab-benchmark').classList.toggle('active', mode === 'benchmark');

    // Toggle sections
    document.getElementById('upload-section').classList.toggle('hidden', mode !== 'upload');
    document.getElementById('benchmark-section').classList.toggle('hidden', mode !== 'benchmark');

    // Hide query/results when switching (they show after doc is loaded)
    if (mode === 'benchmark') {
        loadBenchStats();
        loadBenchQuestions();
    }
}

// ── Upload handlers ─────────────────────────────────────
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const statusPanel = document.getElementById('upload-status');

// Click to upload
uploadArea.addEventListener('click', () => fileInput.click());

// Drag & drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.name.toLowerCase().endsWith('.pdf')) {
        uploadFile(file);
    }
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        uploadFile(fileInput.files[0]);
    }
});

// ── Upload file ─────────────────────────────────────────
async function uploadFile(file) {
    // Hide benchmark ground truth if visible
    document.getElementById('ground-truth-section').classList.add('hidden');
    benchmarkQuestion = null;

    // Show status
    statusPanel.classList.remove('hidden');
    document.getElementById('status-filename').textContent = `Processing: ${file.name}`;
    document.getElementById('upload-spinner').classList.remove('hidden');
    document.getElementById('pipeline-statuses').innerHTML = `
        <div class="status-item"><span class="status-icon">...</span> Parsing document (text + tables + images)…</div>
        <div class="status-item"><span class="status-icon">...</span> Indexing for Normal RAG…</div>
        <div class="status-item"><span class="status-icon">...</span> Uploading to Google File Search…</div>
        <div class="status-item"><span class="status-icon">...</span> Loading for RLM…</div>
    `;

    // Disable upload area
    uploadArea.style.pointerEvents = 'none';
    uploadArea.style.opacity = '0.5';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const resp = await fetch('/upload', { method: 'POST', body: formData });
        const data = await resp.json();

        if (data.error) {
            showUploadError(data.error);
            return;
        }

        showPipelineStatuses(data, file.name);

    } catch (err) {
        showUploadError(err.message);
    } finally {
        uploadArea.style.pointerEvents = '';
        uploadArea.style.opacity = '';
    }
}

// ── Show pipeline statuses (shared between upload & benchmark) ──
function showPipelineStatuses(data, docName) {
    const statusHTML = [];

    // OCR Results
    const ocr = data.ocr;
    if (ocr) {
        const ocrParts = [
            `${ocr.num_pages} pages`,
            `${ocr.num_chunks} chunks`,
            `${ocr.num_tables} tables`,
            `${ocr.num_figures} figures`,
            `${(ocr.text_length || 0).toLocaleString()} chars`,
        ];
        statusHTML.push(
            `<div class="status-item"><span class="status-icon">OK</span> <strong>Document Parsed</strong> — ${ocrParts.join(' · ')} (${ocr.parse_time_s}s)</div>`
        );
    }

    // Normal RAG
    const nr = data.pipelines?.normal_rag;
    if (nr?.status === 'ok') {
        statusHTML.push(
            `<div class="status-item"><span class="status-icon">OK</span> Normal RAG — ${nr.num_chunks} chunks, ${nr.embed_dim}d embeddings (${nr.time_s}s)</div>`
        );
    } else {
        statusHTML.push(
            `<div class="status-item"><span class="status-icon">!!</span> Normal RAG — ${nr?.error || 'failed'}</div>`
        );
    }

    // Google RAG
    const gr = data.pipelines?.google_rag;
    if (gr?.status === 'ok') {
        statusHTML.push(
            `<div class="status-item"><span class="status-icon">OK</span> Google RAG — store ready (${gr.time_s}s)</div>`
        );
    } else {
        statusHTML.push(
            `<div class="status-item"><span class="status-icon">!!</span> Google RAG — ${gr?.error || 'failed'}</div>`
        );
    }

    // RLM
    const rlm = data.pipelines?.rlm;
    if (rlm?.status === 'ok') {
        statusHTML.push(
            `<div class="status-item"><span class="status-icon">OK</span> RLM — ${(rlm.doc_length || 0).toLocaleString()} chars loaded</div>`
        );
    } else {
        statusHTML.push(
            `<div class="status-item"><span class="status-icon">!!</span> RLM — ${rlm?.error || 'failed'}</div>`
        );
    }

    statusHTML.push(
        `<div class="status-item" style="font-weight: 600;"><span class="status-icon">-></span> Ready to query!</div>`
    );

    // Update the correct status panel depending on mode
    if (currentMode === 'benchmark') {
        document.getElementById('bench-pipeline-statuses').innerHTML = statusHTML.join('');
        document.getElementById('bench-spinner').classList.add('hidden');
        document.getElementById('bench-status-text').textContent = `Done: ${docName}`;
    } else {
        document.getElementById('pipeline-statuses').innerHTML = statusHTML.join('');
        document.getElementById('upload-spinner').classList.add('hidden');
        document.getElementById('status-filename').textContent = `Done: ${docName}`;
    }

    // Show query section and results
    document.getElementById('query-section').classList.remove('hidden');
    document.getElementById('results-section').classList.remove('hidden');
    document.getElementById('question-input').focus();
}

function showUploadError(msg) {
    document.getElementById('upload-spinner').classList.add('hidden');
    document.getElementById('pipeline-statuses').innerHTML =
        `<div class="status-item" style="color: #b91c1c;"><span class="status-icon">X</span> ${escapeHtml(msg)}</div>`;
}

// ═══════════════════════════════════════════════════════
// Benchmark
// ═══════════════════════════════════════════════════════

async function loadBenchStats() {
    try {
        const resp = await fetch('/benchmark/stats');
        const data = await resp.json();
        document.getElementById('stat-total').textContent = (data.total_queries || 0).toLocaleString();
        document.getElementById('stat-pdfs').textContent = (data.total_pdfs || 0).toLocaleString();
        document.getElementById('stat-abstractive').textContent = (data.by_type?.abstractive || 0).toLocaleString();
        document.getElementById('stat-extractive').textContent = (data.by_type?.extractive || 0).toLocaleString();

        // Source breakdown stats
        const src = data.by_source || {};
        document.getElementById('stat-text').textContent = (src['text'] || 0).toLocaleString();
        document.getElementById('stat-text-image').textContent = (src['text-image'] || 0).toLocaleString();
        document.getElementById('stat-text-table').textContent = (src['text-table'] || 0).toLocaleString();
        document.getElementById('stat-text-table-image').textContent = (src['text-table-image'] || 0).toLocaleString();
    } catch (err) {
        console.error('Failed to load benchmark stats:', err);
    }
}

let benchOffset = 0;
const BENCH_PAGE_SIZE = 20;
let benchTotal = 0;

async function loadBenchQuestions(shuffle = false, append = false) {
    if (!append) {
        benchOffset = 0;
    }

    const type = document.getElementById('bench-filter-type').value;
    const source = document.getElementById('bench-filter-source').value;

    const params = new URLSearchParams({
        limit: String(BENCH_PAGE_SIZE),
        offset: String(benchOffset),
        shuffle: shuffle ? 'true' : 'false',
    });
    if (type) params.set('type', type);
    if (source) params.set('source', source);

    const container = document.getElementById('bench-questions');
    if (!append) {
        container.innerHTML = '<p class="bench-placeholder">Loading…</p>';
    }

    try {
        const resp = await fetch(`/benchmark/questions?${params}`);
        const data = await resp.json();

        if (!data.questions || data.questions.length === 0) {
            if (!append) {
                container.innerHTML = '<p class="bench-placeholder">No questions match the filters.</p>';
            }
            // Remove load-more button if present
            const existingBtn = document.getElementById('bench-load-more');
            if (existingBtn) existingBtn.remove();
            return;
        }

        benchTotal = data.total || 0;

        const questionsHtml = data.questions.map(q => {
            const typeClass = `type-${q.type}`;
            const sourceClass = getSourceBadgeClass(q.source);
            return `
                <div class="bench-q-card" data-qid="${q.query_id}" onclick="selectBenchQuestion(this, '${q.query_id}')">
                    <div class="bench-q-text">${escapeHtml(q.query)}</div>
                    <div class="bench-q-badges">
                        <span class="bench-q-badge ${typeClass}">${q.type}</span>
                        <span class="bench-q-badge ${sourceClass}">${q.source}</span>
                        <button class="bench-load-btn" onclick="event.stopPropagation(); loadBenchQuestion('${q.query_id}')">
                            Load
                        </button>
                    </div>
                </div>
            `;
        }).join('');

        if (append) {
            // Remove old load-more button before appending
            const existingBtn = document.getElementById('bench-load-more');
            if (existingBtn) existingBtn.remove();
            container.insertAdjacentHTML('beforeend', questionsHtml);
        } else {
            container.innerHTML = questionsHtml;
        }

        // Update offset
        benchOffset += data.questions.length;

        // Add "Load More" button if there are more questions
        if (benchOffset < benchTotal) {
            const remaining = benchTotal - benchOffset;
            const loadMoreHtml = `
                <button id="bench-load-more" class="bench-load-more-btn" onclick="loadBenchQuestions(false, true)">
                    Load More (${remaining.toLocaleString()} remaining)
                </button>
            `;
            container.insertAdjacentHTML('beforeend', loadMoreHtml);
        }

    } catch (err) {
        if (!append) {
            container.innerHTML = `<p class="bench-placeholder" style="color: #b91c1c;">Error: ${err.message}</p>`;
        }
    }
}

function getSourceBadgeClass(source) {
    if (source === 'text') return 'source-text';
    if (source === 'text-image') return 'source-image';
    if (source === 'text-table') return 'source-table';
    return 'source-multi';
}

function selectBenchQuestion(el, queryId) {
    // Toggle selection
    document.querySelectorAll('.bench-q-card').forEach(c => c.classList.remove('selected'));
    el.classList.add('selected');
}

async function loadBenchQuestion(queryId) {
    // Show loading status
    const statusPanel = document.getElementById('bench-load-status');
    statusPanel.classList.remove('hidden');
    document.getElementById('bench-spinner').classList.remove('hidden');
    document.getElementById('bench-status-text').textContent = 'Downloading PDF + parsing document…';
    document.getElementById('bench-pipeline-statuses').innerHTML = `
        <div class="status-item"><span class="status-icon">...</span> Downloading PDF from arXiv…</div>
        <div class="status-item"><span class="status-icon">...</span> Parsing document…</div>
        <div class="status-item"><span class="status-icon">...</span> Indexing for all pipelines…</div>
    `;

    // Disable all load buttons
    document.querySelectorAll('.bench-load-btn').forEach(b => b.disabled = true);

    try {
        const resp = await fetch('/benchmark/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query_id: queryId }),
        });
        const data = await resp.json();

        if (data.error) {
            document.getElementById('bench-spinner').classList.add('hidden');
            document.getElementById('bench-pipeline-statuses').innerHTML =
                `<div class="status-item" style="color: #b91c1c;"><span class="status-icon">X</span> ${escapeHtml(data.error)}</div>`;
            return;
        }

        // Store the benchmark question
        benchmarkQuestion = data;

        // Show pipeline statuses
        showPipelineStatuses(data, `${data.doc_id}.pdf`);

        // Show ground truth
        showGroundTruth(data);

        // Pre-fill the question input
        document.getElementById('question-input').value = data.query;

    } catch (err) {
        document.getElementById('bench-spinner').classList.add('hidden');
        document.getElementById('bench-pipeline-statuses').innerHTML =
            `<div class="status-item" style="color: #b91c1c;"><span class="status-icon">X</span> ${escapeHtml(err.message)}</div>`;
    } finally {
        document.querySelectorAll('.bench-load-btn').forEach(b => b.disabled = false);
    }
}

function showGroundTruth(data) {
    const section = document.getElementById('ground-truth-section');
    section.classList.remove('hidden');

    document.getElementById('gt-type').textContent = data.query_type || '';
    document.getElementById('gt-source').textContent = data.source || '';
    document.getElementById('gt-doc').textContent = data.doc_id || '';
    document.getElementById('gt-answer').textContent = data.ground_truth || '';
}

// ── Query all pipelines ─────────────────────────────────
async function runAllPipelines() {
    const question = document.getElementById('question-input').value.trim();
    if (!question) return;

    const btn = document.getElementById('query-btn');
    btn.disabled = true;
    btn.querySelector('.btn-text').textContent = 'Running…';
    collectedAnswers = {};

    // Hide previous judge results
    document.getElementById('judge-section').classList.add('hidden');

    // Show loading on all cards
    PIPELINES.forEach(id => {
        document.getElementById(`loading-${id}`).classList.remove('hidden');
        document.getElementById(`answer-${id}`).innerHTML = '<span class="placeholder">Processing…</span>';
        document.getElementById(`cot-${id}`).innerHTML = '';
    });

    // Fire all three in parallel
    const promises = PIPELINES.map(pipeline => runPipeline(pipeline, question));
    await Promise.allSettled(promises);

    btn.disabled = false;
    btn.querySelector('.btn-text').textContent = 'Run All Three';

    // Auto-trigger judge if in benchmark mode with ground truth
    if (benchmarkQuestion && benchmarkQuestion.ground_truth && Object.keys(collectedAnswers).length > 0) {
        runJudge(question, benchmarkQuestion.ground_truth, collectedAnswers);
    }
}

async function runPipeline(pipeline, question) {
    try {
        const resp = await fetch('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pipeline, question }),
        });

        const data = await resp.json();

        // Hide loading
        document.getElementById(`loading-${pipeline}`).classList.add('hidden');

        if (data.error) {
            document.getElementById(`answer-${pipeline}`).textContent = `Error: ${data.error}`;
            return;
        }

        // Show answer (rendered as markdown)
        document.getElementById(`answer-${pipeline}`).innerHTML = formatContent(data.answer);

        // Store for judge
        collectedAnswers[pipeline] = data.answer;

        // Show chain of thought
        renderCoT(pipeline, data.steps || []);

    } catch (err) {
        document.getElementById(`loading-${pipeline}`).classList.add('hidden');
        document.getElementById(`answer-${pipeline}`).textContent = `Network error: ${err.message}`;
    }
}

// ── Render chain of thought ─────────────────────────────
function renderCoT(pipeline, steps) {
    const container = document.getElementById(`cot-${pipeline}`);
    container.innerHTML = '';

    steps.forEach((step, i) => {
        const stepEl = document.createElement('div');
        stepEl.className = 'cot-step';

        const headerEl = document.createElement('div');
        headerEl.className = 'cot-step-header';
        headerEl.innerHTML = `
            <span>${escapeHtml(step.title)}</span>
            <span class="arrow">▼</span>
        `;

        const contentEl = document.createElement('div');
        contentEl.className = 'cot-step-content';
        contentEl.innerHTML = formatContent(step.content || '');

        // Auto-open last step (the answer)
        if (i === steps.length - 1) {
            contentEl.classList.add('open');
            headerEl.querySelector('.arrow').classList.add('open');
        }

        headerEl.addEventListener('click', () => {
            contentEl.classList.toggle('open');
            headerEl.querySelector('.arrow').classList.toggle('open');
        });

        stepEl.appendChild(headerEl);
        stepEl.appendChild(contentEl);
        container.appendChild(stepEl);
    });
}

// ── Toggle all CoT steps ────────────────────────────────
function toggleCoT(pipeline) {
    const steps = document.querySelectorAll(`#cot-${pipeline} .cot-step-content`);
    const arrows = document.querySelectorAll(`#cot-${pipeline} .arrow`);
    const allOpen = Array.from(steps).every(s => s.classList.contains('open'));

    steps.forEach(s => {
        if (allOpen) s.classList.remove('open');
        else s.classList.add('open');
    });
    arrows.forEach(a => {
        if (allOpen) a.classList.remove('open');
        else a.classList.add('open');
    });
}

// ── Format content (markdown → HTML) ────────────────────
function formatContent(text) {
    // Escape HTML first
    let html = escapeHtml(text);

    // Code blocks: ```...```
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
        return `<pre><code>${code.trim()}</code></pre>`;
    });

    // Inline code: `...`
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Headings: # → h1, ## → h2, ### → h3, #### → h4
    html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Bold: **...**
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Italic: *...*
    html = html.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, '<em>$1</em>');

    // Horizontal rule: --- or ***
    html = html.replace(/^(---|\*\*\*)$/gm, '<hr>');

    // Blockquotes: > ...
    html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

    // Unordered lists: - item or * item
    html = html.replace(/^(?:- |\* )(.+)$/gm, '<li>$1</li>');
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

    // Ordered lists: 1. item
    html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
    // Wrap consecutive <li> not inside <ul> into <ol>
    html = html.replace(/(<li>.*<\/li>(?:\n<li>.*<\/li>)*)/g, (match) => {
        // Only wrap if not already inside a <ul>
        if (match.includes('<ul>')) return match;
        return `<ol>${match}</ol>`;
    });

    // Paragraphs: wrap remaining text blocks
    // Split by double newlines for paragraph separation
    html = html.replace(/\n\n/g, '</p><p>');

    // Single line breaks
    html = html.replace(/\n/g, '<br>');

    // Clean up: remove <br> right after block elements
    html = html.replace(/(<\/h[1-4]>)<br>/g, '$1');
    html = html.replace(/(<\/ul>)<br>/g, '$1');
    html = html.replace(/(<\/ol>)<br>/g, '$1');
    html = html.replace(/(<\/pre>)<br>/g, '$1');
    html = html.replace(/(<\/blockquote>)<br>/g, '$1');
    html = html.replace(/<br>(<h[1-4]>)/g, '$1');
    html = html.replace(/<br>(<ul>)/g, '$1');
    html = html.replace(/<br>(<ol>)/g, '$1');
    html = html.replace(/<br>(<pre>)/g, '$1');
    html = html.replace(/<br>(<hr>)/g, '$1');

    return html;
}

// ── Escape HTML ─────────────────────────────────────────
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ── Enter key to submit ─────────────────────────────────
document.getElementById('question-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        runAllPipelines();
    }
});

// ═══════════════════════════════════════════════════════
// LLM Judge
// ═══════════════════════════════════════════════════════

async function runJudge(question, groundTruth, answers) {
    const section = document.getElementById('judge-section');
    const loading = document.getElementById('judge-loading');
    const results = document.getElementById('judge-results');
    const errorEl = document.getElementById('judge-error');

    // Show section with loading
    section.classList.remove('hidden');
    loading.classList.remove('hidden');
    results.classList.add('hidden');
    errorEl.classList.add('hidden');

    try {
        const resp = await fetch('/benchmark/judge', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, ground_truth: groundTruth, answers }),
        });
        const data = await resp.json();

        loading.classList.add('hidden');

        if (data.error) {
            errorEl.classList.remove('hidden');
            errorEl.textContent = `Judge error: ${data.error}`;
            return;
        }

        // Show model name
        document.getElementById('judge-model').textContent = data.model || 'Claude Sonnet';

        renderJudgeResults(data);
        results.classList.remove('hidden');

    } catch (err) {
        loading.classList.add('hidden');
        errorEl.classList.remove('hidden');
        errorEl.textContent = `Network error: ${err.message}`;
    }
}

function renderJudgeResults(data) {
    // Summary
    document.getElementById('judge-summary').textContent = data.summary || '';

    // Score cards
    const pipelineConfig = {
        normal_rag: { name: 'Normal RAG', color: 'blue' },
        google_rag: { name: 'Google RAG', color: 'green' },
        rlm: { name: 'RLM', color: 'purple' },
    };

    const scoresContainer = document.getElementById('judge-scores');
    scoresContainer.innerHTML = '';

    for (const [key, config] of Object.entries(pipelineConfig)) {
        const scores = data[key];
        if (!scores) continue;

        const isWinner = data.winner === key;
        const card = document.createElement('div');
        card.className = `judge-score-card${isWinner ? ' winner-card' : ''}`;

        const dimensions = ['accuracy', 'completeness', 'relevance', 'overall'];
        const scoreRows = dimensions.map(dim => {
            const val = scores[dim] || 0;
            return `
                <div class="score-row">
                    <span class="score-label">${dim}</span>
                    <div class="score-bar">
                        <div class="score-bar-fill ${config.color}" style="width: ${val * 10}%"></div>
                    </div>
                    <span class="score-value">${val}/10</span>
                </div>
            `;
        }).join('');

        card.innerHTML = `
            <div class="pipeline-name ${config.color}">
                ${config.name}
                ${isWinner ? '<span class="winner-trophy">Winner</span>' : ''}
            </div>
            ${scoreRows}
            <div class="score-reasoning">${escapeHtml(scores.reasoning || '')}</div>
        `;

        scoresContainer.appendChild(card);
    }

    // Winner banner
    const winnerName = pipelineConfig[data.winner]?.name || data.winner;
    document.getElementById('judge-winner').innerHTML = `
        Winner: ${winnerName}
    `;
}
