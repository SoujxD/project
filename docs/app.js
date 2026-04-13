const SAMPLE_QUESTIONS = [
  "Which customer segments should we prioritize based on this ecommerce dataset?",
  "What patterns distinguish stronger-performing sessions or customers?",
  "Which channels appear most promising for future budget allocation?",
  "What recommendations would improve performance based on this data?"
];

const STORAGE_KEYS = {
  csv: "ise547_dataset_csv",
  name: "ise547_dataset_name"
};

const API_BASE_URL = (window.APP_CONFIG?.API_BASE_URL || "").replace(/\/$/, "");

function hasBackend() {
  return Boolean(API_BASE_URL);
}

function normalize(name) {
  return String(name || "").toLowerCase().replace(/[^a-z0-9]/g, "");
}

function parseCsvText(text) {
  return Papa.parse(text, { header: true, dynamicTyping: true, skipEmptyLines: true });
}

async function loadDefaultDataset() {
  const response = await fetch("./assets/default_dataset.csv");
  const text = await response.text();
  localStorage.setItem(STORAGE_KEYS.csv, text);
  localStorage.setItem(STORAGE_KEYS.name, "default_dataset.csv");
  return parseCsvText(text).data;
}

async function getDataset() {
  const stored = localStorage.getItem(STORAGE_KEYS.csv);
  if (stored) return parseCsvText(stored).data;
  return loadDefaultDataset();
}

function getStoredCsv() {
  return localStorage.getItem(STORAGE_KEYS.csv);
}

function getDatasetName() {
  return localStorage.getItem(STORAGE_KEYS.name) || "default_dataset.csv";
}

function saveDataset(fileName, csvText) {
  localStorage.setItem(STORAGE_KEYS.csv, csvText);
  localStorage.setItem(STORAGE_KEYS.name, fileName);
}

function makeFormData(extraFields = {}) {
  const csv = getStoredCsv();
  const fileName = getDatasetName();
  const blob = new Blob([csv], { type: "text/csv" });
  const formData = new FormData();
  formData.append("file", blob, fileName);
  Object.entries(extraFields).forEach(([key, value]) => formData.append(key, String(value)));
  return formData;
}

async function fetchApi(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

function isNumericColumn(rows, column) {
  const values = rows.map((row) => row[column]).filter((value) => value !== null && value !== undefined && value !== "");
  if (!values.length) return false;
  return values.every((value) => !Number.isNaN(Number(value)));
}

function targetSeries(rows, column) {
  if (!column) return null;
  const mapped = rows.map((row) => {
    const value = row[column];
    if (typeof value === "boolean") return value ? 1 : 0;
    const numeric = Number(value);
    if (!Number.isNaN(numeric) && (numeric === 0 || numeric === 1)) return numeric;
    const text = String(value || "").trim().toLowerCase();
    const dictionary = {
      "true": 1, "false": 0, "yes": 1, "no": 0, "converted": 1, "not_converted": 0,
      "purchase": 1, "no_purchase": 0, "purchased": 1, "not_purchased": 0
    };
    if (text in dictionary) return dictionary[text];
    return null;
  });
  return mapped.every((value) => value === 0 || value === 1 || value === null) ? mapped : null;
}

function findColumn(rows, keywords, exclude = new Set()) {
  if (!rows.length) return null;
  const columns = Object.keys(rows[0]);
  const normalized = Object.fromEntries(columns.map((column) => [column, normalize(column)]));
  for (const keyword of keywords) {
    const nk = normalize(keyword);
    for (const column of columns) {
      if (exclude.has(column)) continue;
      if (normalized[column].includes(nk)) return column;
    }
  }
  return null;
}

function inferSchema(rows) {
  if (!rows.length) return {};
  const target = findColumn(rows, ["revenue", "converted", "conversion", "purchase", "purchased", "order"]) ||
    Object.keys(rows[0]).find((column) => targetSeries(rows, column));
  const time = findColumn(rows, ["month", "date", "week", "season", "day", "period"], new Set([target].filter(Boolean)));
  const customer = findColumn(rows, ["visitor", "customer", "segment", "member", "cohort", "device", "browser"], new Set([target, time].filter(Boolean)));
  const channel = findColumn(rows, ["traffic", "channel", "source", "campaign", "medium", "acquisition", "referrer"], new Set([target, time, customer].filter(Boolean)));
  const engagement = findColumn(rows, ["pagevalue", "page_value", "duration", "session", "product", "cart", "basket", "engagement"], new Set([target, time, customer, channel].filter(Boolean)));
  const friction = findColumn(rows, ["bounce", "exit", "drop", "abandon", "friction"], new Set([target, time, customer, channel, engagement].filter(Boolean)));
  return { target, time, customer, channel, engagement, friction };
}

function groupMetric(rows, schema, dimension, topN = 8) {
  if (!dimension) return [];
  const target = targetSeries(rows, schema.target);
  const groups = new Map();
  rows.forEach((row, index) => {
    const key = row[dimension];
    if (key === undefined || key === null || key === "") return;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(target ? target[index] : 1);
  });
  return Array.from(groups.entries())
    .map(([key, values]) => ({ key: String(key), value: values.reduce((sum, item) => sum + (item || 0), 0) / values.length }))
    .sort((a, b) => b.value - a.value)
    .slice(0, topN);
}

function bucketMetric(rows, schema) {
  const metricColumn = schema.engagement || schema.friction;
  if (!metricColumn || !isNumericColumn(rows, metricColumn)) return { column: null, items: [] };
  const target = targetSeries(rows, schema.target);
  const numeric = rows
    .map((row, index) => ({ value: Number(row[metricColumn]), target: target ? target[index] : 1 }))
    .filter((item) => !Number.isNaN(item.value));
  if (!numeric.length) return { column: metricColumn, items: [] };
  numeric.sort((a, b) => a.value - b.value);
  const q1 = numeric[Math.floor((numeric.length - 1) * 0.25)].value;
  const q2 = numeric[Math.floor((numeric.length - 1) * 0.5)].value;
  const q3 = numeric[Math.floor((numeric.length - 1) * 0.75)].value;
  const buckets = [
    { label: "Q1", min: -Infinity, max: q1, values: [] },
    { label: "Q2", min: q1, max: q2, values: [] },
    { label: "Q3", min: q2, max: q3, values: [] },
    { label: "Q4", min: q3, max: Infinity, values: [] }
  ];
  numeric.forEach((item) => {
    const bucket = buckets.find((entry) => item.value >= entry.min && item.value <= entry.max);
    if (bucket) bucket.values.push(item.target || 0);
  });
  return {
    column: metricColumn,
    items: buckets.map((bucket) => ({
      key: bucket.label,
      value: bucket.values.length ? bucket.values.reduce((sum, item) => sum + item, 0) / bucket.values.length : 0
    }))
  };
}

function summarizeRows(rows) {
  const schema = inferSchema(rows);
  const target = targetSeries(rows, schema.target);
  const valid = target ? target.filter((value) => value !== null) : [];
  const targetRate = valid.length ? valid.filter((value) => value === 1).length / valid.length : null;
  const timeItems = groupMetric(rows, schema, schema.time);
  const customerItems = groupMetric(rows, schema, schema.customer);
  const channelItems = groupMetric(rows, schema, schema.channel);
  const buckets = bucketMetric(rows, schema);
  return {
    schema,
    rows: rows.length,
    columns: rows.length ? Object.keys(rows[0]).length : 0,
    targetRate,
    topTime: timeItems[0]?.key || "Unavailable",
    topCustomer: customerItems[0]?.key || "Unavailable",
    topChannel: channelItems[0]?.key || "Unavailable",
    timeItems,
    customerItems,
    channelItems,
    bucketColumn: buckets.column,
    bucketItems: buckets.items
  };
}

async function getSummary(rows) {
  if (hasBackend() && getStoredCsv()) {
    try {
      return await fetchApi("/api/summary", { method: "POST", body: makeFormData() });
    } catch (error) {
      console.warn("Backend summary failed, falling back to browser mode.", error);
    }
  }
  const summary = summarizeRows(rows);
  return {
    dataset_name: getDatasetName(),
    rows: summary.rows,
    columns: summary.columns,
    target_rate: summary.targetRate,
    top_time: summary.topTime,
    top_customer: summary.topCustomer,
    top_channel: summary.topChannel,
    schema: summary.schema,
    preview: rows.slice(0, 12)
  };
}

/* ── Topbar ── */
function renderTopbar(page) {
  const nav = [
    ["index.html", "Overview"],
    ["analyst.html", "Analyst Demo"],
    ["evaluation.html", "Evaluation"],
    ["presentation.html", "Presentation"]
  ].map(([href, label]) => `<a class="nav-link ${page === href ? "active" : ""}" href="./${href}">${label}</a>`).join("");

  const host = document.getElementById("topbar");
  host.innerHTML = `
    <div class="topbar-row">
      <div>
        <div class="kicker">Ecommerce Multi-Agent Analytics</div>
        <div class="title">From data to insights to presentation.</div>
        <div class="subtitle">Interrogate your ecommerce data with an AI analyst agent, benchmark models, and export a stakeholder deck.</div>
      </div>
      <div class="nav-links">
        ${nav}
        <span class="status-pill" title="${getDatasetName()}">${getDatasetName()}</span>
      </div>
    </div>
  `;
}

/* ── Upload card (index only) ── */
function bindUploader() {
  const input = document.getElementById("datasetUpload");
  if (!input) return;
  input.addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      saveDataset(file.name, reader.result);
      window.location.reload();
    };
    reader.readAsText(file);
  });
}

function renderUploadCard(summary) {
  const host = document.getElementById("uploadCard");
  if (!host) return;
  host.innerHTML = `
    <div class="upload-card fade-in">
      <div class="section-title">Active dataset</div>
      <div class="section-subtitle">Upload a customer, session, or ecommerce performance CSV to power all tabs.</div>
      <div class="upload-row">
        <input id="datasetUpload" type="file" accept=".csv" />
        <a class="button-link secondary" href="./assets/default_dataset.csv" download>Download sample</a>
        <button class="secondary" id="resetDatasetButton" type="button">Reset to sample</button>
      </div>
      <div class="toolbar" style="margin-top:14px;">
        <span class="tag">Dataset: ${getDatasetName()}</span>
        <span class="tag">Rows: ${summary.rows}</span>
        <span class="tag">Columns: ${summary.columns}</span>
        <span class="tag">Outcome: ${summary.schema?.target || "Not detected"}</span>
      </div>
    </div>
  `;
  bindUploader();
  document.getElementById("resetDatasetButton")?.addEventListener("click", () => {
    localStorage.removeItem(STORAGE_KEYS.csv);
    localStorage.removeItem(STORAGE_KEYS.name);
    window.location.reload();
  });
}

/* ── Shared UI helpers ── */
function metricCard(label, value, caption) {
  return `
    <div class="metric-card fade-in">
      <div class="metric-label">${label}</div>
      <div class="metric-value">${value}</div>
      <div class="metric-caption">${caption}</div>
    </div>
  `;
}

function tableFromRows(rows) {
  if (!rows.length) return "<p class='muted'>No rows available.</p>";
  const columns = Object.keys(rows[0]);
  return `
    <div class="dataset-preview">
      <table>
        <thead><tr>${columns.map((c) => `<th>${c}</th>`).join("")}</tr></thead>
        <tbody>${rows.map((row) => `<tr>${columns.map((c) => `<td>${row[c] ?? ""}</td>`).join("")}</tr>`).join("")}</tbody>
      </table>
    </div>
  `;
}

function renderBarList(items, formatter = (v) => v.toFixed(2)) {
  if (!items.length) return "<p class='muted'>Not enough data to render this comparison.</p>";
  return `<div class="bar-list">${items.map((item) => `
    <div class="bar-item">
      <div class="bar-head"><span>${item.key}</span><strong>${formatter(item.value)}</strong></div>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.max(4, item.value * 100)}%"></div></div>
    </div>`).join("")}</div>`;
}

function retrieveContext(rows, question, topK = 5) {
  const tokens = question.toLowerCase().split(/\W+/).filter(Boolean);
  return rows
    .map((row) => {
      const text = Object.entries(row).map(([k, v]) => `${k}: ${v}`).join(", ");
      const score = tokens.reduce((sum, t) => sum + (text.toLowerCase().includes(t) ? 1 : 0), 0);
      return { text, score };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

function buildAnalystFallback(rows, question) {
  const summary = summarizeRows(rows);
  const context = retrieveContext(rows, question, 5);
  const response = {
    summary: `The active dataset suggests performance concentrates in a few stronger segments and channels, with the detected outcome rate at ${summary.targetRate !== null ? `${(summary.targetRate * 100).toFixed(1)}%` : "an unclear rate"}.`,
    key_insights: [
      `Top customer grouping: ${summary.topCustomer}.`,
      `Top channel grouping: ${summary.topChannel}.`,
      `Top time grouping: ${summary.topTime}.`
    ],
    patterns: [
      `Detected outcome column: ${summary.schema.target || "not clearly available"}.`,
      `Strongest bucketed numeric field: ${summary.bucketColumn || "not clearly available"}.`
    ],
    recommendations: [
      "Prioritize the strongest segment and channel pair for incremental investment.",
      "Audit weaker groups to identify friction or discovery gaps.",
      "Track these same dimensions over time to validate improvement."
    ],
    confidence: summary.schema.target ? "medium" : "low"
  };
  return { response, context };
}

/* ── Overview page ── */
async function renderOverviewPage(rows) {
  const summary = await getSummary(rows);
  renderUploadCard(summary);
  document.getElementById("overviewMetrics").innerHTML = [
    metricCard("Rows", summary.rows.toLocaleString(), "Records in the active dataset"),
    metricCard("Columns", summary.columns, "Available fields"),
    metricCard("Outcome Rate", summary.target_rate !== null && summary.target_rate !== undefined ? `${(summary.target_rate * 100).toFixed(1)}%` : "N/A", "Detected conversion or purchase rate"),
    metricCard("Top Segment", summary.top_customer, "Best-performing customer grouping")
  ].join("");
  document.getElementById("overviewContent").innerHTML = `
    <div class="grid-main">
      <div class="hero-card fade-in">
        <div class="section-title">Detected schema</div>
        <p>Fields automatically inferred from the active dataset.</p>
        <ul>
          <li>Outcome column: <strong>${summary.schema?.target || "Not detected"}</strong></li>
          <li>Customer grouping: <strong>${summary.schema?.customer || "Not detected"}</strong></li>
          <li>Channel grouping: <strong>${summary.schema?.channel || "Not detected"}</strong></li>
          <li>Time grouping: <strong>${summary.schema?.time || "Not detected"}</strong></li>
        </ul>
      </div>
      <div class="card fade-in" style="animation-delay:0.07s">
        <div class="section-title">Quick summary</div>
        <ul>
          <li>Best time period: <strong>${summary.top_time}</strong></li>
          <li>Best channel: <strong>${summary.top_channel}</strong></li>
          <li>Best segment: <strong>${summary.top_customer}</strong></li>
        </ul>
      </div>
    </div>
    <div class="card fade-in" style="margin-top:18px; animation-delay:0.12s">
      <div class="section-title">Dataset preview</div>
      <div class="section-subtitle">First 12 rows from the active dataset.</div>
      ${tableFromRows(summary.preview || rows.slice(0, 12))}
    </div>
  `;
}

/* ── Analyst page ── */
function renderAnalystResponse(responseData) {
  const r = responseData.parsed_response;
  const rawEvidence = responseData.retrieved_context || "";
  const evidenceItems = typeof rawEvidence === "string"
    ? rawEvidence.split("\n\n").map((s) => s.trim()).filter(Boolean)
    : [];

  return `
    <div class="card fade-in">
      <div class="section-title">Analysis</div>
      <div class="response-grid" style="margin-top:12px;">
        <div class="response-section summary">
          <div class="response-label">Summary</div>
          <p>${r.summary}</p>
        </div>
        <div class="response-section insights">
          <div class="response-label">Key Insights</div>
          <ul class="response-list">${r.key_insights.map((i) => `<li>${i}</li>`).join("")}</ul>
        </div>
        <div class="response-section patterns">
          <div class="response-label">Patterns</div>
          <ul class="response-list">${r.patterns.map((p) => `<li>${p}</li>`).join("")}</ul>
        </div>
        <div class="response-section recommendations">
          <div class="response-label">Recommendations</div>
          <ul class="response-list">${r.recommendations.map((rec) => `<li>${rec}</li>`).join("")}</ul>
        </div>
      </div>
      <span class="confidence-badge confidence-${r.confidence || "medium"}">Confidence: ${r.confidence || "medium"}</span>
    </div>
    <div class="card fade-in" style="animation-delay:0.08s">
      <div class="section-title">Retrieved Evidence</div>
      <div class="section-subtitle">Top matching records retrieved from the dataset for this question.</div>
      ${evidenceItems.length
        ? `<div class="evidence-list">${evidenceItems.map((item, idx) => `
            <div class="evidence-item">
              <span class="evidence-num">${idx + 1}</span>
              <span class="evidence-text">${item}</span>
            </div>`).join("")}
          </div>`
        : "<p class='muted'>No retrieved evidence available.</p>"
      }
    </div>
  `;
}

async function renderAnalystPage(rows) {
  // Load model and prompt options from the same CSVs used by evaluation
  let modelRows = [];
  let promptRows = [];
  try {
    [modelRows, promptRows] = await Promise.all([
      fetchCsv("./assets/model_comparison.csv"),
      fetchCsv("./assets/prompt_comparison.csv")
    ]);
  } catch (error) {
    console.warn("Could not load evaluation CSVs for analyst selectors, using fallback options.", error);
  }
  const modelOptions = modelRows.map((r) => r.model).filter(Boolean);
  const promptOptions = promptRows.map((r) => r.prompt_style).filter(Boolean);

  const DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct";
  const DEFAULT_PROMPT = "structured_json";
  const FALLBACK_MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-7b-instruct",
    "google/gemma-2-9b-it",
    "qwen/qwen-2.5-7b-instruct"
  ];
  const FALLBACK_PROMPTS = ["basic", "structured_json", "executive", "evidence_constrained"];
  const finalModelOptions = modelOptions.length ? modelOptions : FALLBACK_MODELS;
  const finalPromptOptions = promptOptions.length ? promptOptions : FALLBACK_PROMPTS;

  const samples = SAMPLE_QUESTIONS.map((q, idx) =>
    `<button class="secondary sample-question" data-question="${q}">Sample ${idx + 1}</button>`
  ).join("");

  const modelSelect = `
    <div class="select-group">
      <label class="select-label" for="modelSelect">Model</label>
      <select id="modelSelect">
        ${finalModelOptions.map((m) => `<option value="${m}" ${m === DEFAULT_MODEL ? "selected" : ""}>${m.split("/").pop()}</option>`).join("")}
      </select>
    </div>`;

  const promptSelect = `
    <div class="select-group">
      <label class="select-label" for="promptSelect">Prompt style</label>
      <select id="promptSelect">
        ${finalPromptOptions.map((p) => `<option value="${p}" ${p === DEFAULT_PROMPT ? "selected" : ""}>${p.replace(/_/g, " ")}</option>`).join("")}
      </select>
    </div>`;

  const ragToggle = `
    <div class="select-group">
      <label class="select-label" for="ragSelect">Retrieval (RAG)</label>
      <select id="ragSelect">
        <option value="true" selected>Enabled</option>
        <option value="false">Disabled</option>
      </select>
    </div>`;

  document.getElementById("analystPage").innerHTML = `
    <div class="card fade-in">
      <div class="section-title">Analyst Agent Demo</div>
      <div class="section-subtitle">Ask a business question — the agent retrieves relevant data, reasons over it, and returns a structured answer.</div>
      <div class="toolbar" style="margin-bottom:14px;">${samples}</div>
      <textarea id="questionInput" placeholder="Enter a business question...">${SAMPLE_QUESTIONS[0]}</textarea>
      <div class="config-row" style="margin-top:14px;">
        ${modelSelect}${promptSelect}${ragToggle}
      </div>
      <div class="toolbar" style="margin-top:14px;">
        <button id="runAnalystButton" type="button">Run analysis</button>
      </div>
    </div>
    <div id="analystResults" style="margin-top:18px; display:grid; gap:16px;"></div>
  `;

  document.querySelectorAll(".sample-question").forEach((btn) =>
    btn.addEventListener("click", () => {
      document.getElementById("questionInput").value = btn.dataset.question;
    })
  );

  async function run() {
    const btn = document.getElementById("runAnalystButton");
    const question = document.getElementById("questionInput").value.trim();
    if (!question) return;

    const model       = document.getElementById("modelSelect").value;
    const promptStyle = document.getElementById("promptSelect").value;
    const ragEnabled  = document.getElementById("ragSelect").value === "true";

    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Analyzing…`;
    document.getElementById("analystResults").innerHTML = "";

    let responseData;
    if (hasBackend() && getStoredCsv()) {
      try {
        responseData = await fetchApi("/api/analyst", {
          method: "POST",
          body: makeFormData({
            question,
            model,
            prompt_style: promptStyle,
            rag_enabled: ragEnabled,
            top_k: 5
          })
        });
      } catch (error) {
        console.warn("Backend analyst call failed, falling back.", error);
      }
    }

    if (!responseData) {
      const fallback = buildAnalystFallback(rows, question);
      responseData = {
        parsed_response: fallback.response,
        retrieved_context: fallback.context.map((item, idx) => `[${idx + 1}] ${item.text}`).join("\n\n")
      };
    }

    document.getElementById("analystResults").innerHTML = renderAnalystResponse(responseData);
    btn.disabled = false;
    btn.innerHTML = "Run analysis";
  }

  document.getElementById("runAnalystButton").addEventListener("click", run);
  run();
}

/* ── Evaluation page ── */
async function fetchCsv(path) {
  const response = await fetch(path);
  const text = await response.text();
  return Papa.parse(text, { header: true, dynamicTyping: true, skipEmptyLines: true }).data;
}

async function renderEvaluationPage() {
  const [models, prompts, rag] = await Promise.all([
    fetchCsv("./assets/model_comparison.csv"),
    fetchCsv("./assets/prompt_comparison.csv"),
    fetchCsv("./assets/rag_comparison.csv")
  ]);

  const ragOn  = Number(rag.find((r) => String(r.rag_enabled).toLowerCase() === "true")?.overall_score  || 0).toFixed(2);
  const ragOff = Number(rag.find((r) => String(r.rag_enabled).toLowerCase() === "false")?.overall_score || 0).toFixed(2);

  document.getElementById("evaluationPage").innerHTML = `
    <div class="grid-4 fade-in">
      ${metricCard("Best Model",  models[0]?.model  || "N/A", "Highest overall benchmark score")}
      ${metricCard("Best Prompt", prompts[0]?.prompt_style || "N/A", "Highest overall benchmark score")}
      ${metricCard("RAG On",  ragOn,  "Overall score with retrieval")}
      ${metricCard("RAG Off", ragOff, "Overall score without retrieval")}
    </div>

    <div class="card fade-in" style="margin-top:18px; animation-delay:0.06s">
      <div class="section-title">Benchmark metrics explained</div>
      <ul>
        <li><strong>keyword_score</strong> — expected business keyword coverage</li>
        <li><strong>recommendation_score</strong> — presence and actionability of recommendations</li>
        <li><strong>completeness_score</strong> — completeness of the required JSON schema</li>
        <li><strong>groundedness_score</strong> — overlap with retrieved evidence</li>
        <li><strong>overall_score</strong> — average of the four core metrics</li>
      </ul>
    </div>

    <div class="card fade-in" style="margin-top:18px; animation-delay:0.1s">
      <div class="eval-section-header">
        <div class="section-title" style="margin:0;">Model performance</div>
        <span class="eval-section-pill">Models</span>
      </div>
      <div class="section-subtitle">Overall score by model — higher is better.</div>
      ${renderBarList(models.map((r) => ({ key: r.model, value: Number(r.overall_score) })))}
      <hr class="section-divider">
      <div class="section-title" style="margin-bottom:12px;">Detailed results</div>
      ${tableFromRows(models)}
    </div>

    <div class="card fade-in" style="margin-top:18px; animation-delay:0.14s">
      <div class="eval-section-header">
        <div class="section-title" style="margin:0;">Prompt performance</div>
        <span class="eval-section-pill">Prompts</span>
      </div>
      <div class="section-subtitle">Overall score by prompt style — higher is better.</div>
      ${renderBarList(prompts.map((r) => ({ key: r.prompt_style, value: Number(r.overall_score) })))}
      <hr class="section-divider">
      <div class="section-title" style="margin-bottom:12px;">Detailed results</div>
      ${tableFromRows(prompts)}
    </div>

    <div class="card fade-in" style="margin-top:18px; animation-delay:0.18s">
      <div class="eval-section-header">
        <div class="section-title" style="margin:0;">RAG comparison</div>
        <span class="eval-section-pill">Retrieval</span>
      </div>
      <div class="section-subtitle">How retrieval-augmented generation affects answer quality.</div>
      ${tableFromRows(rag)}
    </div>
  `;
}

/* ── Presentation page ── */
async function renderPresentationPage(rows) {
  let slides;
  let downloadUrl = "./assets/presentation.pptx";

  if (hasBackend() && getStoredCsv()) {
    try {
      const response = await fetchApi("/api/presentation", {
        method: "POST",
        body: makeFormData()
      });
      downloadUrl = `${API_BASE_URL}${response.download_url}`;
      slides = response.slides.map((slide) => ({
        ...slide,
        chart_url: slide.chart_url ? `${API_BASE_URL}${slide.chart_url}` : null
      }));
    } catch (error) {
      console.warn("Backend presentation call failed, falling back to static preview.", error);
    }
  }

  if (!slides) {
    const summary = await getSummary(rows);
    const rateText = summary.target_rate !== null && summary.target_rate !== undefined
      ? `${(summary.target_rate * 100).toFixed(1)}% detected outcome rate`
      : "Outcome rate not detected";
    slides = [
      {
        title: "E-Commerce Customer Behavior Insights",
        speaker_notes: "Introduce the dataset story and frame the deck as a business-ready summary of customer behavior patterns.",
        bullets: [
          "This presentation summarizes the uploaded ecommerce dataset in a stakeholder-friendly format.",
          `The dataset includes ${summary.rows.toLocaleString()} rows and ${summary.columns} columns of customer, channel, and behavioral information.`,
          rateText + "."
        ],
        chart_url: null
      },
      {
        title: "Dataset Snapshot",
        speaker_notes: "Explain how the uploaded dataset was interpreted and which fields drove the analysis.",
        bullets: [
          "The dataset has been automatically profiled to detect customer segments, channel information, and business outcomes.",
          `Peak performance currently appears in ${summary.top_time}.`,
          `Most informative customer grouping detected: ${summary.schema?.customer || "not clearly available"}.`,
          `Most informative acquisition grouping detected: ${summary.schema?.channel || "not clearly available"}.`
        ],
        chart_url: "./assets/time_performance.png"
      },
      {
        title: "Target Customers",
        speaker_notes: "Highlight which customers appear most valuable and where acquisition quality is strongest.",
        bullets: [
          `The strongest customer segment is ${summary.top_customer}.`,
          `The strongest acquisition or channel grouping is ${summary.top_channel}.`,
          "Priority audiences combine stronger conversion signals with stronger engagement behavior.",
          "These segments are the best candidates for campaign prioritization and personalized targeting."
        ],
        chart_url: "./assets/customer_performance.png"
      },
      {
        title: "Behavioral Patterns",
        speaker_notes: "Use this slide to explain the behavioral signals that separate stronger sessions from weaker ones.",
        bullets: [
          `Engagement metric detected: ${summary.schema?.engagement || "not clearly available"}.`,
          `Friction metric detected: ${summary.schema?.friction || "not clearly available"}.`,
          "Bucket analysis shows where commercial performance improves across engagement quartiles.",
          "Behavior and customer quality are working together rather than independently."
        ],
        chart_url: "./assets/engagement_buckets.png"
      },
      {
        title: "Key Findings",
        speaker_notes: "Condense the analysis into the most portable executive findings.",
        bullets: [
          `Top time period: ${summary.top_time}.`,
          `Top customer segment: ${summary.top_customer}.`,
          `Top channel grouping: ${summary.top_channel}.`,
          "The dataset suggests that acquisition quality and on-site engagement both influence business outcomes."
        ],
        chart_url: "./assets/channel_performance.png"
      },
      {
        title: "Recommendations",
        speaker_notes: "Translate the patterns into actions across marketing, merchandising, and experience optimization.",
        bullets: [
          "Focus budget on the strongest channels and customer groups identified in the analysis.",
          "Improve weak customer journeys by reducing friction and strengthening discovery paths.",
          "Use high-intent engagement signals to trigger tailored offers, retention flows, or remarketing.",
          "Track the same dimensions over time to validate whether conversion improves after changes."
        ],
        chart_url: null
      },
      {
        title: "Conclusion",
        speaker_notes: "Close by emphasizing speed, clarity, and repeatability for future ecommerce datasets.",
        bullets: [
          "The uploaded ecommerce dataset can be turned into a clear commercial story without manual chart building.",
          "Target segments, channel quality, and engagement behavior are the main drivers surfaced by the analysis.",
          "This workflow is designed to help analysts move quickly from raw data to presentation-ready outputs."
        ],
        chart_url: null
      }
    ];
  }

  document.getElementById("presentationPage").innerHTML = `
    <div class="card fade-in">
      <div class="section-title">Stakeholder presentation</div>
      <div class="section-subtitle">A ready-to-share PowerPoint deck with dataset insights and strategic recommendations.</div>
      <div class="toolbar">
        <a class="button-link" href="${downloadUrl}" download>Download PowerPoint</a>
        <span class="tag">${slides.length} slides</span>
      </div>
    </div>
    <div class="slide-grid" style="margin-top:18px;">
      ${slides.map((slide, idx) => `
        <div class="preview-card slide fade-in" style="animation-delay:${0.06 * idx}s">
          <div>
            <div class="slide-num">${idx + 1}</div>
            <h3>${slide.title}</h3>
            <p class="slide-notes">${slide.speaker_notes}</p>
            <ul>${slide.bullets.map((b) => `<li>${b}</li>`).join("")}</ul>
          </div>
          <div class="preview-chart">
            ${slide.chart_url
              ? `<img src="${slide.chart_url}" alt="${slide.title}" loading="lazy" />`
              : "<p class='muted small'>No chart for this slide.</p>"
            }
          </div>
        </div>
      `).join("")}
    </div>
  `;
}

/* ── Init ── */
async function init() {
  const page = document.body.dataset.page;
  renderTopbar(page);
  const rows = await getDataset();
  if (page === "index.html") await renderOverviewPage(rows);
  else if (page === "analyst.html") await renderAnalystPage(rows);
  else if (page === "evaluation.html") await renderEvaluationPage();
  else if (page === "presentation.html") await renderPresentationPage(rows);
}

document.addEventListener("DOMContentLoaded", init);
