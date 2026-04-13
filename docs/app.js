const SAMPLE_QUESTIONS = [
  "Which customer segments should we prioritize based on this ecommerce dataset?",
  "What patterns distinguish stronger-performing sessions or customers?",
  "Which channels appear most promising for future budget allocation?",
  "What recommendations would improve performance based on this data?"
];

const STORAGE_KEYS = {
  csv: "ise547_dataset_csv",
  name: "ise547_dataset_name",
  sampling: "ise547_dataset_sampling",
  loading: "ise547_dataset_loading"
};

const API_BASE_URL = (window.APP_CONFIG?.API_BASE_URL || "").replace(/\/$/, "");
const MAX_ANALYSIS_ROWS = 25000;

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
  localStorage.setItem(STORAGE_KEYS.sampling, JSON.stringify({
    sampled: false,
    sourceRows: parseCsvText(text).data.length,
    analysisRows: parseCsvText(text).data.length
  }));
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

function getSamplingInfo() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEYS.sampling) || "null");
  } catch (_) {
    return null;
  }
}

function getLoadingInfo() {
  try {
    return JSON.parse(sessionStorage.getItem(STORAGE_KEYS.loading) || "null");
  } catch (_) {
    return null;
  }
}

function setLoadingInfo(info) {
  if (!info) {
    sessionStorage.removeItem(STORAGE_KEYS.loading);
    return;
  }
  sessionStorage.setItem(STORAGE_KEYS.loading, JSON.stringify(info));
}

function saveDataset(fileName, csvText, samplingInfo = null) {
  localStorage.setItem(STORAGE_KEYS.csv, csvText);
  localStorage.setItem(STORAGE_KEYS.name, fileName);
  if (samplingInfo) {
    localStorage.setItem(STORAGE_KEYS.sampling, JSON.stringify(samplingInfo));
  } else {
    localStorage.removeItem(STORAGE_KEYS.sampling);
  }
}

function sampleRows(rows, limit = MAX_ANALYSIS_ROWS) {
  if (rows.length <= limit) return rows;
  const sampled = [];
  const step = rows.length / limit;
  for (let i = 0; i < limit; i += 1) {
    const index = Math.min(rows.length - 1, Math.floor(i * step));
    sampled.push(rows[index]);
  }
  return sampled;
}

function showLoadingOverlay(message, detail = "This can take a few seconds for larger datasets.") {
  let overlay = document.getElementById("loadingOverlay");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.id = "loadingOverlay";
    overlay.className = "loading-overlay";
    overlay.innerHTML = `
      <div class="loading-card">
        <div class="loading-spinner"></div>
        <div class="loading-title" id="loadingTitle"></div>
        <div class="loading-detail" id="loadingDetail"></div>
      </div>
    `;
    document.body.appendChild(overlay);
  }
  document.getElementById("loadingTitle").textContent = message;
  document.getElementById("loadingDetail").textContent = detail;
  overlay.classList.add("visible");
}

function hideLoadingOverlay() {
  document.getElementById("loadingOverlay")?.classList.remove("visible");
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

function missingSummary(rows) {
  if (!rows.length) return [];
  const columns = Object.keys(rows[0]);
  return columns
    .map((column) => {
      const missing = rows.filter((row) => row[column] === null || row[column] === undefined || row[column] === "").length;
      return { column, missing_pct: Number(((missing / Math.max(rows.length, 1)) * 100).toFixed(2)) };
    })
    .sort((a, b) => b.missing_pct - a.missing_pct)
    .slice(0, 12);
}

function buildEdaFallback(rows) {
  const summary = summarizeRows(rows);
  const missing = missingSummary(rows);
  const warnings = summary.schema?.target
    ? ["Browser fallback mode is active, so advanced backend EDA diagnostics are limited."]
    : ["No clean target column was detected in browser fallback mode."];
  const suggestedQuestions = [];
  if (summary.schema?.customer) suggestedQuestions.push(`Which ${summary.schema.customer} groups should we prioritize?`);
  if (summary.schema?.channel) suggestedQuestions.push(`Which ${summary.schema.channel} groups appear strongest?`);
  if (summary.schema?.time) suggestedQuestions.push(`How does performance change across ${summary.schema.time}?`);
  if (summary.schema?.engagement) suggestedQuestions.push(`How does ${summary.schema.engagement} relate to the detected outcome?`);
  if (!suggestedQuestions.length) suggestedQuestions.push("Which dimensions in this dataset appear most commercially meaningful?");
  return {
    profile: {
      source_format: "tabular",
      analysis_grain: "rows",
      raw_rows: summary.rows,
      analysis_rows: summary.rows,
      columns: summary.columns,
      schema: summary.schema,
      target_rate: summary.targetRate,
      top_time: summary.topTime,
      top_customer: summary.topCustomer,
      top_channel: summary.topChannel,
      top_product_dimension: "Unavailable",
      numeric_columns: []
    },
    quality_checks: {
      duplicate_rows: 0,
      missing_summary: missing,
      null_heavy_columns: missing.filter((row) => row.missing_pct > 20),
      constant_columns: [],
      high_cardinality_columns: [],
      numeric_outliers: [],
      warnings
    },
    chart_manifest: [],
    suggested_questions: suggestedQuestions,
    handoff_summary: {
      dataset_type: "tabular",
      analysis_grain: "rows",
      target_column: summary.schema?.target || null,
      time_column: summary.schema?.time || null,
      customer_dimensions: summary.schema?.customer ? [summary.schema.customer] : [],
      channel_dimensions: summary.schema?.channel ? [summary.schema.channel] : [],
      engagement_metric: summary.schema?.engagement || null,
      friction_metric: summary.schema?.friction || null,
      quality_warnings: warnings,
      top_patterns: [
        `Top customer grouping: ${summary.topCustomer}.`,
        `Top channel grouping: ${summary.topChannel}.`,
        `Top time grouping: ${summary.topTime}.`
      ],
      recommended_questions: suggestedQuestions
    },
    retrieval_chunks: [
      `Browser fallback EDA profile: rows=${summary.rows}, columns=${summary.columns}.`,
      `Detected schema: target=${summary.schema?.target || "none"}, customer=${summary.schema?.customer || "none"}, channel=${summary.schema?.channel || "none"}, time=${summary.schema?.time || "none"}.`
    ],
    key_findings: [
      summary.targetRate !== null ? `Detected outcome rate is ${(summary.targetRate * 100).toFixed(1)}%.` : "No clear outcome rate detected.",
      `Top customer grouping: ${summary.topCustomer}.`,
      `Top channel grouping: ${summary.topChannel}.`
    ]
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
    ["eda.html", "EDA"],
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
  input.addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    showLoadingOverlay("Uploading dataset", "Preparing the file, applying sampling if needed, and refreshing the overview.");
    const reader = new FileReader();
    reader.onload = async () => {
      const rawText = String(reader.result || "");
      const parsed = parseCsvText(rawText);
      const rows = parsed.data || [];
      const sampledRows = sampleRows(rows);
      const samplingInfo = {
        sampled: rows.length > sampledRows.length,
        sourceRows: rows.length,
        analysisRows: sampledRows.length
      };
      setLoadingInfo({
        active: true,
        fileName: file.name,
        sampled: samplingInfo.sampled,
        sourceRows: samplingInfo.sourceRows,
        analysisRows: samplingInfo.analysisRows
      });
      const csvToStore = samplingInfo.sampled ? Papa.unparse(sampledRows) : rawText;
      saveDataset(file.name, csvToStore, samplingInfo);
      window.location.reload();
    };
    reader.readAsText(file);
  });
}

function renderUploadCard(summary) {
  const host = document.getElementById("uploadCard");
  if (!host) return;
  const samplingInfo = getSamplingInfo();
  const loadingInfo = getLoadingInfo();
  const analysisTag = summary.analysis_rows && summary.analysis_grain
    ? `<span class="tag">Analysis: ${Number(summary.analysis_rows).toLocaleString()} ${summary.analysis_grain}</span>`
    : "";
  const sampledTag = samplingInfo?.sampled
    ? `<span class="tag">Sampled: ${Number(samplingInfo.analysisRows).toLocaleString()} of ${Number(samplingInfo.sourceRows).toLocaleString()} rows</span>`
    : "";
  const loadingNote = loadingInfo?.active
    ? `<div class="loading-inline"><span class="inline-spinner"></span><span>Refreshing the overview and regenerating insights for <strong>${loadingInfo.fileName}</strong>${loadingInfo.sampled ? ` using a ${Number(loadingInfo.analysisRows).toLocaleString()}-row sample` : ""}.</span></div>`
    : "";
  host.innerHTML = `
    <div class="upload-card fade-in">
      <div class="section-title">Active dataset</div>
      <div class="section-subtitle">Upload a customer, session, or ecommerce performance CSV to power all tabs. Large files are sampled automatically for responsive analysis.</div>
      ${loadingNote}
      <div class="upload-row">
        <input id="datasetUpload" type="file" accept=".csv" />
        <a class="button-link secondary" href="./assets/default_dataset.csv" download>Download sample</a>
        <button class="secondary" id="resetDatasetButton" type="button">Reset to sample</button>
      </div>
      <div class="toolbar" style="margin-top:14px;">
        <span class="tag">Dataset: ${getDatasetName()}</span>
        <span class="tag">Rows: ${summary.rows}</span>
        <span class="tag">Columns: ${summary.columns}</span>
        ${analysisTag}
        ${sampledTag}
        <span class="tag">Outcome: ${summary.schema?.target || "Not detected"}</span>
      </div>
    </div>
  `;
  bindUploader();
  document.getElementById("resetDatasetButton")?.addEventListener("click", () => {
    localStorage.removeItem(STORAGE_KEYS.csv);
    localStorage.removeItem(STORAGE_KEYS.name);
    localStorage.removeItem(STORAGE_KEYS.sampling);
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

function renderBulletList(items, emptyText = "No items available.") {
  if (!items || !items.length) return `<p class="muted">${emptyText}</p>`;
  return `<ul>${items.map((item) => `<li>${item}</li>`).join("")}</ul>`;
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
  hideLoadingOverlay();
  setLoadingInfo(null);
  document.getElementById("overviewMetrics").innerHTML = [
    metricCard("Rows", summary.rows.toLocaleString(), "Records in the active dataset"),
    metricCard("Columns", summary.columns, "Available fields"),
    metricCard("Outcome Rate", summary.target_rate !== null && summary.target_rate !== undefined ? `${(summary.target_rate * 100).toFixed(1)}%` : "N/A", "Detected conversion or purchase rate"),
    metricCard(
      summary.analysis_grain ? "Analysis Grain" : "Top Segment",
      summary.analysis_grain ? `${summary.analysis_rows?.toLocaleString?.() || summary.analysis_rows} ${summary.analysis_grain}` : summary.top_customer,
      summary.analysis_grain ? "Normalized records used for analysis" : "Best-performing customer grouping"
    )
  ].join("");
  document.getElementById("overviewContent").innerHTML = `
    <div class="grid-main">
      <div class="hero-card fade-in">
        <div class="section-title">Detected schema</div>
        <p>Fields automatically inferred from the active dataset.</p>
        <ul>
          <li>Source format: <strong>${summary.source_format || "tabular"}</strong></li>
          <li>Outcome column: <strong>${summary.schema?.target || "Not detected"}</strong></li>
          <li>Customer grouping: <strong>${summary.schema?.customer || "Not detected"}</strong></li>
          <li>Channel grouping: <strong>${summary.schema?.channel || "Not detected"}</strong></li>
          <li>Time grouping: <strong>${summary.schema?.time || "Not detected"}</strong></li>
        </ul>
      </div>
      <div class="card fade-in" style="animation-delay:0.07s">
        <div class="section-title">Quick summary</div>
        <ul>
          ${getSamplingInfo()?.sampled ? `<li>Large upload sampled to <strong>${Number(getSamplingInfo().analysisRows).toLocaleString()}</strong> rows from <strong>${Number(getSamplingInfo().sourceRows).toLocaleString()}</strong>.</li>` : ""}
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

async function renderEdaPage(rows) {
  let report;
  if (hasBackend() && getStoredCsv()) {
    try {
      report = await fetchApi("/api/eda", { method: "POST", body: makeFormData() });
      report.chart_manifest = (report.chart_manifest || []).map((chart) => ({
        ...chart,
        chart_url: chart.chart_url ? `${API_BASE_URL}${chart.chart_url}` : null
      }));
    } catch (error) {
      console.warn("Backend EDA call failed, falling back to browser mode.", error);
    }
  }
  if (!report) report = buildEdaFallback(rows);

  const profile = report.profile || {};
  const quality = report.quality_checks || {};
  const schema = profile.schema || {};
  const missingRows = (quality.missing_summary || []).slice(0, 8);

  document.getElementById("edaPage").innerHTML = `
    <div class="grid-4 fade-in">
      ${metricCard("Source Format", profile.source_format || "Unknown", "Detected dataset structure")}
      ${metricCard("Analysis Grain", `${profile.analysis_rows?.toLocaleString?.() || profile.analysis_rows || 0} ${profile.analysis_grain || "rows"}`, "Normalized records used for EDA")}
      ${metricCard("Outcome Rate", profile.target_rate !== null && profile.target_rate !== undefined ? `${(profile.target_rate * 100).toFixed(1)}%` : "N/A", "Detected target or purchase rate")}
      ${metricCard("Top Segment", profile.top_customer || "Unavailable", "Strongest detected grouping")}
    </div>

    <div class="grid-main" style="margin-top:18px;">
      <div class="hero-card fade-in">
        <div class="section-title">Dataset profile</div>
        <p class="section-subtitle">Deterministic profiling runs before the analyst agent so the rest of the workflow operates from a cleaner business view.</p>
        <ul>
          <li>Source format: <strong>${profile.source_format || "Unknown"}</strong></li>
          <li>Analysis grain: <strong>${profile.analysis_grain || "rows"}</strong></li>
          <li>Target column: <strong>${schema.target || "Not detected"}</strong></li>
          <li>Time column: <strong>${schema.time || "Not detected"}</strong></li>
          <li>Customer grouping: <strong>${schema.customer || "Not detected"}</strong></li>
          <li>Channel grouping: <strong>${schema.channel || "Not detected"}</strong></li>
        </ul>
      </div>
      <div class="card fade-in" style="animation-delay:0.05s">
        <div class="section-title">Suggested business questions</div>
        <div class="section-subtitle">Questions generated from the detected schema and strongest patterns.</div>
        ${renderBulletList(report.suggested_questions || [], "Suggested questions are unavailable for this dataset.")}
      </div>
    </div>

    <div class="grid-2" style="margin-top:18px;">
      <div class="card fade-in">
        <div class="section-title">Quality checks</div>
        <div class="section-subtitle">Key warnings and structural issues found before analysis.</div>
        ${renderBulletList((quality.warnings || []).concat((quality.constant_columns || []).length ? [`Constant columns: ${(quality.constant_columns || []).join(", ")}`] : []))}
      </div>
      <div class="card fade-in" style="animation-delay:0.05s">
        <div class="section-title">Missingness snapshot</div>
        <div class="section-subtitle">Columns with the highest share of missing values.</div>
        ${missingRows.length ? tableFromRows(missingRows) : "<p class='muted'>No major missingness issues detected.</p>"}
      </div>
    </div>

    <div class="card fade-in" style="margin-top:18px;">
      <div class="section-title">Auto-generated charts</div>
      <div class="section-subtitle">These visuals are generated directly from the uploaded dataset before the analyst agent answers any question.</div>
      <div class="slide-grid" style="margin-top:14px;">
        ${(report.chart_manifest || []).length ? report.chart_manifest.map((chart, idx) => `
          <div class="preview-card slide" style="animation-delay:${0.04 * idx}s">
            <div>
              <h3>${chart.title}</h3>
              <p class="slide-notes">${chart.caption}</p>
            </div>
            <div class="preview-chart">
              ${chart.chart_url ? `<img src="${chart.chart_url}" alt="${chart.title}" loading="lazy" />` : "<p class='muted small'>Backend chart unavailable in fallback mode.</p>"}
            </div>
          </div>
        `).join("") : "<p class='muted'>Charts are unavailable in browser fallback mode.</p>"}
      </div>
    </div>

    <div class="grid-2" style="margin-top:18px;">
      <div class="card fade-in">
        <div class="section-title">Key findings</div>
        <div class="section-subtitle">Portable EDA insights the analyst and presentation agents can build on.</div>
        ${renderBulletList(report.key_findings || [], "No findings available.")}
      </div>
      <div class="card fade-in" style="animation-delay:0.05s">
        <div class="section-title">Handoff summary for analyst</div>
        <div class="section-subtitle">Structured context passed into the analyst stage.</div>
        <div class="response-section summary"><pre style="margin:0;white-space:pre-wrap;">${JSON.stringify(report.handoff_summary || {}, null, 2)}</pre></div>
      </div>
    </div>
  `;
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
  if (page === "index.html" && getLoadingInfo()?.active) {
    showLoadingOverlay("Generating overview insights", "Your dataset is loaded. We are updating the overview cards, schema, and preview.");
  }
  const rows = await getDataset();
  if (page === "index.html") await renderOverviewPage(rows);
  else if (page === "eda.html") await renderEdaPage(rows);
  else if (page === "analyst.html") await renderAnalystPage(rows);
  else if (page === "evaluation.html") await renderEvaluationPage();
  else if (page === "presentation.html") await renderPresentationPage(rows);
}

document.addEventListener("DOMContentLoaded", init);
