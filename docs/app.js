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
        <div class="title">Public multi-page site with backend-powered AI agents.</div>
        <div class="subtitle">
          GitHub Pages hosts the frontend, while the FastAPI backend handles analyst answers and presentation generation when an API URL is configured.
        </div>
      </div>
      <div class="nav-links">
        ${nav}
        <span class="status-pill">${getDatasetName()}</span>
      </div>
    </div>
  `;
}

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
    <div class="upload-card">
      <div class="section-title">Active dataset</div>
      <div class="section-subtitle">Upload a customer, session, or ecommerce performance CSV. If an API backend is configured, the AI-agent pages will send this dataset to the backend for live analysis.</div>
      <div class="upload-row">
        <input id="datasetUpload" type="file" accept=".csv" />
        <a class="button-link secondary" href="./assets/default_dataset.csv" download>Download sample CSV</a>
        <button class="secondary" id="resetDatasetButton" type="button">Reset to sample</button>
      </div>
      <div class="toolbar" style="margin-top:14px;">
        <span class="tag">Dataset: ${getDatasetName()}</span>
        <span class="tag">Rows: ${summary.rows}</span>
        <span class="tag">Columns: ${summary.columns}</span>
        <span class="tag">Outcome: ${summary.schema?.target || "Not detected"}</span>
        <span class="tag">${hasBackend() ? "Backend connected" : "Browser fallback mode"}</span>
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

function metricCard(label, value, caption) {
  return `
    <div class="metric-card">
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
        <thead><tr>${columns.map((column) => `<th>${column}</th>`).join("")}</tr></thead>
        <tbody>${rows.map((row) => `<tr>${columns.map((column) => `<td>${row[column] ?? ""}</td>`).join("")}</tr>`).join("")}</tbody>
      </table>
    </div>
  `;
}

function renderBarList(items, formatter = (value) => value.toFixed(2)) {
  if (!items.length) return "<p class='muted'>Not enough information to render this comparison.</p>";
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
      const text = Object.entries(row).map(([key, value]) => `${key}: ${value}`).join(", ");
      const score = tokens.reduce((sum, token) => sum + (text.toLowerCase().includes(token) ? 1 : 0), 0);
      return { text, score };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

function buildAnalystFallback(rows, question) {
  const summary = summarizeRows(rows);
  const context = retrieveContext(rows, question, 5);
  const response = {
    summary: `The active dataset suggests performance concentrates in a few stronger segments and channels, with the current detected outcome rate at ${summary.targetRate !== null ? `${(summary.targetRate * 100).toFixed(1)}%` : "an unclear rate"}.`,
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

async function renderOverviewPage(rows) {
  const summary = await getSummary(rows);
  renderUploadCard(summary);
  document.getElementById("overviewMetrics").innerHTML = [
    metricCard("Rows", summary.rows, "Records in the active dataset"),
    metricCard("Columns", summary.columns, "Available fields"),
    metricCard("Outcome Rate", summary.target_rate !== null && summary.target_rate !== undefined ? `${(summary.target_rate * 100).toFixed(1)}%` : "N/A", "Detected conversion or purchase rate"),
    metricCard("Top Segment", summary.top_customer, "Best-performing customer grouping")
  ].join("");
  document.getElementById("overviewContent").innerHTML = `
    <div class="grid-main">
      <div class="hero-card">
        <div class="section-title">Current dataset context</div>
        <p>This overview uses the active uploaded dataset and can call the backend for schema-aware summarization when available.</p>
        <ul>
          <li>Detected outcome column: ${summary.schema?.target || "Not detected"}</li>
          <li>Detected customer grouping: ${summary.schema?.customer || "Not detected"}</li>
          <li>Detected channel grouping: ${summary.schema?.channel || "Not detected"}</li>
          <li>Detected time grouping: ${summary.schema?.time || "Not detected"}</li>
        </ul>
      </div>
      <div class="hero-card">
        <div class="section-title">Deployment mode</div>
        <ul>
          <li>Frontend: GitHub Pages static site</li>
          <li>Backend: ${hasBackend() ? "Connected FastAPI service" : "Not configured"}</li>
          <li>Analyst and presentation pages use backend AI when available</li>
          <li>Evaluation remains based on exported benchmark artifacts</li>
        </ul>
      </div>
    </div>
    <div class="card" style="margin-top:18px;">
      <div class="section-title">Dataset preview</div>
      <div class="section-subtitle">First 12 rows from the active dataset.</div>
      ${tableFromRows(summary.preview || rows.slice(0, 12))}
    </div>
  `;
}

async function renderAnalystPage(rows) {
  const summary = await getSummary(rows);
  renderUploadCard(summary);
  const samples = SAMPLE_QUESTIONS.map((question, idx) => `<button class="secondary sample-question" data-question="${question}">Sample ${idx + 1}</button>`).join("");
  document.getElementById("analystPage").innerHTML = `
    <div class="card">
      <div class="section-title">Analyst Agent Demo</div>
      <div class="section-subtitle">${hasBackend() ? "This page is connected to the FastAPI backend, so the analyst response is generated by the Python agent." : "This page is using browser fallback mode because no backend API URL is configured."}</div>
      <div class="toolbar">${samples}</div>
      <div style="margin-top:14px;"><textarea id="questionInput">${SAMPLE_QUESTIONS[0]}</textarea></div>
      <div class="toolbar" style="margin-top:14px;">
        <button id="runAnalystButton" type="button">Run analysis</button>
      </div>
    </div>
    <div id="analystResults" class="grid-main" style="margin-top:18px;"></div>
  `;
  document.querySelectorAll(".sample-question").forEach((button) => button.addEventListener("click", () => {
    document.getElementById("questionInput").value = button.dataset.question;
  }));

  async function run() {
    const question = document.getElementById("questionInput").value;
    let responseData;
    if (hasBackend() && getStoredCsv()) {
      try {
        responseData = await fetchApi("/api/analyst", {
          method: "POST",
          body: makeFormData({
            question,
            model: "meta-llama/llama-3.1-8b-instruct",
            prompt_style: "structured_json",
            rag_enabled: true,
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
    document.getElementById("analystResults").innerHTML = `
      <div class="card">
        <div class="section-title">Structured text response</div>
        <div class="summary-box">Summary\n${responseData.parsed_response.summary}\n\nKey Insights\n- ${responseData.parsed_response.key_insights.join("\n- ")}\n\nPatterns\n- ${responseData.parsed_response.patterns.join("\n- ")}\n\nRecommendations\n- ${responseData.parsed_response.recommendations.join("\n- ")}\n\nConfidence\n${responseData.parsed_response.confidence}</div>
        <div class="section-title" style="margin-top:16px;">Retrieved evidence</div>
        <div class="context-box">${responseData.retrieved_context || "No retrieved context returned."}</div>
      </div>
      <div class="card">
        <div class="section-title">JSON output</div>
        <pre class="json-box">${JSON.stringify(responseData.parsed_response, null, 2)}</pre>
      </div>
    `;
  }

  document.getElementById("runAnalystButton").addEventListener("click", run);
  run();
}

async function fetchCsv(path) {
  const response = await fetch(path);
  const text = await response.text();
  return Papa.parse(text, { header: true, dynamicTyping: true, skipEmptyLines: true }).data;
}

async function renderEvaluationPage(rows) {
  const summary = await getSummary(rows);
  renderUploadCard(summary);
  const [models, prompts, rag] = await Promise.all([
    fetchCsv("./assets/model_comparison.csv"),
    fetchCsv("./assets/prompt_comparison.csv"),
    fetchCsv("./assets/rag_comparison.csv")
  ]);
  document.getElementById("evaluationPage").innerHTML = `
    <div class="grid-4">
      ${metricCard("Best Model", models[0]?.model || "N/A", "Highest benchmark overall score")}
      ${metricCard("Best Prompt", prompts[0]?.prompt_style || "N/A", "Highest benchmark overall score")}
      ${metricCard("RAG On", Number(rag.find((row) => String(row.rag_enabled).toLowerCase() === "true")?.overall_score || 0).toFixed(2), "Overall score with retrieval")}
      ${metricCard("RAG Off", Number(rag.find((row) => String(row.rag_enabled).toLowerCase() === "false")?.overall_score || 0).toFixed(2), "Overall score without retrieval")}
    </div>
    <div class="grid-2" style="margin-top:18px;">
      <div class="card">
        <div class="section-title">What the benchmark columns mean</div>
        <ul>
          <li><strong>keyword_score</strong>: expected business keyword coverage</li>
          <li><strong>recommendation_score</strong>: presence and actionability of recommendations</li>
          <li><strong>completeness_score</strong>: completeness of the required JSON schema</li>
          <li><strong>groundedness_score</strong>: overlap with retrieved evidence</li>
          <li><strong>overall_score</strong>: average of the four core metrics</li>
        </ul>
      </div>
      <div class="card">
        <div class="section-title">Benchmark note</div>
        <p>The evaluation page mirrors the Streamlit benchmark dashboard using exported CSV artifacts so it stays static and GitHub Pages-compatible.</p>
      </div>
    </div>
    <div class="grid-2" style="margin-top:18px;">
      <div class="card"><div class="section-title">Overall score by model</div>${renderBarList(models.map((row) => ({ key: row.model, value: Number(row.overall_score) })))}</div>
      <div class="card"><div class="section-title">Overall score by prompt</div>${renderBarList(prompts.map((row) => ({ key: row.prompt_style, value: Number(row.overall_score) })))}</div>
    </div>
    <div class="grid-2" style="margin-top:18px;">
      <div class="card"><div class="section-title">Model table</div>${tableFromRows(models)}</div>
      <div class="card"><div class="section-title">Prompt table</div>${tableFromRows(prompts)}</div>
    </div>
  `;
}

async function renderPresentationPage(rows) {
  const summary = await getSummary(rows);
  renderUploadCard(summary);
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
    slides = [
      {
        title: "E-Commerce Customer Behavior Insights",
        speaker_notes: "Static preview based on the active dataset.",
        bullets: [`Dataset rows: ${summary.rows}`, `Detected outcome rate: ${summary.target_rate !== null && summary.target_rate !== undefined ? `${(summary.target_rate * 100).toFixed(1)}%` : "N/A"}`, `Top customer segment: ${summary.top_customer}`],
        chart_url: "./assets/time_performance.png"
      },
      {
        title: "Target Customers",
        speaker_notes: "Customer and channel groupings inferred from the dataset.",
        bullets: [`Top customer grouping: ${summary.top_customer}`, `Top channel grouping: ${summary.top_channel}`, "Use these segments to guide campaign prioritization and messaging."],
        chart_url: "./assets/customer_performance.png"
      }
    ];
  }

  document.getElementById("presentationPage").innerHTML = `
    <div class="card">
      <div class="section-title">Presentation preview</div>
      <div class="section-subtitle">${hasBackend() ? "When the backend is connected, this page requests a newly generated presentation from the Python presentation agent." : "Backend not configured, so this page shows the latest exported static presentation artifact."}</div>
      <div class="toolbar"><a class="button-link" href="${downloadUrl}">Download PowerPoint</a></div>
    </div>
    <div class="slide-grid" style="margin-top:18px;">
      ${slides.map((slide) => `
        <div class="preview-card slide">
          <div>
            <h3>${slide.title}</h3>
            <p class="small">${slide.speaker_notes}</p>
            <ul>${slide.bullets.map((bullet) => `<li>${bullet}</li>`).join("")}</ul>
          </div>
          <div class="preview-chart">${slide.chart_url ? `<img src="${slide.chart_url}" alt="${slide.title}" />` : "<p class='muted'>No chart attached to this slide.</p>"}</div>
        </div>
      `).join("")}
    </div>
  `;
}

async function init() {
  const page = document.body.dataset.page;
  renderTopbar(page);
  const rows = await getDataset();
  if (page === "index.html") await renderOverviewPage(rows);
  else if (page === "analyst.html") await renderAnalystPage(rows);
  else if (page === "evaluation.html") await renderEvaluationPage(rows);
  else if (page === "presentation.html") await renderPresentationPage(rows);
}

document.addEventListener("DOMContentLoaded", init);
