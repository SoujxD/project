from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agents.analyst_agent import AnalystRAGAgent
from agents.presentation_agent import PresentationGeneratorAgent
from evaluation.evaluator import EvaluationPipeline


DEFAULT_DATASET_PATH = BASE_DIR / "data" / "dataset.csv"
QUESTIONS_PATH = BASE_DIR / "data" / "evaluation_questions.json"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_QUESTIONS = [
    "Which customer segments should we prioritize based on the uploaded dataset?",
    "What patterns distinguish stronger-performing sessions or customers?",
    "Which channels appear most promising for future budget allocation?",
    "What recommendations would improve performance based on this data?",
]

SCORE_COLORS = {
    "high": "#1d7a61",
    "mid": "#b07d2e",
    "low": "#a03030",
}


# -----------------------------
# Page Styling
# -----------------------------
def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f5f8fb;
            --card: rgba(255,255,255,0.88);
            --card-strong: rgba(255,255,255,0.96);
            --border: rgba(21, 76, 97, 0.08);
            --shadow: 0 12px 30px rgba(17, 52, 68, 0.07);
            --text: #163747;
            --muted: #617a86;
            --accent: #1d667d;
            --accent-soft: rgba(29, 102, 125, 0.10);
            --success: #1d7a61;
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(130, 197, 219, 0.14), transparent 24%),
                linear-gradient(180deg, #fbfdff 0%, #f3f7fa 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1240px;
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
        }

        [data-testid="stSidebar"] {
            background: #f7fafc;
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebarNav"] { display: none; }

        /* ── Topbar ── */
        .topbar {
            background: rgba(255,255,255,0.82);
            backdrop-filter: blur(10px);
            border: 1px solid var(--border);
            border-radius: 22px;
            box-shadow: var(--shadow);
            padding: 1.35rem 1.5rem;
            margin-bottom: 0.75rem;
        }

        .topbar-grid {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .eyebrow {
            font-size: 0.7rem;
            letter-spacing: 0.15em;
            font-weight: 700;
            text-transform: uppercase;
            color: #4c8598;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .eyebrow::before {
            content: "";
            display: inline-block;
            width: 16px;
            height: 2px;
            background: #4c8598;
            border-radius: 2px;
        }

        .hero-title {
            font-size: 1.9rem;
            line-height: 1.06;
            font-weight: 750;
            background: linear-gradient(135deg, #163747, #1d7a9a);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-top: 0.3rem;
            max-width: 780px;
        }

        .hero-subtitle {
            margin-top: 0.5rem;
            color: var(--muted);
            max-width: 780px;
            line-height: 1.55;
            font-size: 0.93rem;
        }

        .pill-row {
            display: flex;
            gap: 0.45rem;
            flex-wrap: wrap;
            align-items: center;
            margin-top: 0.25rem;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            background: var(--accent-soft);
            color: var(--accent);
            border: 1px solid rgba(29, 102, 125, 0.1);
            padding: 0.42rem 0.75rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
            white-space: nowrap;
        }

        .pill.active {
            background: var(--accent);
            color: white;
            border-color: transparent;
        }

        /* ── Dataset status bar ── */
        .dataset-bar {
            background: rgba(255,255,255,0.75);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 0.75rem 1.1rem;
            margin-bottom: 0.9rem;
            display: flex;
            align-items: center;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .dataset-bar-label {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--muted);
        }

        /* ── Section headers ── */
        .section-head {
            margin-top: 0.2rem;
            margin-bottom: 1rem;
            padding-left: 0.5rem;
            border-left: 3px solid var(--accent);
        }

        .section-title {
            font-size: 1.12rem;
            font-weight: 750;
            color: #173746;
            margin-bottom: 0.2rem;
        }

        .section-subtitle {
            color: var(--muted);
            line-height: 1.5;
            font-size: 0.9rem;
        }

        /* ── Cards ── */
        .card, .metric-card, .info-card, .preview-card, .hero-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 20px;
            box-shadow: var(--shadow);
        }

        .card, .info-card, .preview-card { padding: 1.1rem 1.15rem; }

        .hero-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(241,247,250,0.95));
            padding: 1.2rem 1.25rem;
        }

        .metric-card {
            padding: 1rem 1.1rem;
            height: 100%;
            border-top: 3px solid var(--accent);
        }

        .metric-label {
            color: #6d8895;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }

        .metric-value {
            margin-top: 0.2rem;
            color: var(--accent);
            font-size: 1.6rem;
            line-height: 1.1;
            font-weight: 760;
        }

        .metric-caption {
            margin-top: 0.25rem;
            color: var(--muted);
            font-size: 0.88rem;
        }

        /* ── Score badge ── */
        .score-badge {
            display: inline-block;
            padding: 3px 9px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
        }

        .score-high { background: rgba(29, 122, 97, 0.12); color: #1d7a61; }
        .score-mid  { background: rgba(176, 125, 46, 0.12); color: #b07d2e; }
        .score-low  { background: rgba(160, 48, 48, 0.12);  color: #a03030; }

        /* ── Info card ── */
        .info-card {
            background: rgba(241, 248, 251, 0.9);
            border-left: 3px solid rgba(29, 102, 125, 0.22);
        }

        .label-text {
            color: #69818c;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.22rem;
        }

        .muted { color: var(--muted); }

        .list-tight ul { margin-top: 0.35rem; margin-bottom: 0; padding-left: 1.15rem; }
        .list-tight li { margin-bottom: 0.38rem; }

        .divider-space { height: 0.5rem; }

        /* ── Buttons ── */
        .stButton > button,
        .stDownloadButton > button {
            border-radius: 999px;
            border: 1px solid rgba(29, 102, 125, 0.14);
            padding: 0.62rem 1.1rem;
            font-weight: 600;
            transition: all 0.18s;
        }

        .stButton > button[kind="primary"],
        .stDownloadButton > button {
            background: var(--accent) !important;
            color: white !important;
        }

        /* ── Inputs ── */
        .stTextArea textarea,
        .stTextInput input,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] > div {
            border-radius: 14px !important;
        }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.45rem;
            background: transparent;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0;
        }

        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border: 1px solid transparent;
            border-bottom: none;
            border-radius: 12px 12px 0 0;
            padding: 0.55rem 1.05rem;
            color: var(--muted);
            height: auto;
            font-weight: 600;
            font-size: 0.88rem;
            transition: background 0.15s, color 0.15s;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(29, 102, 125, 0.06);
            color: var(--accent);
        }

        .stTabs [aria-selected="true"] {
            background: rgba(255,255,255,0.9) !important;
            color: var(--accent) !important;
            border-color: var(--border) !important;
        }

        /* ── Expander ── */
        .stExpander {
            border: 1px solid var(--border) !important;
            border-radius: 16px !important;
        }

        /* ── Alerts ── */
        .stAlert { border-radius: 14px; }

        /* ── Misc ── */
        .small-caption { color: var(--muted); font-size: 0.86rem; line-height: 1.5; }
        .nav-note { margin-top: 0.5rem; color: var(--muted); font-size: 0.88rem; }

        .slide-title {
            font-size: 1.05rem;
            font-weight: 730;
            color: #173746;
            margin-bottom: 0.22rem;
        }

        .notes-box {
            margin-top: 0.55rem;
            padding: 0.85rem 0.95rem;
            border-radius: 14px;
            background: rgba(245, 249, 252, 0.95);
            border: 1px solid rgba(29, 102, 125, 0.09);
            color: #526a76;
            font-size: 0.9rem;
            line-height: 1.5;
        }

        .slide-number {
            display: inline-block;
            background: var(--accent-soft);
            color: var(--accent);
            border-radius: 999px;
            padding: 2px 10px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: var(--accent);
        }

        .sidebar-stat {
            display: flex;
            justify-content: space-between;
            font-size: 0.88rem;
            padding: 5px 0;
            border-bottom: 1px solid var(--border);
        }

        .sidebar-stat span:last-child {
            font-weight: 700;
            color: var(--accent);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Cached Loaders
# -----------------------------
@st.cache_data(show_spinner=False)
def load_dataset(dataset_path: str) -> pd.DataFrame:
    return pd.read_csv(dataset_path)


@st.cache_resource(show_spinner=False)
def build_agent(dataset_path: str) -> AnalystRAGAgent:
    return AnalystRAGAgent(dataset_path=dataset_path)


@st.cache_resource(show_spinner=False)
def build_presenter() -> PresentationGeneratorAgent:
    return PresentationGeneratorAgent(output_dir=OUTPUT_DIR)


def load_results() -> pd.DataFrame | None:
    results_path = OUTPUT_DIR / "results.csv"
    if results_path.exists():
        return pd.read_csv(results_path)
    return None


# -----------------------------
# Session + Dataset Handling
# -----------------------------
def ensure_session_defaults() -> None:
    defaults: dict[str, Any] = {
        "active_dataset_path": str(DEFAULT_DATASET_PATH),
        "active_dataset_name": DEFAULT_DATASET_PATH.name,
        "question_text": SAMPLE_QUESTIONS[0],
        "analyst_result": None,
        "ppt_path": None,
        "slides": [],
        "chart_paths": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def save_uploaded_dataset() -> None:
    uploaded_file = st.sidebar.file_uploader("Upload ecommerce CSV", type=["csv"])
    if uploaded_file is None:
        return

    saved_path = UPLOAD_DIR / uploaded_file.name
    saved_path.write_bytes(uploaded_file.getbuffer())

    st.session_state["active_dataset_path"] = str(saved_path)
    st.session_state["active_dataset_name"] = uploaded_file.name
    st.session_state["analyst_result"] = None
    st.session_state["ppt_path"] = None
    st.session_state["slides"] = []
    st.session_state["chart_paths"] = {}


def get_active_dataset() -> tuple[pd.DataFrame, str, str, AnalystRAGAgent]:
    dataset_path = st.session_state["active_dataset_path"]
    dataset_name = st.session_state["active_dataset_name"]
    dataset = load_dataset(dataset_path)
    agent = build_agent(dataset_path)
    return dataset, dataset_name, dataset_path, agent


# -----------------------------
# UI Helpers
# -----------------------------
def render_topbar(dataset_name: str, dataset: pd.DataFrame) -> None:
    is_custom = dataset_name != DEFAULT_DATASET_PATH.name
    dataset_pill = (
        f'<span class="pill active">{dataset_name}</span>'
        if is_custom
        else f'<span class="pill">{dataset_name}</span>'
    )
    st.markdown(
        f"""
        <div class="topbar">
            <div class="topbar-grid">
                <div>
                    <div class="eyebrow">Ecommerce Multi-Agent Analytics</div>
                    <div class="hero-title">Analytics workflow: upload, query, evaluate, present.</div>
                    <div class="hero-subtitle">
                        Upload an ecommerce CSV, ask questions through the analyst agent, run benchmark
                        evaluations, and generate a stakeholder-ready presentation — all from one dataset.
                    </div>
                </div>
                <div>
                    <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:var(--muted);margin-bottom:0.45rem;">Active dataset</div>
                    <div class="pill-row">
                        {dataset_pill}
                        <span class="pill">{len(dataset):,} rows</span>
                        <span class="pill">{len(dataset.columns)} cols</span>
                    </div>
                    <div class="pill-row" style="margin-top:0.4rem;">
                        <span class="pill">Overview</span>
                        <span class="pill">Analyst</span>
                        <span class="pill">Evaluation</span>
                        <span class="pill">Presentation</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-head">
            <div class="section-title">{title}</div>
            <div class="section-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_class(val: float) -> str:
    if val >= 0.75:
        return "score-high"
    if val >= 0.5:
        return "score-mid"
    return "score-low"


def executive_response_text(parsed: dict[str, Any]) -> str:
    insights = "\n".join(f"- {item}" for item in parsed.get("key_insights", [])) or "- No key insights returned."
    patterns = "\n".join(f"- {item}" for item in parsed.get("patterns", [])) or "- No patterns returned."
    recommendations = "\n".join(f"- {item}" for item in parsed.get("recommendations", [])) or "- No recommendations returned."

    return (
        f"Summary\n{parsed.get('summary', 'No summary available.')}\n\n"
        f"Key Insights\n{insights}\n\n"
        f"Patterns\n{patterns}\n\n"
        f"Recommendations\n{recommendations}\n\n"
        f"Confidence\n{parsed.get('confidence', 'low').title()}"
    )


# -----------------------------
# Charts
# -----------------------------
def plot_bar(series: pd.Series, title: str, horizontal: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    color = "#1d667d"
    if horizontal:
        series.sort_values().plot(kind="barh", ax=ax, color=color)
        ax.set_ylabel("")
    else:
        series.plot(kind="bar", ax=ax, color=color)
        ax.set_xlabel("")
    ax.set_title(title, fontsize=12, pad=10, color="#163747")
    ax.grid(alpha=0.12, axis="x" if horizontal else "y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor("#f7fafc")
    ax.set_facecolor("#f7fafc")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    image = ax.imshow(df.values, cmap="GnBu", aspect="auto", vmin=0, vmax=1)
    ax.set_title(title, fontsize=12, pad=10, color="#163747")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=9)

    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            val = df.iloc[row, col]
            text_color = "white" if val > 0.65 else "#153847"
            ax.text(col, row, f"{val:.2f}", ha="center", va="center", fontsize=9, color=text_color)

    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    fig.patch.set_facecolor("#f7fafc")
    ax.set_facecolor("#f7fafc")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# -----------------------------
# Overview
# -----------------------------
def infer_overview_metrics(dataset: pd.DataFrame, presenter: PresentationGeneratorAgent) -> dict[str, str]:
    summary = presenter.summarize_dataset(st.session_state["active_dataset_path"])
    rate = f"{summary['target_rate']:.1%}" if summary["target_rate"] is not None else "N/A"

    return {
        "rows": f"{summary['rows']:,}",
        "columns": str(summary["columns"]),
        "outcome": rate,
        "top_segment": str(summary["top_customer"]),
        "top_channel": str(summary["top_channel"]),
        "top_time": str(summary["top_time"]),
    }


def render_overview(dataset: pd.DataFrame, dataset_name: str) -> None:
    presenter = build_presenter()
    metrics = infer_overview_metrics(dataset, presenter)
    schema = presenter.infer_schema(dataset)

    section_header(
        "Dataset Overview",
        "A high-level summary of the active dataset — the same data drives the analyst, evaluation, and presentation tabs.",
    )

    metric_cols = st.columns(4)
    with metric_cols[0]:
        metric_card("Rows", metrics["rows"], "Total records")
    with metric_cols[1]:
        metric_card("Columns", metrics["columns"], "Available fields")
    with metric_cols[2]:
        metric_card("Outcome Rate", metrics["outcome"], "Detected conversion rate")
    with metric_cols[3]:
        metric_card("Top Segment", metrics["top_segment"], "Best-performing group")

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown('<div class="hero-card">', unsafe_allow_html=True)
        st.markdown("#### Detected schema")

        def schema_row(label: str, value: str | None) -> None:
            display = f"`{value}`" if value else "_not detected_"
            st.markdown(f"**{label}:** {display}")

        schema_row("Dataset", dataset_name)
        schema_row("Outcome column", schema["target"])
        schema_row("Customer grouping", schema["customer"])
        schema_row("Channel grouping", schema["channel"])
        schema_row("Time grouping", schema["time"])
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="hero-card">', unsafe_allow_html=True)
        st.markdown("#### Workflow")
        st.markdown(
            """
            1. Upload a CSV in the sidebar (or use the bundled dataset)
            2. Run analyst questions in the **Analyst** tab
            3. Benchmark models & prompts in the **Evaluation** tab
            4. Export a polished deck from the **Presentation** tab
            """
        )
        st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="small-caption">Top channel: <strong>{metrics["top_channel"]}</strong> &nbsp;·&nbsp; '
            f'Top period: <strong>{metrics["top_time"]}</strong></div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    st.markdown("#### Data preview")
    st.dataframe(dataset.head(20), use_container_width=True, hide_index=False)


# -----------------------------
# Analyst Demo
# -----------------------------
def render_analyst_demo(agent: AnalystRAGAgent) -> None:
    section_header(
        "Analyst Agent",
        "Ask a business question and receive a structured, evidence-backed answer from the active dataset.",
    )

    ctrl_cols = st.columns([1, 1, 1, 1.15])
    with ctrl_cols[0]:
        model = st.selectbox("Model", agent.available_models(), key="demo_model")
    with ctrl_cols[1]:
        prompt_style = st.selectbox("Prompt style", agent.available_prompt_styles(), key="demo_prompt")
    with ctrl_cols[2]:
        top_k = st.slider("Retrieved rows", min_value=1, max_value=8, value=5)
    with ctrl_cols[3]:
        rag_enabled = st.toggle("Use retrieved context", value=True, key="demo_rag")

    st.markdown("##### Try a sample question")
    sample_cols = st.columns(len(SAMPLE_QUESTIONS))
    for idx, sample in enumerate(SAMPLE_QUESTIONS):
        if sample_cols[idx].button(f"Sample {idx + 1}", key=f"sample_q_{idx}", use_container_width=True):
            st.session_state["question_text"] = sample
            rerun_app()

    st.text_area("Business question", key="question_text", height=110, placeholder="e.g. Which visitor segments convert best and why?")

    run_col, note_col = st.columns([0.28, 0.72])
    with run_col:
        run_clicked = st.button("Run analyst agent", type="primary", use_container_width=True)
    with note_col:
        st.markdown(
            '<div class="small-caption" style="margin-top:0.6rem;">Runs against the active dataset shown in the sidebar.</div>',
            unsafe_allow_html=True,
        )

    if run_clicked:
        with st.spinner("Generating answer from the active dataset…"):
            result = agent.answer_question(
                question=st.session_state["question_text"],
                model=model,
                prompt_style=prompt_style,
                rag_enabled=rag_enabled,
                top_k=top_k,
            )
        st.session_state["analyst_result"] = result

    result = st.session_state.get("analyst_result")
    if not result:
        st.info("Select a sample question above or write your own, then click **Run analyst agent**.")
        return

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)

    response_col, json_col = st.columns([1.1, 0.9])
    parsed = result.parsed_response

    with response_col:
        st.markdown("#### Executive briefing")
        conf = parsed.get("confidence", "low")
        conf_class = "score-high" if conf == "high" else ("score-mid" if conf == "medium" else "score-low")
        st.markdown(
            f'Confidence: <span class="score-badge {conf_class}">{conf.title()}</span>',
            unsafe_allow_html=True,
        )
        st.text_area(
            "Structured response",
            value=executive_response_text(parsed),
            height=310,
            label_visibility="collapsed",
        )

    with json_col:
        st.markdown("#### Parsed JSON")
        st.json(parsed, expanded=True)

    with st.expander("Retrieved dataset evidence", expanded=False):
        if result.retrieved_context:
            st.text(result.retrieved_context)
        else:
            st.caption("RAG was disabled for this run.")


# -----------------------------
# Evaluation Dashboard
# -----------------------------
def render_evaluation_dashboard() -> None:
    section_header(
        "Evaluation Dashboard",
        "Benchmark all model and prompt combinations on the bundled academic dataset for consistent, reproducible comparisons.",
    )

    results_df = load_results()
    benchmark_agent = AnalystRAGAgent(dataset_path=DEFAULT_DATASET_PATH)
    pipeline = EvaluationPipeline(
        agent=benchmark_agent,
        questions_path=QUESTIONS_PATH,
        output_dir=OUTPUT_DIR,
    )

    ctrl_cols = st.columns([0.7, 2.3, 1.1])
    with ctrl_cols[0]:
        limit = st.number_input("Questions", min_value=10, max_value=100, value=20, step=10)
    with ctrl_cols[1]:
        st.markdown(
            """
            <div class="info-card">
                <div style="font-size:0.78rem;font-weight:700;color:#4c8598;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:4px;">Benchmark note</div>
                <div class="muted" style="font-size:0.88rem;">
                    Evaluation always runs on the bundled dataset and question set for reproducibility.
                    The analyst and presentation tabs remain flexible for uploaded data.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ctrl_cols[2]:
        run_eval = st.button("Run benchmark", type="primary", use_container_width=True)

    if run_eval:
        with st.spinner("Running evaluation across all models, prompts, and RAG settings…"):
            results_df = pipeline.run(limit=int(limit))
        st.success(f"Evaluation complete — {len(results_df)} rows saved.")

    if results_df is None:
        st.info("Run the benchmark once to populate this dashboard.")
        return

    aggregated = pipeline.aggregate_results(results_df)
    model_table = aggregated["model_comparison"]
    prompt_table = aggregated["prompt_comparison"]
    rag_table = aggregated["rag_comparison"]

    rag_lift = "N/A"
    if True in rag_table.index and False in rag_table.index:
        rag_lift = f"{rag_table.loc[True, 'overall_score'] - rag_table.loc[False, 'overall_score']:+.2f}"

    metric_cols = st.columns(4)
    with metric_cols[0]:
        metric_card("Benchmark Rows", str(len(results_df)), "Question-model-prompt-RAG combos")
    with metric_cols[1]:
        metric_card("Best Model", str(model_table.index[0]), "Highest avg overall score")
    with metric_cols[2]:
        metric_card("Best Prompt", str(prompt_table.index[0]), "Highest avg overall score")
    with metric_cols[3]:
        metric_card("RAG Lift", rag_lift, "Score delta: RAG on vs off")

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)

    explain_left, explain_right = st.columns(2)
    with explain_left:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("#### Score definitions")
        st.markdown(
            """
            - `keyword_score` — expected business keyword coverage
            - `recommendation_score` — presence and usefulness of recommendations
            - `completeness_score` — completeness of the JSON structure
            - `groundedness_score` — overlap with retrieved evidence
            - `overall_score` — simple average of the four metrics
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with explain_right:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("#### Reading the visuals")
        st.markdown(
            """
            - Horizontal bars rank models and prompt styles by overall score
            - Heatmaps show per-metric composition, not just the average
            - Color intensity: darker = higher score (0 → 1 scale)
            - Raw rows are available in the expandable table below
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)

    viz_cols = st.columns(2)
    with viz_cols[0]:
        plot_bar(model_table["overall_score"], "Overall score by model", horizontal=True)
    with viz_cols[1]:
        plot_bar(prompt_table["overall_score"], "Overall score by prompt style", horizontal=True)

    heat_cols = st.columns(2)
    with heat_cols[0]:
        plot_heatmap(
            model_table[["keyword_score", "recommendation_score", "completeness_score", "groundedness_score"]],
            "Metric mix by model",
        )
    with heat_cols[1]:
        plot_heatmap(
            prompt_table[["keyword_score", "recommendation_score", "completeness_score", "groundedness_score"]],
            "Metric mix by prompt style",
        )

    with st.expander("Aggregated benchmark tables", expanded=False):
        st.markdown("**By model**")
        st.dataframe(model_table, use_container_width=True)
        st.markdown("**By prompt style**")
        st.dataframe(prompt_table, use_container_width=True)
        st.markdown("**RAG vs no-RAG**")
        st.dataframe(rag_table, use_container_width=True)

    display_columns = [
        "question_id", "category", "model", "prompt_style", "rag_enabled",
        "keyword_score", "recommendation_score", "completeness_score",
        "groundedness_score", "overall_score", "summary",
    ]
    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    st.markdown("#### Sample benchmark rows")
    st.dataframe(results_df[display_columns].head(40), use_container_width=True)


# -----------------------------
# Presentation Generator
# -----------------------------
def render_presentation_generator(dataset_path: str) -> None:
    section_header(
        "Presentation Generator",
        "Generate a stakeholder-ready PowerPoint directly from the active dataset — charts included.",
    )
    presenter = build_presenter()

    action_left, action_right = st.columns([0.32, 0.68])
    with action_left:
        generate_clicked = st.button("Generate presentation", type="primary", use_container_width=True)
    with action_right:
        st.markdown(
            '<div class="small-caption" style="margin-top:0.65rem;">'
            'Creates charts and builds the deck from the active dataset. '
            'Previous outputs are overwritten.'
            '</div>',
            unsafe_allow_html=True,
        )

    if generate_clicked:
        with st.spinner("Building charts and assembling presentation…"):
            ppt_path, slides, chart_paths = presenter.create_presentation(
                dataset_path=dataset_path,
                output_path=OUTPUT_DIR / "presentation.pptx",
            )
        st.session_state["ppt_path"] = ppt_path
        st.session_state["slides"] = slides
        st.session_state["chart_paths"] = chart_paths
        st.success(f"Presentation ready — {len(slides)} slides generated.")

    slides = st.session_state.get("slides", [])
    chart_paths = st.session_state.get("chart_paths", {})

    if not slides:
        st.info("Click **Generate presentation** to build the deck from the active dataset.")
        return

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    st.markdown(f"#### Slide preview &nbsp; <span style='color:var(--muted);font-size:0.88rem;font-weight:400;'>{len(slides)} slides</span>", unsafe_allow_html=True)

    for idx, slide in enumerate(slides, start=1):
        preview_cols = st.columns([1.1, 0.9])

        with preview_cols[0]:
            st.markdown(
                f"""
                <div class="preview-card">
                    <div class="slide-number">Slide {idx} of {len(slides)}</div>
                    <div class="slide-title">{slide.title}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for bullet in slide.bullets:
                st.write(f"- {bullet}")

            st.markdown(
                f"""
                <div class="notes-box">
                    <strong>Speaker notes:</strong><br>{slide.speaker_notes}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with preview_cols[1]:
            if slide.chart_key and slide.chart_key in chart_paths:
                st.image(str(chart_paths[slide.chart_key]), use_column_width=True)
            else:
                st.markdown(
                    """
                    <div class="info-card" style="text-align:center;padding:2rem 1rem;">
                        <div class="muted" style="font-size:0.88rem;">No chart attached to this slide.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if idx < len(slides):
            st.markdown('<hr style="border:none;border-top:1px solid var(--border);margin:0.75rem 0;">', unsafe_allow_html=True)

    ppt_path = st.session_state.get("ppt_path")
    if ppt_path and Path(ppt_path).exists():
        st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
        with open(ppt_path, "rb") as file_obj:
            st.download_button(
                label="Download PowerPoint",
                data=file_obj.read(),
                file_name="presentation.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            )


# -----------------------------
# Sidebar
# -----------------------------
def render_sidebar(dataset: pd.DataFrame, dataset_name: str) -> None:
    with st.sidebar:
        st.markdown("## Dataset")
        save_uploaded_dataset()

        st.markdown("---")
        st.markdown("### Active file")

        is_custom = dataset_name != DEFAULT_DATASET_PATH.name
        label = f"**{dataset_name}**" + (" _(custom)_" if is_custom else " _(bundled)_")
        st.markdown(label)

        st.markdown(
            f"""
            <div class="sidebar-stat"><span>Rows</span><span>{len(dataset):,}</span></div>
            <div class="sidebar-stat"><span>Columns</span><span>{len(dataset.columns)}</span></div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")

        if st.button("Reset to bundled dataset", use_container_width=True):
            st.session_state["active_dataset_path"] = str(DEFAULT_DATASET_PATH)
            st.session_state["active_dataset_name"] = DEFAULT_DATASET_PATH.name
            st.session_state["analyst_result"] = None
            st.session_state["ppt_path"] = None
            st.session_state["slides"] = []
            st.session_state["chart_paths"] = {}
            rerun_app()

        st.markdown("---")
        st.markdown("### About")
        st.caption(
            "ISE547 · Multi-agent analytics · "
            "RAG-powered analyst, model evaluation, and presentation generation."
        )


# -----------------------------
# Main App
# -----------------------------
def main() -> None:
    st.set_page_config(
        page_title="Ecommerce Analytics Studio",
        page_icon="📊",
        layout="wide",
    )

    inject_styles()
    ensure_session_defaults()

    dataset, dataset_name, dataset_path, agent = get_active_dataset()
    render_sidebar(dataset, dataset_name)

    render_topbar(dataset_name, dataset)

    tab_overview, tab_analyst, tab_eval, tab_pres = st.tabs(
        ["Overview", "Analyst", "Evaluation", "Presentation"]
    )

    with tab_overview:
        render_overview(dataset, dataset_name)

    with tab_analyst:
        render_analyst_demo(agent)

    with tab_eval:
        render_evaluation_dashboard()

    with tab_pres:
        render_presentation_generator(dataset_path)


if __name__ == "__main__":
    main()
