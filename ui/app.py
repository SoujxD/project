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

        [data-testid="stSidebarNav"] {
            display: none;
        }

        .app-shell {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .topbar {
            background: rgba(255,255,255,0.78);
            backdrop-filter: blur(8px);
            border: 1px solid var(--border);
            border-radius: 24px;
            box-shadow: var(--shadow);
            padding: 1.25rem 1.35rem;
        }

        .topbar-grid {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .eyebrow {
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            font-weight: 700;
            text-transform: uppercase;
            color: #4c8598;
        }

        .hero-title {
            font-size: 2.05rem;
            line-height: 1.06;
            font-weight: 750;
            color: #173746;
            margin-top: 0.3rem;
            max-width: 780px;
        }

        .hero-subtitle {
            margin-top: 0.55rem;
            color: var(--muted);
            max-width: 780px;
            line-height: 1.5;
        }

        .pill-row {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            align-items: center;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: var(--accent-soft);
            color: var(--accent);
            border: 1px solid rgba(29, 102, 125, 0.08);
            padding: 0.45rem 0.78rem;
            border-radius: 999px;
            font-size: 0.86rem;
            font-weight: 600;
            white-space: nowrap;
        }

        .upload-card {
            background: rgba(255,255,255,0.88);
            border: 1px solid var(--border);
            border-radius: 22px;
            box-shadow: var(--shadow);
            padding: 1.1rem 1.15rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }

        .section-head {
            margin-top: 0.2rem;
            margin-bottom: 0.85rem;
        }

        .section-title {
            font-size: 1.15rem;
            font-weight: 750;
            color: #173746;
            margin-bottom: 0.22rem;
        }

        .section-subtitle {
            color: var(--muted);
            line-height: 1.5;
        }

        .card, .metric-card, .info-card, .preview-card, .hero-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 22px;
            box-shadow: var(--shadow);
        }

        .card, .info-card, .preview-card {
            padding: 1.1rem 1.15rem;
        }

        .hero-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(241,247,250,0.95));
            padding: 1.25rem 1.25rem;
        }

        .metric-card {
            padding: 1rem 1.05rem;
            height: 100%;
        }

        .metric-label {
            color: #6d8895;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .metric-value {
            margin-top: 0.22rem;
            color: var(--accent);
            font-size: 1.65rem;
            line-height: 1.1;
            font-weight: 760;
        }

        .metric-caption {
            margin-top: 0.28rem;
            color: var(--muted);
            font-size: 0.92rem;
        }

        .label-text {
            color: #69818c;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }

        .muted {
            color: var(--muted);
        }

        .list-tight ul {
            margin-top: 0.35rem;
            margin-bottom: 0rem;
            padding-left: 1.15rem;
        }

        .list-tight li {
            margin-bottom: 0.38rem;
        }

        .divider-space {
            height: 0.35rem;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 999px;
            border: 1px solid rgba(29, 102, 125, 0.12);
            padding: 0.6rem 1rem;
            font-weight: 600;
        }

        .stTextArea textarea,
        .stTextInput input,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] > div {
            border-radius: 14px !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.55rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.75);
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 0.58rem 0.95rem;
            color: #395564;
            height: auto;
        }

        .stTabs [aria-selected="true"] {
            background: var(--accent-soft) !important;
            color: #184f62 !important;
        }

        .stAlert {
            border-radius: 16px;
        }

        .small-caption {
            color: var(--muted);
            font-size: 0.88rem;
        }

        .nav-note {
            margin-top: 0.6rem;
            color: var(--muted);
            font-size: 0.92rem;
        }

        .slide-title {
            font-size: 1.05rem;
            font-weight: 730;
            color: #173746;
            margin-bottom: 0.25rem;
        }

        .notes-box {
            margin-top: 0.55rem;
            padding: 0.8rem 0.9rem;
            border-radius: 16px;
            background: rgba(245, 249, 252, 0.95);
            border: 1px solid rgba(29, 102, 125, 0.08);
            color: #526a76;
            font-size: 0.93rem;
            line-height: 1.45;
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
def render_topbar(dataset_name: str) -> None:
    st.markdown(
        f"""
        <div class="topbar">
            <div class="topbar-grid">
                <div>
                    <div class="eyebrow">Ecommerce Multi-Agent Analytics</div>
                    <div class="hero-title">
                        A cleaner multi-page analytics workflow in Streamlit form.
                    </div>
                    <div class="hero-subtitle">
                        Upload an ecommerce CSV, inspect derived insights, review benchmark evaluation results,
                        and generate a presentation from the same active dataset without leaving the app.
                    </div>
                </div>
                <div class="pill-row">
                    <span class="pill">Overview</span>
                    <span class="pill">Analyst Demo</span>
                    <span class="pill">Evaluation</span>
                    <span class="pill">Presentation</span>
                    <span class="pill">{dataset_name}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_card(dataset: pd.DataFrame, dataset_name: str, presenter: PresentationGeneratorAgent) -> None:
    schema = presenter.infer_schema(dataset)
    st.markdown(
        f"""
        <div class="upload-card">
            <div class="section-title">Active dataset</div>
            <div class="section-subtitle">
                Upload a customer, session, or ecommerce performance CSV from the sidebar. The same dataset is reused
                across overview, analyst, and presentation workflows.
            </div>
            <div class="pill-row">
                <span class="pill">Dataset: {dataset_name}</span>
                <span class="pill">Rows: {len(dataset)}</span>
                <span class="pill">Columns: {len(dataset.columns)}</span>
                <span class="pill">Outcome: {schema['target'] or 'Not detected'}</span>
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


def info_card(title: str, body_md: str) -> None:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="section-title" style="font-size:1rem;">{title}</div>
            <div class="muted list-tight">{body_md}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    if horizontal:
        series.sort_values().plot(kind="barh", ax=ax)
        ax.set_ylabel("")
    else:
        series.plot(kind="bar", ax=ax)
        ax.set_xlabel("")
    ax.set_title(title)
    ax.grid(alpha=0.15, axis="x" if horizontal else "y")
    fig.tight_layout()
    st.pyplot(fig)


def plot_heatmap(df: pd.DataFrame, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    image = ax.imshow(df.values, cmap="GnBu", aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)

    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            ax.text(
                col,
                row,
                f"{df.iloc[row, col]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="#153847",
            )

    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig)


# -----------------------------
# Overview
# -----------------------------
def infer_overview_metrics(dataset: pd.DataFrame, presenter: PresentationGeneratorAgent) -> dict[str, str]:
    summary = presenter.summarize_dataset(st.session_state["active_dataset_path"])
    rate = f"{summary['target_rate']:.1%}" if summary["target_rate"] is not None else "N/A"

    return {
        "rows": str(summary["rows"]),
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
        "This page mirrors the public overview page and keeps the rest of the workflow grounded in the same active dataset.",
    )

    metric_cols = st.columns(4)
    with metric_cols[0]:
        metric_card("Rows", metrics["rows"], "Total records in the dataset")
    with metric_cols[1]:
        metric_card("Columns", metrics["columns"], "Available fields for analysis")
    with metric_cols[2]:
        metric_card("Outcome Rate", metrics["outcome"], "Detected conversion or purchase rate")
    with metric_cols[3]:
        metric_card("Top Segment", metrics["top_segment"], "Best-performing customer grouping")

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown('<div class="hero-card">', unsafe_allow_html=True)
        st.markdown("#### Current dataset context")
        st.markdown(f"**Active dataset:** `{dataset_name}`")
        st.markdown(f"**Detected outcome column:** `{schema['target'] or 'not clearly detected'}`")
        st.markdown(f"**Detected customer grouping:** `{schema['customer'] or 'not clearly detected'}`")
        st.markdown(f"**Detected channel grouping:** `{schema['channel'] or 'not clearly detected'}`")
        st.markdown(f"**Detected time grouping:** `{schema['time'] or 'not clearly detected'}`")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="hero-card">', unsafe_allow_html=True)
        st.markdown("#### Workflow")
        st.markdown(
            """
            - Upload one CSV in the sidebar  
            - Explore it through the analyst agent  
            - Review benchmark evaluation outputs  
            - Generate a polished presentation from the same data  
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="divider-space"></div>', unsafe_allow_html=True)
    st.markdown("#### Data preview")
    st.dataframe(dataset.head(20), use_container_width=True)


# -----------------------------
# Analyst Demo
# -----------------------------
def render_analyst_demo(agent: AnalystRAGAgent) -> None:
    section_header(
        "Analyst Agent Demo",
        "This page mirrors the public analyst page, but here the answer is produced by the Python analyst agent rather than browser-only heuristics.",
    )

    controls = st.columns([1, 1, 1, 1.15])
    with controls[0]:
        model = st.selectbox("Model", agent.available_models(), key="demo_model")
    with controls[1]:
        prompt_style = st.selectbox("Prompt style", agent.available_prompt_styles(), key="demo_prompt")
    with controls[2]:
        top_k = st.slider("Retrieved rows", min_value=1, max_value=8, value=5)
    with controls[3]:
        rag_enabled = st.toggle("Use retrieved context", value=True, key="demo_rag")

    st.markdown("##### Sample business prompts")
    sample_cols = st.columns(len(SAMPLE_QUESTIONS))
    for idx, sample in enumerate(SAMPLE_QUESTIONS):
        if sample_cols[idx].button(f"Sample {idx + 1}", key=f"sample_q_{idx}", use_container_width=True):
            st.session_state["question_text"] = sample

    st.text_area("Business question", key="question_text", height=120)

    run_col, note_col = st.columns([0.28, 0.72])
    with run_col:
        run_clicked = st.button("Run analyst agent", type="primary", use_container_width=True)
    with note_col:
        st.markdown(
            '<div class="small-caption">The answer is generated against the currently active dataset in the sidebar.</div>',
            unsafe_allow_html=True,
        )

    if run_clicked:
        with st.spinner("Generating answer from the active dataset..."):
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
        st.info("Run a sample question or write your own to preview the analyst output.")
        return

    response_col, json_col = st.columns([1.1, 0.9])
    with response_col:
        st.markdown("#### Executive briefing")
        st.text_area(
            "Structured response",
            value=executive_response_text(result.parsed_response),
            height=330,
            label_visibility="collapsed",
        )

    with json_col:
        st.markdown("#### Parsed JSON")
        st.json(result.parsed_response)

    st.markdown("#### Retrieved evidence")
    st.text(result.retrieved_context or "RAG was disabled for this run.")


# -----------------------------
# Evaluation Dashboard
# -----------------------------
def render_evaluation_dashboard() -> None:
    section_header(
        "Evaluation Dashboard",
        "This page mirrors the public evaluation page while keeping the benchmark tied to the bundled academic dataset for consistent grading and comparison.",
    )

    results_df = load_results()
    benchmark_agent = AnalystRAGAgent(dataset_path=DEFAULT_DATASET_PATH)
    pipeline = EvaluationPipeline(
        agent=benchmark_agent,
        questions_path=QUESTIONS_PATH,
        output_dir=OUTPUT_DIR,
    )

    top_controls = st.columns([0.9, 2.1, 1.05])
    with top_controls[0]:
        limit = st.number_input("Benchmark size", min_value=10, max_value=100, value=20, step=10)
    with top_controls[1]:
        st.markdown(
            """
            <div class="info-card">
                <div class="section-title" style="font-size:1rem;">Benchmark note</div>
                <div class="muted">
                    This panel uses the bundled benchmark dataset and question set for consistent grading and comparison.
                    The upload-driven analyst and presentation tabs remain flexible, while evaluation stays fixed.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_controls[2]:
        run_eval = st.button("Run benchmark evaluation", type="primary", use_container_width=True)

    if run_eval:
        with st.spinner("Running benchmark evaluation across models, prompts, and RAG settings..."):
            results_df = pipeline.run(limit=int(limit))
        st.success(f"Saved {len(results_df)} benchmark rows.")

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
        metric_card("Runs", str(len(results_df)), "Question-model-prompt-RAG combinations")
    with metric_cols[1]:
        metric_card("Best Model", str(model_table.index[0]), "Highest average overall score")
    with metric_cols[2]:
        metric_card("Best Prompt", str(prompt_table.index[0]), "Highest average overall score")
    with metric_cols[3]:
        metric_card("RAG Lift", rag_lift, "Average change in overall score")

    explain_left, explain_right = st.columns(2)
    with explain_left:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("#### What the scores represent")
        st.markdown(
            """
            - `keyword_score`: expected business keyword coverage  
            - `recommendation_score`: presence and usefulness of recommendations  
            - `completeness_score`: completeness of the JSON structure  
            - `groundedness_score`: overlap with retrieved evidence  
            - `overall_score`: simple average of the four metrics  
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with explain_right:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("#### How to read the visuals")
        st.markdown(
            """
            - Bar charts rank models and prompt styles by overall score  
            - Heatmaps show score composition instead of only the average  
            - Tables help document evaluation rigor in the report  
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    viz_cols = st.columns(2)
    with viz_cols[0]:
        plot_bar(model_table["overall_score"], "Overall score by model", horizontal=True)
    with viz_cols[1]:
        plot_bar(prompt_table["overall_score"], "Overall score by prompt", horizontal=True)

    heat_cols = st.columns(2)
    with heat_cols[0]:
        plot_heatmap(
            model_table[["keyword_score", "recommendation_score", "completeness_score", "groundedness_score"]],
            "Metric mix by model",
        )
    with heat_cols[1]:
        plot_heatmap(
            prompt_table[["keyword_score", "recommendation_score", "completeness_score", "groundedness_score"]],
            "Metric mix by prompt",
        )

    with st.expander("Aggregated benchmark tables", expanded=False):
        st.markdown("**By model**")
        st.dataframe(model_table, use_container_width=True)
        st.markdown("**By prompt**")
        st.dataframe(prompt_table, use_container_width=True)
        st.markdown("**RAG vs no-RAG**")
        st.dataframe(rag_table, use_container_width=True)

    display_columns = [
        "question_id",
        "category",
        "model",
        "prompt_style",
        "rag_enabled",
        "keyword_score",
        "recommendation_score",
        "completeness_score",
        "groundedness_score",
        "overall_score",
        "summary",
    ]
    st.markdown("#### Sample benchmark rows")
    st.dataframe(results_df[display_columns].head(40), use_container_width=True)


# -----------------------------
# Presentation Generator
# -----------------------------
def render_presentation_generator(dataset_path: str) -> None:
    section_header(
        "Presentation Generator",
        "This page mirrors the public presentation page while generating the full PowerPoint directly from the active dataset.",
    )
    presenter = build_presenter()

    action_left, action_right = st.columns([0.3, 0.7])
    with action_left:
        generate_clicked = st.button("Generate presentation", type="primary", use_container_width=True)
    with action_right:
        st.markdown(
            '<div class="small-caption">The slide deck is created from the same active dataset used in the overview and analyst tabs.</div>',
            unsafe_allow_html=True,
        )

    if generate_clicked:
        with st.spinner("Creating charts and presentation assets from the active dataset..."):
            ppt_path, slides, chart_paths = presenter.create_presentation(
                dataset_path=dataset_path,
                output_path=OUTPUT_DIR / "presentation.pptx",
            )
        st.session_state["ppt_path"] = ppt_path
        st.session_state["slides"] = slides
        st.session_state["chart_paths"] = chart_paths
        st.success("Presentation generated.")

    slides = st.session_state.get("slides", [])
    chart_paths = st.session_state.get("chart_paths", {})

    if slides:
        st.markdown("#### Slide preview")
        for idx, slide in enumerate(slides, start=1):
            preview_cols = st.columns([1.08, 0.92])

            with preview_cols[0]:
                st.markdown(
                    f"""
                    <div class="preview-card">
                        <div class="label-text">Slide {idx}</div>
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
                    st.image(str(chart_paths[slide.chart_key]), width=430)
                else:
                    st.markdown(
                        """
                        <div class="info-card">
                            <div class="muted">No chart attached to this slide.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("")

    ppt_path = st.session_state.get("ppt_path")
    if ppt_path and Path(ppt_path).exists():
        with open(ppt_path, "rb") as file_obj:
            st.download_button(
                label="Download PowerPoint",
                data=file_obj.read(),
                file_name="presentation.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                use_container_width=False,
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
        st.caption(dataset_name)
        st.markdown(f"**Rows:** {len(dataset)}")
        st.markdown(f"**Columns:** {len(dataset.columns)}")

        if st.button("Reset to bundled dataset", use_container_width=True):
            st.session_state["active_dataset_path"] = str(DEFAULT_DATASET_PATH)
            st.session_state["active_dataset_name"] = DEFAULT_DATASET_PATH.name
            st.session_state["analyst_result"] = None
            st.session_state["ppt_path"] = None
            st.session_state["slides"] = []
            st.session_state["chart_paths"] = {}
            rerun_app()


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
    presenter = build_presenter()

    st.markdown('<div class="app-shell">', unsafe_allow_html=True)
    render_topbar(dataset_name)
    render_upload_card(dataset, dataset_name, presenter)

    page = st.radio(
        "Navigate",
        ["Overview", "Analyst Demo", "Evaluation", "Presentation"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="nav-note">Use the navigation above to move between the same four sections shown on the public HTML site.</div>',
        unsafe_allow_html=True,
    )

    if page == "Overview":
        render_overview(dataset, dataset_name)
    elif page == "Analyst Demo":
        render_analyst_demo(agent)
    elif page == "Evaluation":
        render_evaluation_dashboard()
    else:
        render_presentation_generator(dataset_path)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
