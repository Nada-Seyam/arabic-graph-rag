"""
Streamlit chat GUI for the Arabic GraphRAG system.

This file only wires an interface on top of the EXISTING pipeline
(`src.rag.pipeline.GraphRAGPipeline`) — it does not reimplement retrieval,
routing, reranking, compression, or generation. Every metric shown in the
"Retrieval Details" panel comes directly from the `PipelineResult` object
already returned by `pipeline.answer()`, or from the JSON evaluation
reports that `scripts/run_evaluation.py` already produces under `reports/`.
Nothing here invents or hardcodes a metric.

Run from the project root (same folder as config.py and requirements.txt):
    streamlit run app.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from src.rag.pipeline import GraphRAGPipeline, PipelineResult  # noqa: E402

# --------------------------------------------------------------------------- #
# Page setup + minimal RTL styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="نظام الأسئلة والأجوبة - الوثائق الإجرائية",
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* RTL layout for Arabic content areas. Left as-is for widgets/labels
       that are inherently LTR (numbers, code, file names). */
    .stChatMessage, .stMarkdown, .stAlert, .stExpander, .stDataFrame {
        direction: rtl;
        text-align: right;
        font-family: "Segoe UI", Tahoma, Arial, sans-serif;
    }
    .stChatInputContainer textarea {
        direction: rtl;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Pipeline (built once per server process, not per interaction)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="جاري تحميل النظام (قد يستغرق ذلك بعض الوقت في أول مرة)...")
def get_pipeline() -> GraphRAGPipeline:
    return GraphRAGPipeline()


def format_citations(citations: List[Dict[str, Any]]) -> List[str]:
    """Mirrors scripts/run_query.py's citation formatting exactly, so the
    GUI shows sources the same way the CLI already does."""
    lines = []
    for c in citations:
        source = c.get("source", "unknown")
        page_start = c.get("page_start", "?")
        page_end = c.get("page_end", page_start)
        page_label = f"{page_start}" if page_start == page_end else f"{page_start}-{page_end}"
        lines.append(f"{source} (صفحة {page_label})")
    return lines


def render_retrieval_details(result: PipelineResult, wall_clock_seconds: float) -> None:
    """Renders only fields that actually exist on PipelineResult / its
    cost_summary — nothing here is invented or estimated."""
    col1, col2, col3 = st.columns(3)
    col1.metric("المسار (Route)", result.route)
    col2.metric("النموذج المستخدم", result.model_used)
    col3.metric("الزمن الكلي للاستعلام (ثانية)", f"{wall_clock_seconds:.1f}")

    st.markdown("**الخطوات المنفذة (techniques_executed):**")
    st.write(", ".join(result.techniques_executed) if result.techniques_executed else "—")

    col4, col5, col6 = st.columns(3)
    col4.metric("عدد المقاطع المستخدمة في الإجابة", len(result.retrieved_chunks))
    col5.metric("عدد عقد الرسم البياني (graph nodes)", len(result.graph_nodes))
    col6.metric("عدد الروابط (graph edges)", len(result.graph_edges))

    if result.crag_info:
        st.markdown("**معلومات CRAG (التصحيح التلقائي للاسترجاع):**")
        crag = result.crag_info
        ccol1, ccol2, ccol3 = st.columns(3)
        ccol1.metric("متوسط الصلة (average_relevance)", f"{crag.get('average_relevance', 0):.3f}")
        ccol2.metric("جولات التصحيح", crag.get("corrective_rounds", 0))
        ccol3.metric("تم التصحيح؟", "نعم" if crag.get("triggered_correction") else "لا")
        if crag.get("corrective_queries"):
            st.markdown("استعلامات التصحيح:")
            for q in crag["corrective_queries"]:
                st.write(f"- {q}")

    cost = result.cost_summary or {}
    st.markdown("**التكلفة والاستخدام (cost_summary):**")
    ucol1, ucol2, ucol3, ucol4 = st.columns(4)
    ucol1.metric("عدد استدعاءات النموذج", cost.get("num_calls", 0))
    ucol2.metric("إجمالي رموز الإدخال", cost.get("total_prompt_tokens", 0))
    ucol3.metric("إجمالي رموز الإخراج", cost.get("total_completion_tokens", 0))
    ucol4.metric("التكلفة التقديرية (USD)", f"{cost.get('total_cost_usd', 0):.6f}")

    by_purpose = cost.get("by_purpose", {})
    if by_purpose:
        st.markdown("**تفصيل الاستدعاءات حسب الغرض (by_purpose):**")
        rows = [
            {
                "الغرض": purpose,
                "عدد الاستدعاءات": stats.get("num_calls", 0),
                "رموز الإدخال": stats.get("prompt_tokens", 0),
                "رموز الإخراج": stats.get("completion_tokens", 0),
                "التكلفة (USD)": stats.get("cost_usd", 0.0),
            }
            for purpose, stats in by_purpose.items()
        ]
        st.dataframe(rows, width="stretch", hide_index=True)

    st.caption(
        f"total_latency_ms (من نتيجة الأنبوب - يعكس آخر خطوة توليد فقط): "
        f"{result.total_latency_ms:.1f} ms"
    )

    if result.errors:
        st.warning("أخطاء تم رصدها أثناء المعالجة (تمت معالجتها تلقائياً قدر الإمكان):")
        for err in result.errors:
            st.write(f"- {err}")


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []  # each: {role, content, citations?, result?, elapsed?}


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("الإعدادات")

    if st.button("🗑️ مسح المحادثة", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    pdf_dir = Path("data/pdfs")
    if pdf_dir.exists():
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        st.markdown(f"**الوثائق المتاحة ({len(pdfs)}):**")
        for p in pdfs:
            st.caption(f"📄 {p.name}")

    st.divider()

    reports_dir = Path("reports")
    if reports_dir.exists():
        report_files = sorted(reports_dir.glob("*.md")) + sorted(reports_dir.glob("*.csv"))
        if report_files:
            with st.expander("📊 تقارير التقييم الموجودة"):
                selected = st.selectbox(
                    "اختر تقريراً لعرضه",
                    options=report_files,
                    format_func=lambda p: p.name,
                )
                if selected:
                    if selected.suffix == ".md":
                        st.markdown(selected.read_text(encoding="utf-8"))
                    elif selected.suffix == ".csv":
                        import pandas as pd

                        st.dataframe(pd.read_csv(selected), width="stretch")


# --------------------------------------------------------------------------- #
# Main chat area
# --------------------------------------------------------------------------- #
st.title("📄 نظام الأسئلة والأجوبة - الوثائق الإجرائية")
st.caption("اسأل عن أي إجراء وارد في الأدلة، أو ابدأ بتحية عادية.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            st.markdown("**المصادر:**")
            for line in msg["citations"]:
                st.write(f"- {line}")
        if msg.get("result") is not None:
            with st.expander("🔍 تفاصيل الاسترجاع (Retrieval Details)"):
                render_retrieval_details(msg["result"], msg.get("elapsed", 0.0))

user_query = st.chat_input("اكتب سؤالك هنا...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("جاري البحث والمعالجة..."):
            try:
                pipeline = get_pipeline()
                start = time.perf_counter()
                result: PipelineResult = pipeline.answer(user_query)
                elapsed = time.perf_counter() - start
            except Exception as exc:  # noqa: BLE001
                st.error(f"حدث خطأ أثناء معالجة السؤال: {exc}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"حدث خطأ أثناء معالجة السؤال: {exc}"}
                )
                st.stop()

        st.markdown(result.answer)

        citation_lines = format_citations(result.citations)
        if citation_lines:
            st.markdown("**المصادر:**")
            for line in citation_lines:
                st.write(f"- {line}")

        with st.expander("🔍 تفاصيل الاسترجاع (Retrieval Details)"):
            render_retrieval_details(result, elapsed)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "citations": citation_lines,
            "result": result,
            "elapsed": elapsed,
        }
    )