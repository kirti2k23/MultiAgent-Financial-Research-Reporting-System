# Streamlit demo UI — enter ticker, see report rendered live
"""
dashboard/app.py
================
Streamlit dashboard — the user-facing interface for the research agent.

WHAT THIS FILE DOES:
    Provides a simple web UI where a manager can:
    1. Enter a company name and ticker
    2. Click "Generate Report"
    3. See the full research report with key metrics

HOW IT CONNECTS TO THE PIPELINE:
    Dashboard → POST /api/v1/analyze → API → Pipeline → Report
    The dashboard never calls the pipeline directly.
    It always goes through the API.

HOW TO RUN:
    Make sure the API is running first:
        uvicorn src.api.main:app --port 8000

    Then in a second terminal:
        cd financial-research-agent
        streamlit run dashboard/app.py

    Open browser at: http://localhost:8501

DEPENDENCIES:
    pip install streamlit requests
"""

import requests
import streamlit as st

# ── API configuration ─────────────────────────────────────────────────────────
API_BASE_URL = "http://localhost:8000/api/v1"


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# Must be the first Streamlit command called.
# Sets the browser tab title and page layout.
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Financial Research Agent",
    page_icon  = "📊",
    layout     = "wide",
)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.title("📊 Financial Research Agent")
st.markdown(
    "Autonomous AI system that generates professional research reports "
    "by analyzing news, stock data, and SEC filings."
)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────────────────────────────────────

# Use two columns for side-by-side inputs
col1, col2 = st.columns(2)

with col1:
    company_name = st.text_input(
        label       = "Company Name",
        placeholder = "e.g. NVIDIA Corporation",
        help        = "Full legal name of the company"
    )

with col2:
    ticker = st.text_input(
        label       = "Ticker Symbol",
        placeholder = "e.g. NVDA",
        help        = "Stock ticker symbol (will be uppercased automatically)"
    )

# Generate button — centered
col_left, col_center, col_right = st.columns([2, 1, 2])
with col_center:
    generate = st.button(
        "🔍 Generate Report",
        type = "primary",
        use_container_width = True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE EXECUTION
# Runs when the user clicks "Generate Report"
# ─────────────────────────────────────────────────────────────────────────────

if generate:

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not company_name or not ticker:
        st.error("Please enter both company name and ticker symbol.")
        st.stop()

    # ── Check API is running ──────────────────────────────────────────────────
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=5)
        health.raise_for_status()
    except Exception:
        st.error(
            "⚠️ API is not running. Please start it first:\n\n"
            "```bash\nuvicorn src.api.main:app --port 8000\n```"
        )
        st.stop()

    # ── Run pipeline ──────────────────────────────────────────────────────────
    # st.spinner shows a loading animation while the pipeline runs
    with st.spinner(
        f"Generating research report for {company_name} ({ticker.upper()})... "
        "This takes 2-3 minutes."
    ):
        try:
            response = requests.post(
                f"{API_BASE_URL}/analyze",
                json    = {
                    "company_name": company_name,
                    "ticker":       ticker.upper(),
                },
                # timeout=300 → allow up to 5 minutes for the pipeline
                timeout = 300,
            )
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The pipeline took too long. Try again.")
            st.stop()

        except requests.exceptions.HTTPError as e:
            st.error(f"❌ API error: {e}")
            st.stop()

        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            st.stop()

    # ── Display results ───────────────────────────────────────────────────────
    st.success("✅ Report generated successfully!")
    st.divider()

    # ── Metrics row ───────────────────────────────────────────────────────────
    # Show key metrics in a clean row of cards
    st.subheader("📈 Key Metrics")

    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        price = data.get("current_price")
        st.metric(
            label = "Stock Price",
            value = f"${price}" if price else "N/A",
        )

    with m2:
        mcap = data.get("market_cap_billions")
        st.metric(
            label = "Market Cap",
            value = f"${mcap}B" if mcap else "N/A",
        )

    with m3:
        pe = data.get("pe_ratio")
        st.metric(
            label = "P/E Ratio",
            value = f"{pe}x" if pe else "N/A",
        )

    with m4:
        rev = data.get("revenue_growth_pct")
        st.metric(
            label = "Revenue Growth",
            value = f"{rev}%" if rev else "N/A",
        )

    with m5:
        # Sentiment with emoji indicator
        label = data.get("sentiment_label", "N/A")
        score = data.get("sentiment_score")
        emoji = "🟢" if label == "positive" else "🔴" if label == "negative" else "🟡"
        st.metric(
            label = "Sentiment",
            value = f"{emoji} {label.capitalize()}" if label != "N/A" else "N/A",
            delta = f"{score}" if score else None,
        )

    st.divider()

    # ── News summary ──────────────────────────────────────────────────────────
    news_summary = data.get("news_summary")
    if news_summary:
        st.subheader("📰 News Summary")
        st.info(news_summary)

    st.divider()

    # ── Final report ──────────────────────────────────────────────────────────
    st.subheader("📄 Full Research Report")

    report = data.get("final_report")
    if report:
        # st.markdown renders markdown formatting (headers, bold, bullets)
        st.markdown(report)
    else:
        st.warning("Report generation failed — no report available.")

    st.divider()

    # ── Pipeline metadata ─────────────────────────────────────────────────────
    with st.expander("🔧 Pipeline Details"):
        st.write("**Completed agents:**", data.get("completed_agents"))
        st.write("**Filing source:**", data.get("filing_source"))

        errors = data.get("errors")
        if errors:
            st.warning(f"Non-fatal errors: {errors}")

    # ── Download button ───────────────────────────────────────────────────────
    # Let the user download the report as a text file
    if report:
        st.download_button(
            label    = "⬇️ Download Report",
            data     = report,
            file_name = f"{ticker.upper()}_research_report.md",
            mime     = "text/markdown",
        )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# Shows instructions and example companies
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("ℹ️ How to use")
    st.markdown("""
    1. Enter the **company name** and **ticker symbol**
    2. Click **Generate Report**
    3. Wait 2-3 minutes for the pipeline to run
    4. View the report and download it
    """)

    st.divider()

    st.header("💡 Example companies")
    st.markdown("""
    | Company | Ticker |
    |---------|--------|
    | NVIDIA Corporation | NVDA |
    | Apple Inc | AAPL |
    | Microsoft Corporation | MSFT |
    | Tesla Inc | TSLA |
    | Amazon.com Inc | AMZN |
    """)

    st.divider()

    st.header("⚙️ System status")

    # Check if API is running and show status
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if health.status_code == 200:
            st.success("API: ✅ Running")
        else:
            st.error("API: ❌ Error")
    except Exception:
        st.error("API: ❌ Not running")
        st.code("uvicorn src.api.main:app --port 8000")