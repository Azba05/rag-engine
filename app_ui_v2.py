import streamlit as st
import requests

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="NSE Financial RAG Analytics",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Enterprise Financial Analyst Dashboard")
st.caption(
    "AI-powered financial document retrieval and analysis using Retrieval-Augmented Generation (RAG)"
)

st.markdown("---")

# --------------------------------------------------
# Layout
# --------------------------------------------------
col1, col2 = st.columns([1, 3])

with col1:

    st.subheader("Target Filters")

    company = st.selectbox(
        "Select Company Context:",
        [
            "Infosys",
            "Reliance",
            "TCS",
            "Titan",
            "ITC",
            "Tata Steel",
            "Adani Power",
            "Mahindra",
            "Bajaj Auto",
            "Wipro",
            "Cipla",
            "Asian Paints",
            "Ambuja Cement",
            "Sun Pharma",
            "Torrent Pharma",
            "Tech Mahindra",
            "JSW Steel",
            "TVS Motor",
            "Colgate",
            "Tata Power",
        ]
    )

with col2:

    st.subheader("Query Details")

    user_query = st.text_input(
        f"Ask a financial question about {company}:",
        value=f"What is the revenue from operations for {company}?",
        placeholder="Example: What was the net profit?"
    )

    execute_button = st.button(
        "Run Financial Extraction",
        type="primary",
        use_container_width=True
    )

# --------------------------------------------------
# Query Backend
# --------------------------------------------------
if execute_button:

    with st.spinner("Searching annual reports..."):

        try:

            payload = {
                "question": user_query
            }

            response = requests.post(
                "http://127.0.0.1:8000/ask",
                json=payload
            )

            if response.status_code == 200:

                result = response.json()

                if not result["company_found"]:

                    st.error(
                        "⚠️ Company could not be identified from the question."
                    )

                else:

                    # --------------------------------------------------
                    # Retrieved Answer
                    # --------------------------------------------------

                    st.subheader("📊 Retrieved Financial Information")

                    st.text_area(
                        label="Retrieved Context",
                        value=result["answer"],
                        height=350
                    )

                    # --------------------------------------------------
                    # Source Documents
                    # --------------------------------------------------

                    st.markdown("---")
                    st.subheader("📚 Source Documents")

                    if result["sources"]:

                        for i, source in enumerate(result["sources"], start=1):

                            with st.expander(
                                f"📄 Source {i} • {source.get('source', 'Unknown')}",
                                expanded=False
                            ):

                                st.write(
                                    f"**Company:** {source.get('company', 'Unknown')}"
                                )

                                st.write(
                                    f"**Document:** {source.get('source', 'Unknown')}"
                                )

                                st.write(
                                    f"**PDF Page:** {source.get('page', 'N/A')}"
                                )

                                st.write(
                                    f"**Printed Page:** {source.get('printed_page', source.get('page', 'N/A'))}"
                                )

                                st.write(
                                    f"**Similarity Score:** {source.get('score', 'N/A')}"
                                )

                    else:

                        st.info("No source information available.")

            else:

                st.error(
                    f"Backend Error (HTTP {response.status_code})"
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "🔌 Connection Refused!\n\n"
                "Please start the FastAPI backend first:\n\n"
                "`uvicorn main:app --reload`"
            )

        except Exception as e:

            st.exception(e)