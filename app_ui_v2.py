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
   "AI-powered financial analysis and document retrieval across 20 NSE-listed companies"
)

st.markdown("---")

# --------------------------------------------------
# Layout
# --------------------------------------------------
st.subheader("Query Details")

user_query = st.text_input(
    "Ask a financial question:",
    placeholder="Example: what is the revenue operation of infosys?"
)

execute_button = st.button(
    "Run Financial Analysis",
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