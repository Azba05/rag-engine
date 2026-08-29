#1. imports
import os
import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

#2. Load environment variables
load_dotenv()
# --------------------------------------------------
# OpenRouter LLM Configuration
# --------------------------------------------------

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY not found in .env file"
    )

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost",
    "X-OpenRouter-Title": "NSE Financial RAG Analytics"
}

#3. Initialize the embedding function for vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#4. Initialize the persistent Chroma DB
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

COMPANIES = {
    "reliance": "RELIANCE2025_2026",

    "tcs": "TCS2025_2026",

    "titan": "TITAN2025_2026",

    "itc": "ITC2025_2026",

    "tata steel": "TATASTEEL2025_2026",
    "tatasteel": "TATASTEEL2025_2026",

    "tata power": "TATAPOWER2025_2026",
    "tatapower": "TATAPOWER2025_2026",

    "infosys": "INFOSYS2025_2026",

    "wipro": "WIPRO2025_2026",

    "cipla": "CIPLA2025_2026",

    "asian paints": "ASIANPAINTS2025_2026",
    "asianpaints": "ASIANPAINTS2025_2026",

    "ambuja": "AMBUJACEMENT2025_2026",
    "ambuja cement": "AMBUJACEMENT2025_2026",

    "sun pharma": "SUNPHARMA2025_2026",
    "sunpharma": "SUNPHARMA2025_2026",

    "torrent pharma": "TORRENTPHARMA2025_2026",
    "torrentpharma": "TORRENTPHARMA2025_2026",

    "tech mahindra": "TECHM2025_2026",
    "techm": "TECHM2025_2026",

    "jsw": "JSWSTEEL2025_2026",
    "jsw steel": "JSWSTEEL2025_2026",

    "tvs": "TVSMOTOR2025_2026",
    "tvs motor": "TVSMOTOR2025_2026",

    "colgate": "COLGATE2025_2026",

    "adani power": "ADANIPOWER2025_2026",
    "adanipower": "ADANIPOWER2025_2026",

    "mahindra": "M&M2025_2026",
    "m&m": "M&M2025_2026"
}

#6.5. LLM Generation Function
def generate_llm_answer(question, retrieved_context):

    prompt = f"""
You are a financial analyst.

Answer the user's question ONLY using the retrieved context below.

IMPORTANT RULES:
1. Do not use outside knowledge.
2. Do not invent or estimate financial figures.
3. Do not hallucinate.
4. If the required information is not present in the context, clearly say that the information was not found.
5. For comparison questions, compare the companies using only figures explicitly present in the retrieved context.
6. Preserve financial numbers exactly as provided in the context.
7. Give a concise, professional financial answer.
8. Mention the relevant company names and figures clearly.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{retrieved_context}

Now provide the final answer.
"""

    payload = {
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=60
        )

        print("\n========== OPENROUTER ==========")
        print("Status:", response.status_code)

        data = response.json()

        print("Response received from OpenRouter.")
        print("================================\n")

        # Check API error
        if response.status_code != 200:

            print("OpenRouter Error:")
            print(data)

            return (
                "The language model could not generate an answer. "
                "Please try the query again."
            )

        # Check choices
        if not data.get("choices"):

            print("No choices returned by OpenRouter.")
            print(data)

            return (
                "The language model returned no answer."
            )

        message = data["choices"][0].get("message", {})

        # Get generated content safely
        content = message.get("content")

        if content is not None and str(content).strip():

            return str(content).strip()

        # Some models/providers may return reasoning instead
        reasoning = message.get("reasoning")

        if reasoning is not None and str(reasoning).strip():

            print("Warning: Model returned reasoning but no final content.")

            return (
                "The language model did not return a final answer."
            )

        # Nothing usable was returned
        print("No usable content returned.")
        print("Message:", message)

        return (
            "The language model did not return a usable answer."
        )

    except requests.exceptions.RequestException as e:

        print("OpenRouter request error:", e)

        return (
            "Unable to connect to the language model."
        )

    except Exception as e:

        print("LLM processing error:", e)

        return (
            "An error occurred while generating the answer."
        )


#7. retrieval function
def retrieve_answers(question):
    """
    Pure retrieval function: Searches Chroma vector store for relevant chunks
    and returns them directly without sending them through an LLM.
    """
    
    keywords = {

    # Financial Statements
    "revenue": "revenue sales turnover income revenue from operations top line",

    "profit": "profit net profit PAT earnings profit after tax PBT",

    "assets": "assets total assets current assets non-current assets property plant equipment PPE",

    "liabilities": "liabilities total liabilities borrowings obligations debt current liabilities",

    "cash": "cash cash flow operating cash flow free cash flow financing cash flow",

    "dividend": "dividend payout dividend per share DPS",

    # Financial Metrics
    "expenses": "expenses expenditure operating expenses finance cost depreciation amortization",

    "ebitda": "EBITDA operating profit earnings before interest tax depreciation amortization",

    "margin": "margin gross margin operating margin EBITDA margin net margin",

    "eps": "EPS earnings per share diluted EPS basic EPS",

    "debt": "debt borrowings loans leverage gearing",

    "working capital": "working capital inventory receivables payables",

    "capex": "capital expenditure CAPEX investment expansion projects",

    # Business
    "business": "business model operations products services",

    "segment": "segment business segment operating segment geographical segment",

    "market": "market domestic international exports imports",

    "customers": "customers clients consumers customer",

    # Strategy
    "strategy": "strategy strategic priorities roadmap transformation initiatives",

    "future": "future outlook growth plans expansion opportunities guidance",

    "growth": "growth expansion future opportunities",

    "risk": "risk risks risk management risk factors uncertainties threats",

    "competition": "competition competitors competitive",

    # ESG
    "esg": "ESG sustainability environmental social governance",

    "climate": "climate carbon emissions net zero renewable green energy",

    "csr": "CSR corporate social responsibility community",

    # Governance
    "board": "board directors committees governance leadership",

    "shareholders": "shareholders equity share capital investor relations",

    # Technology
    "innovation": "innovation research development R&D patents",

    "digital": "digital transformation AI automation analytics",

    "cybersecurity": "cybersecurity cyber security information security",

    # Misc
    "subsidiaries": "subsidiaries joint ventures associates",

    "acquisition": "acquisition merger acquisition partnership",

    "manufacturing": "manufacturing plants factories production capacity"
}

    expanded_question = question.lower()

    for key, value in keywords.items():
        if key in expanded_question:
            expanded_question += " " + value

    docs = vectorstore.similarity_search_with_score(
        expanded_question,
        k=20
    )

    print("\n========== RETRIEVAL DEBUG ==========")
    print("Question:", question)
    print("Expanded question:", expanded_question)
    print("Documents retrieved:", len(docs))

    for i, (doc, score) in enumerate(docs):
       print(f"\n--- Result {i+1} ---")
       print("Score:", score)
       print("Company:", doc.metadata.get("company"))
       print("Source:", doc.metadata.get("source"))
       print("Page:", doc.metadata.get("page"))
       print("Content:", doc.page_content[:500])

    print("=====================================\n")

    filtered = []
    seen = set()

    for doc, score in docs:

        content = doc.page_content.strip().lower()

        if content in seen:
            continue

        seen.add(content)

        filtered.append((doc, score))

        print("Documents after filtering:", len(filtered))

    # Sort by similarity (lowest score = best)
    filtered.sort(key=lambda x: x[1])

    # Keep only top 5
    filtered = filtered[:5]

    if not filtered:
        return {
            "answer": "The retrieved documents do not contain sufficient information to answer this question.",
            "sources": []
        }

    retrieved_text = ""

    sources = []

    for i, (doc, score) in enumerate(filtered, start=1):

        retrieved_text += f"[Result {i}]\n{doc.page_content}\n\n"

        sources.append({
            "company": doc.metadata.get("company"),
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "printed_page": doc.metadata.get(
                "printed_page",
                doc.metadata.get("page")
            ),
            "score": round(score, 4)
        })

    # Generate final answer using retrieved context

    print("\n========== LLM INPUT ==========")
    print("Question:", question)
    print("Retrieved context length:", len(retrieved_text))
    print("Retrieved context:")
    print(retrieved_text)
    print("===============================\n")

    llm_answer = generate_llm_answer(
        question,
        retrieved_text
    )

    return {
        "answer": llm_answer,
        "sources": sources
    }