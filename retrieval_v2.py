#1. imports
import os
import re
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

#2. Load environment variables
load_dotenv()

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

def detect_company(question):
    """
    Detects the company mentioned in the user's question.
    Returns the company identifier used in Chroma metadata.
    """

    question = question.lower()

    for alias, company in sorted(
        COMPANIES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        pattern = r"\b" + re.escape(alias) + r"\b"

        if re.search(pattern, question):
            return company

    return None

#7. retrieval function
def retrieve_answers(question):
    """
    Pure retrieval function: Searches Chroma vector store for relevant chunks
    and returns them directly without sending them through an LLM.
    """
    company = detect_company(question)

    if company is None:
        return {
            "company_found": False,
            "answer": "Requested company not found in the knowledge base.",
            "sources": []
        }

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
        k=15,
        filter={"company": company}
    )

    filtered = []
    seen = set()

    for doc, score in docs:

        # Skip weak matches
        if score > 0.85:
            continue

        content = doc.page_content.strip().lower()

        if content in seen:
            continue

        seen.add(content)

        filtered.append((doc, score))

    # Sort by similarity (lowest score = best)
    filtered.sort(key=lambda x: x[1])

    # Keep only top 5
    filtered = filtered[:5]

    if not filtered:
        return {
            "company_found": True,
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

    return {
        "company_found": True,
        "answer": retrieved_text,
        "sources": sources
    }