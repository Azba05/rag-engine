import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROK_API_KEY"),
    base_url="https://api.x.ai/v1"
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

SYSTEM_PROMPT = """
You are a Retrieval-Augmented Financial Analyst.

Your ONLY source of truth is the retrieved context below.

Never use outside knowledge.

Never use your training knowledge.

Never guess.

Never infer.

Never estimate.

Never complete missing information.

If the retrieved context does not explicitly contain the answer, respond EXACTLY:

"The retrieved documents do not contain sufficient information to answer this question."

--------------------------------------------------
Rules
--------------------------------------------------

1. Answer ONLY from the retrieved context.

2. Maximum 5 sentences.

3. Be concise and professional.

4. Copy all financial values exactly as written.

5. Preserve units exactly (₹, $, crore, million, billion, %, etc.).

6. Do not round numbers.

7. Do not calculate missing values.

8. Do not interpret financial information.

9. Do not explain concepts unless they appear in the context.

10. If multiple retrieved chunks disagree, say:

"The retrieved documents contain conflicting information."

Then briefly summarize both versions.

11. If the question asks "Why", only answer if the reason is explicitly stated.

Otherwise respond:

"The retrieved documents do not specify the reason."

12. Never mention information from companies that are not present in the retrieved context.

13. Never mention assumptions.

14. Never fabricate names, dates, financial values, percentages or business strategies.

--------------------------------------------------
Output Format
--------------------------------------------------

Answer:
<3–5 sentence answer>

Source:
<Company Name>

--------------------------------------------------

Context:
{context}

Question:
{question}

Answer:
"""

COMPANIES = {
    "reliance":"RELIANCE2025_2026",
    "tcs":"TCS2025_2026",
    "titan":"TITAN2025_2026",
    "itc":"ITC2025_2026",
    "tata steel":"TATASTEEL2025_2026",
    "tatasteel":"TATASTEEL2025_2026",
    "tata power":"TATAPOWER2025_2026",
    "tatapower":"TATAPOWER2025_2026",
    "infosys":"INFOSYS2025_2026",
    "wipro":"WIPRO2025_2026",
    "cipla":"CIPLA2025_2026",
    "asian paints":"ASIANPAINTS2025_2026",
    "asianpaints":"ASIANPAINTS2025_2026",
    "ambuja":"AMBUJACEMENTS2025_2026",
    "sun pharma":"SUNPHARMA2025_2026",
    "sunpharma":"SUNPHARMA2025_2026",
    "torrent pharma":"TORRENTPHARMA2025_2026",
    "tech mahindra":"TECHM2025_2026",
    "techm":"TECHM2025_2026",
    "jsw":"JSWSTEEL2025_2026",
    "jsw steel":"JSWSTEEL2025_2026",
    "tvs":"TVSMOTOR2025_2026",
    "tvs motor":"TVSMOTOR2025_2026",
    "colgate":"COLGATE2025_2026",
    "adani power":"ADANIPOWER2025_2026",
    "adanipower":"ADANIPOWER2025_2026",
    "mahindra":"M&M2025_2026",
    "m&m":"M&M2025_2026"
}


def detect_company(question):

    q = question.lower()

    for alias, company in sorted(
        COMPANIES.items(),
        key=lambda x: len(x[0]),
        reverse=True,
    ):
        if re.search(r"\b"+re.escape(alias)+r"\b", q):
            return company

    return None


def ask_grok(question):

    company = detect_company(question)

    if company is None:
        return {
            "company_found":False,
            "answer":"Requested company not found in the knowledge base.",
            "sources":[]
        }

    docs = vectorstore.similarity_search_with_score(
        question,
        k=8,
        filter={"company":company}
    )

    filtered=[]
    seen=set()

    for doc,score in docs:

        if score>1:
            continue

        if doc.page_content in seen:
            continue

        seen.add(doc.page_content)
        filtered.append(doc)

    if len(filtered)==0:
        return {
            "company_found":True,
            "answer":"The retrieved documents do not contain sufficient information to answer this question.",
            "sources":[]
        }

    context="\n\n".join(d.page_content for d in filtered)

    response=client.chat.completions.create(
        model="grok-4",
        temperature=0,
        messages=[
            {
                "role":"system",
                "content":SYSTEM_PROMPT
            },
            {
                "role":"user",
                "content":f"""
Context:

{context}

Question:

{question}
"""
            }
        ]
    )

    answer=response.choices[0].message.content

    return {
        "company_found":True,
        "answer":answer,
        "sources":filtered
    }