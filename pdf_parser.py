import os
import json
import fitz

INPUT_FOLDER = "data/reports"
OUTPUT_FILE = "data/chunks.json"

#chunking 250/50
def chunk_text(text, chunk_size=250, overlap=50):
    words = text.split()

    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):

        chunk = " ".join(words[i:i + chunk_size])

        if chunk.strip():
            chunks.append(chunk)

    return chunks


documents = []

for pdf in os.listdir(INPUT_FOLDER):

    if not pdf.endswith(".pdf"):
        continue

    company = os.path.splitext(pdf)[0]

    pdf_path = os.path.join(INPUT_FOLDER, pdf)

    print(f"Processing {pdf}")

    doc = fitz.open(pdf_path)

    chunk_id = 0

    for page_number, page in enumerate(doc):

        text = page.get_text("text")

        if not text:
            continue

        text = " ".join(text.split())

        # Skip pages with almost no text
        if len(text.split()) < 100:
            continue

        # Skip obvious divider pages
        divider_keywords = [
            "Portfolio Overview",
            "Corporate Overview",
            "Strategic Review",
            "Statutory Reports",
            "Financial Statements",
            "Integrated Annual Report"
        ]

        # If the page is very short and mostly contains section titles
        if (
            len(text.split()) < 180
            and sum(k in text for k in divider_keywords) >= 2
        ):
            continue

        page_chunks = chunk_text(text)

        for chunk in page_chunks:

            chunk = chunk.strip()

            # Skip tiny chunks
            if len(chunk.split()) < 80:
                continue

            documents.append(
                {
                    "company": company,
                    "pdf_file": pdf,
                    "page": page_number + 1,
                    "chunk_id": chunk_id,
                    "tokens": len(chunk.split()),
                    "text": chunk,
                }
            )

            chunk_id += 1

print(f"\nTotal chunks created: {len(documents)}")

os.makedirs("data", exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(documents, f, indent=2, ensure_ascii=False)

print(f"Saved to {OUTPUT_FILE}")