import os
import pymupdf  
import json
from pathlib import Path

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF"""
    doc = pymupdf.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        text += page.get_text()
    doc.close()
    return text

def chunk_text(text, chunk_size=512, overlap=100):
    """Split text into chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def process_all_pdfs():
    """Process all PDFs in data/reports/"""
    reports_dir = "data/reports"
    
    if not os.path.exists(reports_dir):
        print(f"Error: {reports_dir} folder not found!")
        return
    
    all_chunks = []
    
    for pdf_file in os.listdir(reports_dir):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(reports_dir, pdf_file)
            company_name = pdf_file.replace('.pdf', '').replace('_fy2025', '').replace('_fy2026', '')
            
            print(f"Processing: {pdf_file}...")
            
            text = extract_text_from_pdf(pdf_path)
            
            chunks = chunk_text(text)
            
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    "company": company_name,
                    "pdf_file": pdf_file,
                    "chunk_id": i,
                    "text": chunk,
                    "tokens": len(chunk.split())
                }
                all_chunks.append(chunk_data)
            
            print(f"  → Created {len(chunks)} chunks from {company_name}")
    
    output_file = "data/chunks.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    print(f"\n Total chunks created: {len(all_chunks)}")
    print(f" Chunks saved to: {output_file}")

if __name__ == "__main__":
    process_all_pdfs()