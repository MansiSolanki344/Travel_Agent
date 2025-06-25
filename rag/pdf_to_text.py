# rag/pdf_to_text.py
import fitz  # PyMuPDF
from pathlib import Path

def extract_clean_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text.strip().replace("\n", " ")

def convert_all_pdfs(source_dir="rag/books", target_dir="rag/clean"):
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    for file in Path(source_dir).glob("*.pdf"):
        print(f"📄 Converting {file.name}")
        text = extract_clean_text_from_pdf(file)
        out_path = Path(target_dir) / f"{file.stem}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    convert_all_pdfs()
