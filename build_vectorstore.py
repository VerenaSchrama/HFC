import os
import json
from pathlib import Path
from dotenv import load_dotenv
import pdfplumber

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# === 1. Load environment ===
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
assert openai_key, "❌ OPENAI_API_KEY not found in .env"

# === 2. Load PDF and extract text ===
pdf_path = Path("data/raw_book/InFloBook.pdf")
text = ""
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text += page.extract_text() + "\n"

print(f"📄 Extracted text from {len(pdf.pages)} pages.")

# === 3. Chunk text ===
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

chunks = chunk_text(text)
print(f"✂️ Chunked into {len(chunks)} blocks.")

# === 4. Save chunks as JSON ===
json_path = Path("data/processed/chunks_AlisaVita.json")
json_path.parent.mkdir(parents=True, exist_ok=True)
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"💾 Chunks saved to {json_path}")

# === 5. Embed and store in Chroma ===
embedding_model = OpenAIEmbeddings(openai_api_key=openai_key)
persist_dir = "data/vectorstore"

vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embedding_model,
    persist_directory=persist_dir
)
vectorstore.persist()

print(f"✅ Vectorstore created and saved to {persist_dir}")