from pathlib import Path
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber

# 📍 Pad naar je bronbestand (PDF)
pdf_path = Path("data/raw_book/InFloBook.pdf")

# 📤 Extract tekst uit PDF
text = ""
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text += page.extract_text() + "\n"

# 🔪 Chunken met langchain's TextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_text(text)

# ✅ Opslaan als JSON
output_path = Path("data/processed/chunks_AlisaVita.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"✅ Extracted and chunked {len(chunks)} passages to {output_path}")

import os
import json
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()
persist_dir = "data/vectorstore"

# 📖 Chunks laden
with open("data/processed/chunks_AlisaVita.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

documents = [
    Document(page_content=chunk, metadata={"source": "AlisaVita"})
    for chunk in chunks
]

# 🔑 Embedding model
embedding_model = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

# 💾 Vectorstore bouwen en opslaan
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory=persist_dir,
    collection_name="langchain"
)
vectorstore.persist()

print(f"✅ Vectorstore created and saved to: {persist_dir}")