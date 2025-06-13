import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

# 🔐 Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

print("🔑 API key found?", openai_key is not None)

# ⚠️ Deprecation waarschuwing negeren voorlopig
embedding_model = OpenAIEmbeddings(openai_api_key=openai_key)

# 📦 Load vectorstore
vectorstore = Chroma(
    persist_directory="data/vectorstore",
    embedding_function=embedding_model
)

# 🧠 Inspect vectorstore
print("✅ Total documents stored:", vectorstore._collection.count())