# utils.py
import os
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 🔐 Load .env if needed (fallback to env var)
load_dotenv()

# Get key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# ✅ RAG-based model loading
@st.cache_resource
def load_rag_chain():
    # 1. Load embedding model
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # 2. Load persisted vectorstore
    vectorstore = Chroma(
        persist_directory="data/vectorstore",  # ✅ Correct path
        embedding_function=embedding_model
    )
    retriever = vectorstore.as_retriever()

    # 3. Define the system prompt
    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template="""
You are a cycle-aware nutrition assistant based on holistic and scientific insights.

Always answer user questions helpfully and always provide answers in a warm, empowering tone.

If the source materials don’t mention something, don’t say “The text does not provide…” — instead say:
“There are no specific recommendations for your question in science, but what science does advise is…”

Be concise, clear, and nurturing in your responses.

Answer based on the context below:

Context:
{context}

Question:
{question}
"""
    )

    # 4. Memory to keep chat history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # 5. Combine in a retrieval chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key),
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        return_source_documents=True,
        output_key="answer"
    )

    return qa_chain

# ✅ Optional session tools
def reset_session():
    keys_defaults = {
        "phase": None,
        "cycle_length": None,
        "dietary_preferences": [],
        "support_goal": None,
        "last_period": None,
        "second_last_period": None,
        "personalization_completed": False,
        "chat_history": []
    }
    for key, default in keys_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

def add_to_chat_history(role, message):
    if st.session_state.chat_history is None:
        st.session_state.chat_history = []
    st.session_state.chat_history.append((role, message))