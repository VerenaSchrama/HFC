# ✅ Nieuwe setup met FAISS in plaats van Chroma

import os
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import json

load_dotenv()

openai_api_key = st.secrets["OPENAI_API_KEY"]

# ✅ Nieuwe FAISS-based RAG chain
@st.cache_resource
def load_rag_chain():
    # Load embedding model
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Load local chunks from JSON
    with open("data/processed/chunks_AlisaVita.json", "r") as f:
        chunks = json.load(f)

    # Create FAISS vectorstore from text chunks
    vectorstore = FAISS.from_texts(chunks, embedding_model)

    retriever = vectorstore.as_retriever()

    # Prompt
    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template="""
You are a cycle-aware nutrition assistant based on holistic and scientific insights.

Always answer user questions helpfully and always provide answers in a warm, empowering tone.

If the user is asking about different cycle phases, use the following structure:

**Menstrual Phase**
- Key needs:
- Recommended foods:

**Follicular Phase**
- Key needs:
- Recommended foods:

**Ovulatory Phase**
- Key needs:
- Recommended foods:

**Luteal Phase**
- Key needs:
- Recommended foods:

For the foods, refer to ingredients and nutrients rather than recipes or dishes.

If the source materials really don't mention anything related to the question, say:
“There are limited recommendations for your question based on science, but what science does advise is…”

Be concise, clear, and nurturing in your responses.
Keep a warm and empowering tone.

Answer based on the context below:

Context:
{context}

Question:
{question}
"""
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key),
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        return_source_documents=True,
        output_key="answer"
    )

    return qa_chain


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