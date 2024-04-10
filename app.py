import streamlit as st
import os
import qdrant_client
from dotenv import load_dotenv
from langchain import hub
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatAnyscale
from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv()
ANYSCALE_API_KEY = os.getenv("ANYSCALE_API_KEY")

llm = ChatAnyscale(model_name="mistralai/Mistral-7B-Instruct-v0.1")
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# initializing qdrant client
client = qdrant_client.QdrantClient(
    "https://e5109977-5004-4f22-95fd-3cf22b4b639a.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key=os.getenv("QDRANT_API_KEY"),
)

# setting up retriever
doc_store = Qdrant(
    client=client,
    collection_name="sl_marraige_law",
    embeddings=embeddings,
)

retriever = doc_store.as_retriever(search_type="mmr", search_kwargs=dict(k=3))
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def chat(question):
    ai_msg = rag_chain.invoke(question)
    return ai_msg


st.title("SL Marraige Law Q/A Chatbot")


if "conversation" not in st.session_state:
    st.session_state.conversation = []


def submit_question():
    if st.session_state.question.strip() != "":
        response = chat(st.session_state.question)
        st.session_state.conversation.append(("You", st.session_state.question))
        st.session_state.conversation.append(("Chatbot", response))
        st.session_state.question = ""


user_question = st.text_input(
    "Ask me anything:", key="question", on_change=submit_question
)
for speaker, message in st.session_state.conversation:
    st.markdown(f"**{speaker}:** {message}")