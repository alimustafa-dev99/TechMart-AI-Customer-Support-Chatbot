import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
import streamlit as st

load_dotenv()

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "c:/gen ai internship/4th project/data/TechMart Customer Support Guidebook.pdf"
DB_PATH = "c:/gen ai internship/4th project/data/store_db"

st.set_page_config(page_title="TechMart Support - Alex", page_icon="🛒")
st.title("🛒 TechMart Support (Alex)")


@st.cache_resource
def load_rag():
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.2)
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    # Load existing Chroma DB if built; otherwise build it once from the PDF
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    else:
        loader = PyPDFLoader(file_path=PDF_PATH)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splitted_doc = splitter.split_documents(docs)
        store = Chroma.from_documents(
            documents=splitted_doc,
            embedding=embeddings,
            persist_directory=DB_PATH,
        )

    return llm, store


llm, store = load_rag()

# Rewrite follow-up questions using conversation history
rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Rewrite the user's latest question into a standalone search query using the "
        "chat history if needed. Return ONLY the standalone query.",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{query}"),
])
rewrite_chain = rewrite_prompt | llm | StrOutputParser()

# Grounded Alex system prompt with memory placeholder
qa_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are Alex, the official AI customer support assistant for TechMart, an online electronics store.

Ground your answers strictly in the TechMart context provided below:
<context>
{context}
</context>

Guidelines:
- Answer concisely, clearly, and politely to help the customer.
- Only state product names, prices, discounts, categories, order details, or store policies that appear in the context above.
- If the answer is not found in the context, say clearly: "I don't know based on the available TechMart information." Do not guess or invent details.
- When listing multiple products, use short bullet points with regular price, discounted price/savings, and category if available in the context.
- Use prior conversation history only to understand follow-up questions; still verify all factual claims against the context.""",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{query}"),
])
qa_chain = qa_prompt | llm | StrOutputParser()

# 1. Clear session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# 2. Render chat message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 3. Chat input & real-time streaming response
if query := st.chat_input("Ask Alex about TechMart products, prices, or policies..."):
    # Build LangChain history (last 8 messages = 4 turns)
    chat_history = [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in st.session_state.messages[-8:]
    ]

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Rewrite follow-up query if history exists
    search_query = (
        rewrite_chain.invoke({"chat_history": chat_history, "query": query}).strip()
        if chat_history
        else query
    )
    docs = store.similarity_search(search_query, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)

    with st.chat_message("assistant"):
        full_response = st.write_stream(
            qa_chain.stream({
                "chat_history": chat_history,
                "context": context,
                "query": query,
            })
        )

    st.session_state.messages.append({"role": "assistant", "content": full_response})