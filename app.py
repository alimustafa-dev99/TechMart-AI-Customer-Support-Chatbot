import os
import warnings

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()


def get_google_api_key() -> str | None:
    """Read GOOGLE_API_KEY or GEMINI_API_KEY from Streamlit Secrets or env."""
    for key_name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        try:
            if key_name in st.secrets and st.secrets[key_name]:
                key = str(st.secrets[key_name]).strip()
                if key:
                    os.environ["GOOGLE_API_KEY"] = key
                    os.environ["GEMINI_API_KEY"] = key
                    return key
        except Exception:
            pass

    env_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if env_key and env_key.strip():
        key = env_key.strip()
        os.environ["GOOGLE_API_KEY"] = key
        os.environ["GEMINI_API_KEY"] = key
        return key

    return None


PDF_PATH = "data/TechMart Customer Support Guidebook.pdf"
DB_PATH = "/tmp/techmart_store_db"

st.set_page_config(page_title="TechMart Support - Alex", page_icon="🛒")
st.title("🛒 TechMart Support (Alex)")


@st.cache_resource(show_spinner="Loading TechMart knowledge base...")
def load_rag():
    api_key = get_google_api_key()
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing. Add GOOGLE_API_KEY to Streamlit Secrets or your .env file."
        )

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF guidebook not found at: {PDF_PATH}")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0.2,
        api_key=api_key,
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        api_key=api_key,
    )

    os.makedirs(DB_PATH, exist_ok=True)

    # Load existing Chroma DB if already built; otherwise index the PDF once
    if os.listdir(DB_PATH):
        store = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings,
            collection_name="techmart_guidebook",
        )
    else:
        loader = PyPDFLoader(file_path=PDF_PATH)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splitted_docs = splitter.split_documents(docs)
        store = Chroma.from_documents(
            documents=splitted_docs,
            embedding=embeddings,
            persist_directory=DB_PATH,
            collection_name="techmart_guidebook",
        )

    return llm, store


try:
    llm, store = load_rag()
except Exception as exc:
    st.error(f"Startup failed: {exc}")
    st.stop()

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

# 1. Session state initialization
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
    clean_query = query.strip()
    if not clean_query:
        st.stop()

    # Build LangChain history (last 8 messages = 4 turns)
    chat_history = [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in st.session_state.messages[-8:]
    ]

    st.session_state.messages.append({"role": "user", "content": clean_query})
    with st.chat_message("user"):
        st.markdown(clean_query)

    # Rewrite follow-up query if history exists, falling back to clean_query if blank
    search_query = clean_query
    if chat_history:
        try:
            rewritten = rewrite_chain.invoke({"chat_history": chat_history, "query": clean_query}).strip()
            if rewritten:
                search_query = rewritten
        except Exception:
            search_query = clean_query

    try:
        docs = store.similarity_search(search_query, k=3)
        context = "\n\n".join(doc.page_content for doc in docs)

        with st.chat_message("assistant"):
            full_response = st.write_stream(
                qa_chain.stream({
                    "chat_history": chat_history,
                    "context": context,
                    "query": clean_query,
                })
            )

        response_text = full_response if isinstance(full_response, str) else str(full_response or "")
        st.session_state.messages.append({"role": "assistant", "content": response_text})
    except Exception as exc:
        st.error(f"Could not generate a response: {exc}")
