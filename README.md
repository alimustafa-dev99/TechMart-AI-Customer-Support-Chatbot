# TechMart AI Customer Support Chatbot ("Alex")

An interactive Retrieval-Augmented Generation (RAG) customer support web application built with **Streamlit**, **LangChain**, **ChromaDB**, and **Google Gemini**. It answers customer inquiries grounded in **TechMart's Customer Support Guidebook**, supports multi-turn follow-up questions, and streams responses in real time.

---

## Features
* **Guidebook-Grounded RAG:** Loads and chunks `TechMart Customer Support Guidebook.pdf`, indexes embeddings in Chroma, and retrieves relevant support context for each question.
* **Multi-Turn Follow-Ups:** Uses chat history (`MessagesPlaceholder`) to rewrite follow-up questions into standalone search queries before retrieval.
* **Grounded Responses:** Prompted to stay strictly grounded in retrieved store documentation rather than guessing policies or product details.
* **Streaming Streamlit UI:** Real-time token streaming (`st.write_stream`), session-based chat history, and a clear/reset chat control.

---

## Project Structure

```text
tech-mart-chatbot/
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
└── data/
    └── TechMart Customer Support Guidebook.pdf
