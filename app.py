import streamlit as st
from src.chatbot import build_chatbot

st.set_page_config(
    page_title="RAG Chatbot Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Chatbot Assistant")
st.caption("Ask questions about the documents in the /docs folder.")

@st.cache_resource(show_spinner="Loading documents and building knowledge base...")
def get_chatbot():
    return build_chatbot()

try:
    chatbot = get_chatbot()
except ValueError as e:
    st.error(str(e))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask something about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chatbot.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": "default"}}
            )
            answer = result["answer"]
            sources = result.get("context", [])

        st.markdown(answer)

        if sources:
            with st.expander("📄 Sources"):
                for doc in sources:
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "")
                    st.markdown(f"- `{source}` page {page}" if page != "" else f"- `{source}`")

    st.session_state.messages.append({"role": "assistant", "content": answer})