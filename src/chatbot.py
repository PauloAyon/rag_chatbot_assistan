from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from src.config import GROQ_API_KEY, MODEL_NAME
from src.rag_engine import load_documents, build_vector_store, get_retriever

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def build_chatbot():
    """Build and return the full conversational RAG chain."""

    # Load documents
    documents = load_documents()
    if not documents:
        raise ValueError("No documents found in /docs folder. Add at least one PDF or TXT file.")

    # Build vector store and retriever
    vector_store = build_vector_store(documents)
    retriever = get_retriever(vector_store)

    # Initialize LLM
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0.3
    )

    # Prompt to reformulate the question considering chat history
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and the latest user question, "
                   "reformulate a standalone question that can be understood "
                   "without the chat history. Do NOT answer, just reformulate."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    # Prompt to answer based on retrieved context
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user's question "
                   "based on the following context:\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, answer_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Wrap with message history
    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return conversational_chain