import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()

# --- 1. Memory Store (Configuration remains the same) ---
# We use st.session_state to store the history function and the chain itself,
# which is Streamlit's standard way to manage state across reruns.
if "memory_store" not in st.session_state:
    st.session_state.memory_store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieves or creates an InMemoryChatMessageHistory object for a given session ID 
    from Streamlit's persistent session state."""
    
    # Use st.session_state.memory_store instead of the old global 'store'
    if session_id not in st.session_state.memory_store:
        st.session_state.memory_store[session_id] = InMemoryChatMessageHistory()
    
    return st.session_state.memory_store[session_id]

# --- 2. Prompt Template (Configuration remains the same) ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the user's question clearly and keep answers concise."),
    MessagesPlaceholder(variable_name='history'),
    ("user", "{question}")
])

# --- 3. Define the Core Chain (LCEL) ---
model = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()
chain = prompt | model | output_parser

# --- 4. Wrap the Chain with History Management ---
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# --- 5. Streamlit Application Logic ---

st.set_page_config(page_title="Context-Aware Chatbot")
st.title("LangChain Context-Aware Chatbot 🤖")

# --- Initialize Session State for Chat History ---
# Streamlit needs to manage chat history using its st.session_state feature.
# We store the *messages* themselves here for display purposes.
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Past Messages ---
# Loop through the stored messages and display them in the chat interface.
for message in st.session_state.messages:
    # Streamlit's chat_message container handles the display role (user/assistant)
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle New User Input ---
# This block executes when the user submits a message via the chat input box.
if prompt_input := st.chat_input("Ask me anything..."):

    # 1. Display and Save User Message
    with st.chat_message("user"):
        st.markdown(prompt_input)
    # Save the user message to Streamlit's session state list
    st.session_state.messages.append({"role": "user", "content": prompt_input})

    # 2. Invoke the LangChain Runnable
    # Use a fixed session ID for the terminal memory store
    SESSION_ID = "streamlit_session_1"
    st.write(
        f"🔍 Memory has {len(get_session_history(SESSION_ID).messages)} messages")

    # The LangChain invocation uses the same logic as your terminal app
    result = with_message_history.invoke(
        {"question": prompt_input},
        config={"configurable": {"session_id": SESSION_ID}}
    )

    # 3. Display and Save AI Message
    with st.chat_message("assistant"):
        st.markdown(result)
    # Save the AI response to Streamlit's session state list
    st.session_state.messages.append({"role": "assistant", "content": result})
