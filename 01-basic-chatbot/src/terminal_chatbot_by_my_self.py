import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

# Load environment variables (like the OPENAI_API_KEY) from a .env file
load_dotenv()

# --- 1. Define the Prompt Template ---
# This template defines the structure of the input sent to the LLM.
prompt = ChatPromptTemplate.from_messages([
    # System instruction sets the behavior/persona of the assistant.
    ("system", "You are a helpful assistant. Answer the user's question clearly."),
    
    # MessagesPlaceholder is CRITICAL: This is where the RunnableWithMessageHistory 
    # will inject the previous messages (the conversation history).
    MessagesPlaceholder(variable_name='history'),
    
    # The current message from the user.
    ("user", "{question}")
])

# --- 2. Define the History Store ---
# 'store' acts as an in-memory database, mapping session IDs to their chat history objects.
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieves or creates a chat history object for a given session ID."""
    # Check if a history object already exists for this session ID.
    if session_id not in store:
        # If not, create a new history object (messages are stored in RAM).
        store[session_id] = InMemoryChatMessageHistory()
    # Return the existing or newly created history object.
    return store[session_id]


# --- 3. Define the Core Chain (Stateless Pipeline) ---
# Initialize the chat model (the AI engine).
model = ChatOpenAI()

# Initialize the parser to convert the model's output object into a simple string.
output_parser = StrOutputParser()

# Define the core processing pipeline using LCEL (LangChain Expression Language).
# This chain is currently stateless (it doesn't handle memory yet).
chain = prompt | model | output_parser

# --- 4. Wrap the Chain with History Management (Stateful Pipeline) ---
# RunnableWithMessageHistory is the "Super Manager" that handles memory automatically.
with_message_history = RunnableWithMessageHistory(
    # The stateless chain to be wrapped.
    runnable=chain,
    # The function that retrieves the history object.
    get_session_history=get_session_history,
    # Specifies which variable in the input dictionary holds the new user question.
    input_messages_key="question",
    # Specifies which variable in the prompt template is the history placeholder.
    history_messages_key="history",
)

# --- 5. Chat Loop Execution ---
# Define a static session ID for this single terminal instance.
session_id = "terminal_user_123"

while True:
    # Take input from the user.
    user_message = input("You: ")
    
    # Exit condition.
    if user_message.lower() == "exit":
        print("AI: Good Bye!")
        break
    
    # Invoke the history-aware runnable.
    # CRITICAL: The configuration must include the session_id.
    # The wrapper uses this ID to load past context before running the chain,
    # and then saves the current exchange (user message + AI result) afterward.
    result = with_message_history.invoke(
        {"question": user_message},
        # Pass the session ID to link the current turn to the correct history store.
        config={"configurable": {"session_id": session_id}}
    )
    
    # Print the context-aware result.
    print("AI: ", result)