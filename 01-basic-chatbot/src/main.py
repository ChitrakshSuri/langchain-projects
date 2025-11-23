from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Model
model = ChatOpenAI()

# Prompt with MessagesPlaceholder for chat history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and friendly AI assistant. Keep your replies clear, conversational, and easy to understand."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_message}")
])



# Output parser
parser = StrOutputParser()

# Chain using LCEL
chain = prompt | model | parser

# Chat loop
while True:
    user_message = input("You: ")
    if user_message.lower() == "exit":
        print("Goodbye!")
        break
    
    # Invoke chain
    result = chain.invoke({
        "user_message": user_message
    })
    
    
    print("Assistant:", result)