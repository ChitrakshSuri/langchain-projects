from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the user's question clearly."),
    ("user", "Question: {question}")
])

# Streamlit UI
st.title("Chatbot")

user_input = st.text_input("Ask something:")

# OpenAI model
llm = ChatOpenAI()
parser = StrOutputParser()
chain = prompt | llm | parser

# Run only when user gives input
if user_input:
    with st.spinner("Thinking..."):
        response = chain.invoke({"question": user_input})
    st.write(response)
