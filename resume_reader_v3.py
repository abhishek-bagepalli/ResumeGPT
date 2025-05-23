from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
import json
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
# from streamlit_gsheets import GSheetsConnection
# import speech_recognition as sr

def doc_loader(file_path):
    try:
        loader = PyPDFLoader(file_path)
        pages = []
        for page in loader.lazy_load():
            pages.append(page)
        return pages
    except Exception as e:
        st.error(f"Error loading PDF: {str(e)}")
        return []

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = StreamlitChatMessageHistory()
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

def main():
    st.set_page_config(
        page_title="ResumeGPT",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("ResumeGPT")
    st.write("Ask me anything about the resume!")

    initialize_session_state()

    try:
        # Initialize Google credentials
        google_creds = st.secrets["google_creds"]
        creds = Credentials.from_service_account_info(
            google_creds, 
            scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        )
        gc = gspread.authorize(creds)
    except Exception as e:
        st.error(f"Error initializing Google credentials: {str(e)}")
        return

    try:
        # Load and process the PDF
        pages = doc_loader('resume.pdf')
        if not pages:
            st.error("No pages found in the PDF")
            return

        # Chunk the documents
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(pages)

        # Create vector store if not already created
        if st.session_state.vector_store is None:
            with st.spinner("Creating vector store..."):
                st.session_state.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"]),
                    persist_directory="./chroma_db"
                )

        # Initialize chat history
        if len(st.session_state.messages.messages) == 0:
            st.session_state.messages.add_ai_message("Hello! I'm ResumeGPT. Ask me anything about the resume!")

        # Display chat history
        for msg in st.session_state.messages.messages:
            st.chat_message(msg.type).write(msg.content)

        # Chat input
        if user_input := st.chat_input("Ask a question about the resume"):
            st.chat_message("human").write(user_input)

            # Get relevant documents
            docs = st.session_state.vector_store.similarity_search(user_input, k=2)
            retrieved_content = "\n".join([doc.page_content for doc in docs])

            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Your name is ResumeGPT. Your purpose is to respond to the question. If applicable, answer questions about the resume. Here is the relevant content: {resume}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ])

            # Initialize LLM
            llm = ChatOpenAI(temperature=0.2, openai_api_key=st.secrets["OPENAI_API_KEY"])
            qa_chain = prompt | llm

            # Create chain with history
            chain_with_history = RunnableWithMessageHistory(
                qa_chain,
                lambda session_id: st.session_state.messages,
                input_messages_key="question",
                history_messages_key="history",
            )

            # Get response
            with st.spinner("Thinking..."):
                config = {"configurable": {"session_id": "any"}}
                response = chain_with_history.invoke(
                    {"question": user_input, "resume": retrieved_content}, 
                    config
                )

            st.chat_message("ai").write(response.content)

            # Log to Google Sheets
            try:
                wks = gc.open("ResumeGPT").sheet1
                wks.append_rows([[user_input, response.content]])
            except Exception as e:
                st.warning(f"Could not log to Google Sheets: {str(e)}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
    # print(result)
    # sys.stdout.flush()
