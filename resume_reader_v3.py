from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai  import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
import sys
import json
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

def doc_loader(file_path):
    loader = PyPDFLoader(file_path)
    pages = []
    for page in loader.lazy_load():
        pages.append(page)
    return pages

def main():

    # gc = gspread.service_account("temp1-321221-dabd56a34836.json")

    # Use st.secrets to get the service account info
    credentials = Credentials.from_service_account_info(st.secrets["connections"]["gsheets"])
    gc = gspread.authorize(credentials)

    # conn = st.connection("gsheets",type=GSheetsConnection)

    pages = doc_loader('resume.pdf')

    msgs = StreamlitChatMessageHistory()

    # print(f"{pages[0].metadata}\n")
    # print(pages[0].page_content)

    vector_store = InMemoryVectorStore.from_documents(pages, OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")))

    docs = vector_store.similarity_search("Deloitte", k=2)
    # for doc in docs:
    #     print(doc)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Your name is ResumeGPT. Your purpose is to respond to the question. If applicable, answer questions about Abhishek Bagepalli's resume. Here is his resume. {resume}"),
            # ("system", "Your are a helpful chatbot"),
            MessagesPlaceholder(variable_name="history"),  # Enables chat memory
            ("human", "{question}"),
        ]
    )

    llm = ChatOpenAI(temperature=0.2, openai_api_key=os.getenv("OPENAI_API_KEY"))

    qa_chain = prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        qa_chain,
        lambda session_id: msgs,  # Always return the instance created earlier
        input_messages_key="question",
        history_messages_key="history",
    )

    # Streamlit app
    st.title("ResumeGPT")
    st.write("Ask me anything about Abhishek's skills, experience, or projects!")
    if len(msgs.messages) == 0:
        msgs.add_ai_message("Hello!")

    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    # user_input = "Where has Abhishek worked?"
    if user_input := st.chat_input():
        st.chat_message("human").write(user_input)

        # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
        config = {"configurable": {"session_id": "any"}}
        response2 = chain_with_history.invoke({"question":user_input,"resume":docs[0].page_content}, config)
        # print(user_input)
        # print(response2)

        st.chat_message("ai").write(response2.content)

        wks = gc.open("ResumeGPT").sheet1

        # wks.update([[user_input,response2.content]])

        wks.append_rows([[user_input,response2.content]])
        # existing_data = conn.read()

        # df = pd.DataFrame(data=[{"user_input":user_input,"response":response2.content}])

        # print(existing_data)

        # updated_data = pd.concat([existing_data,df],ignore_index=True)

        # print(updated_data)      

        # conn.update(data=updated_data)

        # print(existing_data.columns)
        
        # existing_data['user_input'].append(user_input)
        # existing_data['response'].append(response2.content)
        #display AI response
        return json.dumps({"response":response2.content,"user_input":user_input})
    
if __name__ == "__main__":
    main()
    # print(result)
    # sys.stdout.flush()
