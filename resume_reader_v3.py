from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai  import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st

def doc_loader(file_path):
    loader = PyPDFLoader(file_path)
    pages = []
    for page in loader.lazy_load():
        pages.append(page)
    return pages



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
st.write("")
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

    # print(response2)


    #display AI response
    st.chat_message("ai").write(response2.content)
