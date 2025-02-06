import json
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
import os

# Load your resume data
def load_resume(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

# Initialize the LangChain setup
def setup_chain(resume_data):
    # Flatten the JSON into text for embedding
    docs = flatten_resume(resume_data)

    # Create embeddings and a vector store
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vector_store = FAISS.from_texts(docs, embeddings)

    # Define a LangChain QA chain
    retriever = vector_store.as_retriever()
    llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return qa_chain

def flatten_resume(resume_data):
    docs = []
    for section, content in resume_data.items():
        if isinstance(content, dict):
            # If content is a dictionary, format it as key-value pairs
            content = "\n".join([f"{key}: {value}" for key, value in content.items()])
        elif isinstance(content, list):
            # If content is a list, handle each item appropriately
            flattened_list = []
            for item in content:
                if isinstance(item, dict):
                    # Flatten dictionaries within lists
                    flattened_list.append(
                        "\n".join([f"{key}: {value}" for key, value in item.items()])
                    )
                elif isinstance(item, list):
                    # Convert nested lists to strings
                    flattened_list.append("\n".join(map(str, item)))
                else:
                    # Add string items directly
                    flattened_list.append(str(item))
            content = "\n".join(flattened_list)
        else:
            # Convert other content to strings directly
            content = str(content)
        # Append the section name and content as a single document
        docs.append(f"{section}:\n{content}")
    return docs

# Streamlit app
st.title("Resume Chatbot")
st.write("Ask me anything about my skills, experience, or projects!")

# Load the resume data and set up the chain
resume_data = load_resume('resume.json')
qa_chain = setup_chain(resume_data)

# Handle user queries
context = "Your name is ResumeGPT. You exist to help the user understand the contents of a resume. Please answer the user's questions based on the resume data. You many recommend the candidate to be a good fit for a job, or provide additional information about their skills and experience."
query = st.text_input("Enter your question:")
if query:
    response = qa_chain.run(query)
    st.write(response)
