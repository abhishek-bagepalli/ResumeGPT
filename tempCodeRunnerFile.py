text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # chunks = text_splitter.split_documents(pages)

    # # Create the Chroma vector store (stored in ./chroma_db by default)
    # vector_store = Chroma.from_documents(
    #     documents=chunks,
    #     embedding=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
    #     persist_directory="./chroma_db"
    # )

    # user_input = 'How do you think Abhishek is doing?'

    # docs = vector_store.similarity_search(user_input, k=2)
    # retrieved_content = "\n".join([doc.page_content for doc in docs])

    # print(retrieved_content)