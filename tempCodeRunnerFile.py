prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "Your name is ResumeGPT. Your purpose is to answer questions about Abhishek Bagepalli's resume. You have access to his education, experience, skills and projects."),
#         # MessagesPlaceholder(variable_name="history"),  # Enables chat memory
#         ("human", "{question}"),
#     ]
# )

# llm = OpenAI(temperature=0.2, openai_api_key=os.getenv("OPENAI_API_KEY"))