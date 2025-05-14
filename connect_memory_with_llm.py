import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st

# Load environment variables (optional)
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Get Hugging Face token from environment variable
HF_TOKEN = os.environ.get("HF_TOKEN")  # Make sure you've set this in your environment
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Step 1: Setup LLM (Mistral with HuggingFace)
def load_llm(huggingface_repo_id):
    try:
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            task="text-generation",
            temperature=0.5,
            huggingfacehub_api_token=HF_TOKEN,
            model_kwargs={"max_length": 512}
        )
        return llm
    except Exception as e:
        print(f"Error loading LLM: {str(e)}")
        return None  # Return None if there's an error


# Step 2: Setup custom prompt template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
Don't provide anything outside of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Step 3: Load FAISS vector database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 4: Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 5: Query input and response
user_query = st.text_input("Write your query here:")
if st.button("Submit"):
    if user_query:
        response = qa_chain.invoke({'query': user_query})
        st.write("RESULT: ", response["result"])
        st.write("SOURCE DOCUMENTS: ", response["source_documents"])
    else:
        st.warning("Please enter a query.")