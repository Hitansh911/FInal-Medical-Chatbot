import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub

# Load environment variables from .env
load_dotenv()

# --- Configuration ---
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # or your preferred model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="MediBot - Your AI Health Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Enhanced Styling ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
<style>
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #b2ebf2, #4dd0e1, #26c6da);
        color: #ffffff;
        transition: background 0.5s ease-in-out;
    }

    h1 {
        text-align: center;
        font-weight: 600;
        color: #ffffff;
        border-bottom: 2px solid #ffffff;
        margin-bottom: 20px;
        padding-bottom: 10px;
    }

    /* Chat bubbles with fade-in animation */
    .stChatMessage {
        border-radius: 16px;
        padding: 14px 18px;
        margin-bottom: 12px;
        animation: fadeInUp 0.6s ease-in-out;
        transition: all 0.3s ease-in-out;
        font-size: 15px;
    }

    div[data-testid="stChatMessage"][style*="flex-direction: row-reverse;"] .stChatMessage {
        background-color: #004d40;
        color: #ffffff;
        border: 1px solid #00796b;
    }

    div[data-testid="stChatMessage"]:not([style*="flex-direction: row-reverse;"]) .stChatMessage {
        background-color: #ffffff;
        color: #004d40;
        border: 1px solid #26a69a;
    }

    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 1.5px solid #ffffff;
        background-color: #e0f7fa;
        color: #004d40;
        padding: 10px;
        transition: all 0.3s ease-in-out;
    }

    .stTextInput > div > div > input:focus {
        border-color: #00bcd4;
        box-shadow: 0 0 10px rgba(0, 188, 212, 0.4);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #004d40;
        color: #e0f7fa;
        border-right: 2px solid #00796b;
    }

    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #80deea;
    }

    [data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
        color: #b2dfdb;
    }

    [data-testid="stSidebar"] .stButton>button {
        background-color: #00bcd4;
        color: #ffffff;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
    }

    [data-testid="stSidebar"] .stButton>button:hover {
        background-color: #00838f;
        color: #fff;
        transform: scale(1.05);
    }

    /* Expanders */
    .st-expander {
        border: 1px solid #26c6da;
        background-color: #e0f2f1;
        color: #00695c;
    }

    .st-expander-header {
        background-color: #b2ebf2;
        color: #004d40;
        font-weight: 600;
    }

    .st-expander-content {
        background-color: #e0f7fa;
        color: #003f5c;
    }

    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
</style>
""", unsafe_allow_html=True)




# --- Caching Functions ---
@st.cache_resource
def get_vectorstore():
    """Loads the FAISS vector store."""
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error loading vector store: {e}. Please ensure '{DB_FAISS_PATH}' exists and is a valid FAISS index.")
        return None


@st.cache_resource
def load_llm(huggingface_repo_id):
    """Loads the HuggingFace LLM."""
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        st.error(
            "HUGGINGFACEHUB_API_TOKEN not found in environment variables. Please set it in your .env file.")
        return None

    try:
        llm = HuggingFaceHub(
            repo_id=huggingface_repo_id,
            huggingfacehub_api_token=token,
            model_kwargs={"temperature": 0.5, "max_length": 512, "repetition_penalty": 1.2}
        )
        return llm
    except Exception as e:
        st.error(f"Error loading LLM from HuggingFace Hub: {e}")
        return None

# --- Core Logic Functions ---
def set_custom_prompt(custom_prompt_template):
    """Creates a PromptTemplate from the given template string."""
    prompt = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def get_qa_chain():
    """Initializes and returns the RetrievalQA chain."""
    vectorstore = get_vectorstore()
    llm = load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID)

    if not vectorstore or not llm:
        st.warning(
            "Could not initialize the QA chain due to missing vector store or LLM.")
        return None

    CUSTOM_PROMPT_TEMPLATE = """
    You are MediBot, a helpful AI assistant. Your goal is to provide informative and safe responses based on the provided context.
    Use the pieces of information provided in the context to answer the user's question.
    If you don't know the answer from the context or if the question is outside the medical scope of the context, clearly state that you don't have enough information.
    Do not make up answers or provide information beyond the given context.
    Always prioritize user safety. If a query seems to indicate a serious medical condition, gently suggest consulting a healthcare professional.
    Keep your answers concise and easy to understand.

    Context: {context}
    Question: {question}

    Answer:
    """

    custom_prompt = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': custom_prompt}
    )
    return qa_chain

# --- UI Rendering ---
def display_chat_history():
    """Displays the chat history using Streamlit's chat elements."""
    for message in st.session_state.messages:
        avatar_icon = "🧑‍⚕️" if message['role'] == 'assistant' else "👤"
        with st.chat_message(message['role'], avatar=avatar_icon):
            st.markdown(message['content'])


def display_source_documents(source_documents):
    """Displays source documents in an expander."""
    if source_documents:
        with st.expander("View Sources", expanded=False):
            for i, doc in enumerate(source_documents):
                st.markdown(f"**Source {i + 1}:**")
                if hasattr(doc, 'metadata') and doc.metadata:
                    st.caption(f"Content: {doc.page_content[:200]}...")
                    st.json(doc.metadata, expanded=False)
                else:
                    st.caption(doc.page_content)
                st.markdown("---")


# --- Main Application ---
def main():
    # --- Sidebar ---
    with st.sidebar:
        st.image(
            "https://placehold.co/150x150/87CEEB/FFFFFF?text=MediBot&style=for-the-badge",
            use_container_width=True)  # Light blue logo
        st.markdown("## About MediBot")
        st.markdown(
            "MediBot is an AI-powered assistant designed to provide information based on a knowledge base. "
            "It can help answer your health-related questions."
        )
        st.markdown("---")
        st.markdown("### **Disclaimer**")
        st.warning(
            "MediBot is **not a substitute for professional medical advice**, diagnosis, or treatment. "
            "Always seek the advice of your physician or other qualified health provider with any "
            "questions you may have regarding a medical condition."
        )
        st.markdown("---")
        st.markdown("### How to Use")
        st.markdown(
            """
            1. Type your health-related question in the chat box below.
            2. MediBot will search its knowledge base for relevant information.
            3. Review the answer and the source documents if provided.
            """
        )
        st.markdown("---")
        if st.button("Clear Chat History"):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hello! I'm MediBot. How can I help you with your health questions today?"
                }
            ]
            st.rerun()

    # --- Main Chat Interface ---
    st.title("MediBot - Your AI Health Assistant")

    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm MediBot. How can I help you with your health questions today?"
            }
        ]

    display_chat_history()

    user_query = st.chat_input("Ask MediBot about your health concerns...")

    if user_query:
        st.session_state.messages.append({'role': 'user', 'content': user_query})
        with st.chat_message('user', avatar="👤"):
            st.markdown(user_query)

        qa_chain = get_qa_chain()
        if qa_chain:
            with st.spinner("MediBot is thinking..."):
                try:
                    response = qa_chain.invoke({'query': user_query})
                    result = response["result"]
                    source_documents = response["source_documents"]

                    with st.chat_message('assistant', avatar="🧑‍⚕️"):
                        st.markdown(result)
                        display_source_documents(source_documents)

                    st.session_state.messages.append(
                        {'role': 'assistant', 'content': result})

                except Exception as e:
                    error_message = f"An error occurred: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append(
                        {'role': 'assistant', 'content': "Sorry, I encountered an error processing your request."})
        else:
            error_message_chain = "MediBot is currently unavailable. Please check configurations."
            st.error(error_message_chain)
            st.session_state.messages.append(
                {'role': 'assistant', 'content': "Sorry, I'm unable to process your request right now."})


if __name__ == "__main__":
    main()
