import streamlit as st
import openai 
from openai import OpenAI
import pandas as pd
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

# Access OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["secrets"]["OPENAI_API_KEY"]
client = OpenAI(api_key= st.secrets["secrets"]["OPENAI_API_KEY"])

# Set LLM and embedding models
Settings.llm = OpenAI(model="gpt-4o")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

# Set page configuration
st.set_page_config(
    page_title="מטרופולינט AI",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)

# Inject custom CSS for RTL support and styling adjustments
st.markdown(
    """
    <style>
    body {
        direction: rtl;
        text-align: right;
        background-color: #FFFAF0;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        margin: 0;
    }
    .stApp {
        max-width: 700px; /* כדי להגביל את רוחב האפליקציה */
    }
    .stAlert, .stButton button, .stTextInput div, .stTextArea div, .stMarkdown,
    .stRadio div, .stCheckbox div, .stSelectbox div, .stMultiselect div, 
    .stDateInput div, .stTimeInput div, .stFileUploader div, .stMetric {
        direction: rtl;
        text-align: right;
    }
    h1 {
        color: #FF4500; /* צבע כתום */
        font-size: 18px;
    }
    .main > div {
        padding-top: 0 !important;
        height: 70vh; /* גובה מתאים כדי למקם באמצע הדף */
        display: flex;
        flex-direction: column;
        justify-content: center; /* ממרכז את הקונטיינר באמצע הדף */
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        height: 50vh;
        justify-content: flex-start;
        margin-bottom: 10px;
    }
    .chat-input {
        background: white;
        padding: 5px;
    }
    .user-message {
        text-align: right;
        color:black;
        background-color: #FFA07A; /* צבע כתום בהיר */
        padding: 5px;
        border-radius: 10px;
        margin-bottom: 5px;
    }
    .assistant-message {
        text-align: left;
        color:black;
        background-color: #FF6347; /* צבע אדום */
        padding: 5px;
        border-radius: 10px;
        margin-bottom: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("מטרופולינט AI 🤖🗨️")
st.info("שאל אותי הכל!")

# Initialize chat messages history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "לדוגמא: שאלות על תברים",
        }
    ]

# Load the knowledge center CSV
@st.cache_data(show_spinner=False)
def load_data():
    docs = []
    knowledge_center = pd.read_csv("AI_chat/data/knowledge_center.csv")
    for _, row in knowledge_center.iterrows():
        combined_text = f"{row['semanticsearch']} {row['TAGS']}"
        docs.append(Document(
            text=combined_text,
            doc_id=row['id'],
        ))
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir="./data")
    print("Index built and persisted to storage")
    return index

index = load_data()

# Initialize the chat engine
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True
    )

# Function to perform semantic search in the knowledge file
def check_knowledge_center(question):
    st.session_state.chat_engine.reset() 
    response = st.session_state.chat_engine.chat(question)
    print(response.response)
    return response.response

# Function to generate a response from OpenAI
def generate_response(prompt):
    try:
        response = client.chat.completions.create(model="gpt-4o",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ])
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Display the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Prompt for user input and save to chat history
prompt = st.chat_input("השאלה שלך...", key="chat-input")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
        
    # Generate a new response
    with st.chat_message("assistant"):
        with st.spinner("חושב..."):
            answer = check_knowledge_center(prompt)
        if answer:
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            response = generate_response(prompt)
            if response:
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
