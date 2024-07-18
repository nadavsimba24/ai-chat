import streamlit as st
from openai import OpenAI
import pandas as pd
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# Set LLM and embedding models
Settings.llm = OpenAI(model="gpt-4o")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

# Set page configuration
st.set_page_config(
    page_title="AI Chat",
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
        background-color: white;
    }
    .stAlert, .stButton button, .stTextInput div, .stTextArea div, .stMarkdown,
    .stRadio div, .stCheckbox div, .stSelectbox div, .stMultiselect div, 
    .stDateInput div, .stTimeInput div, .stFileUploader div, .stMetric {
        direction: rtl;
        text-align: right;
    }
    h1 {
        font-size: 12px;
    }
    .main > div {
        padding-top: 0 !important;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        height: 30vh; /* Adjust this value as needed */
        justify-content: space-between;
    }
    .chat-input {
        position: sticky;
        bottom: 0;
        background: white;
        padding: 10px;
    }
    .user-message {
        text-align: right;
        color:black;
        background-color: #DCF8C6;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .assistant-message {
        text-align: left;
        color:black;
        background-color: #F1F0F0;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .message-container {
        display: flex;
        align-items: flex-end;
        margin-bottom: 10px;
    }
    .message-container img {
        border-radius: 50%;
        margin: 0 10px;
    }
    .message-container.user {
        justify-content: flex-end;
    }
    .message-container.assistant {
        justify-content: flex-start;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("OpenAI and Streamlit Integration")
st.info("שאל אותי הכל!")

# Initialize chat messages history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "לדוגמא: שאלות על השכונה שלי ",
        }
    ]

# Load the knowledge center CSV
@st.cache_data(show_spinner=False)
def load_data():
    docs = []
    knowledge_center = pd.read_csv("data/knowledge_center.csv")
    for _, row in knowledge_center.iterrows():
        docs.append(Document(
            text=row['semanticsearch'],
            doc_id=row['id'],
        ))
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir="./data")
    print("Index built and persisted to storage")
    return index

index = load_data()

# st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# Initialize the chat engine
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True
    )

# Function to perform semantic search in the knowledge file
def check_knowledge_center(question):
    response = st.session_state.chat_engine.chat(question)
    print("check_knowledge_center")
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
        print("generate_response")
        print(response.choices[0].message.content.strip())
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Display the chat history
# for message in st.session_state.messages:

    # if message["role"] == "user":
    #     st.markdown(f"""
    #         <div class="message-container user">
    #             <div class="user-message">
    #                 {message['content']}
    #             </div>
    #               <img src="https://via.placeholder.com/40?text=U" alt="User" width="40" height="40">
    #         </div>
    #     """, unsafe_allow_html=True)
    # else:
    #     st.markdown(f"""
    #         <div class="message-container assistant">
    #             <img src="https://via.placeholder.com/40?text=R" alt="Assistant" width="40" height="40">
    #             <div class="assistant-message">
    #                 {message['content']}
    #             </div>
    #         </div>
    #     """, unsafe_allow_html=True)
    
# Display the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
# Create a form for input and submission
# with st.form(key='my_form', clear_on_submit=True):

#     prompt = st.text_input("Hidden Prompt", "",placeholder="השאלה שלך ..", label_visibility='collapsed',key="chat-input")
#     submit_button = st.form_submit_button(label='')

# if submit_button and prompt:
#     # Add user's message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     # Show spinner while processing
#     with st.spinner("Thinking..."):
#         # Check for the answer in the knowledge center
#         answer = check_knowledge_center(prompt)
#         if answer:
#             # Add the found answer to the chat history
#             st.session_state.messages.append({"role": "assistant", "content": answer})
#         else:
#             # Generate OpenAI's response
#             response = generate_response(prompt)
#             if response:
#                 # Add OpenAI's response to chat history
#                 st.session_state.messages.append({"role": "assistant", "content": response})

#     # Rerun to display updated messages
#     st.rerun()




# Prompt for user input and save to chat history
prompt = st.chat_input("השאלה שלך...", key="chat-input")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
        
    # Generate a new response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = check_knowledge_center(prompt)
        if answer:
            # Add the found answer to the chat history
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            # Generate OpenAI's response
            response = generate_response(prompt)
            if response:
                # Add OpenAI's response to chat history
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            

st.markdown("</div>", unsafe_allow_html=True)