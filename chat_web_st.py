import os
import streamlit as st
from datetime import datetime
from chatbot import DocChatbot

# Constants
VECTORDB_PATH = "./data/vector_store"
UPLOAD_FOLDER = "./data/uploaded/"
SUPPORTED_FILE_TYPES = [".pdf", ".md", ".txt", ".docx", ".csv", ".xml"]
EXISTING_VECTOR_OPTION = "-- Existing Vector Stores --"

# Initialize chatbot
docChatBot = DocChatbot()

def get_available_indexes():
    """Fetch the list of available vector stores"""
    available_indexes = docChatBot.get_available_indexes(VECTORDB_PATH)
    return [EXISTING_VECTOR_OPTION] + available_indexes

def save_uploaded_file(uploaded_file, timestamp):
    """Save uploaded file locally and return the saved file path"""
    ext_name = os.path.splitext(uploaded_file.name)[-1]
    local_file_name = f"{UPLOAD_FOLDER}{timestamp}{ext_name}"
    with open(local_file_name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return local_file_name

def process_selected_file(selected_index, uploaded_file):
    """Process the selected vector store or uploaded file"""
    if selected_index == EXISTING_VECTOR_OPTION and uploaded_file:
        ext_name = os.path.splitext(uploaded_file.name)[-1]
        if ext_name not in SUPPORTED_FILE_TYPES:
            st.error("Unsupported file type.")
            st.stop()
        
        # Save the uploaded file and process it
        timestamp = int(datetime.timestamp(datetime.now()))
        local_file_name = save_uploaded_file(uploaded_file, timestamp)

        docChatBot.init_vector_db_from_documents([local_file_name])
        docChatBot.save_vector_db_to_local(VECTORDB_PATH, f"{timestamp}{ext_name}")
    else:
        docChatBot.load_vector_db_from_local(VECTORDB_PATH, selected_index)

    # Update session state
    st.session_state['docChatBot'] = docChatBot
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi!😊"}]

def display_chat_history():
    """Display chat history in Streamlit chat interface"""
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

def get_llm_response(user_input):
    """Fetch the LLM response and references for the user input"""
    docChatBot = st.session_state['docChatBot']
    docChatBot.init_streaming(st.empty(), st.empty())
    docChatBot.init_chatchain()

    result_answer, result_source = docChatBot.get_answer(user_input, st.session_state.messages)
    return result_answer, result_source

def display_references(result_source):
    """Display the reference documents in the chat"""
    with st.expander("References"):
        for i, doc in enumerate(result_source):
            source_str = os.path.basename(doc.metadata.get("source", ""))
            page_str = f"P{doc.metadata.get('page', '') + 1}" if "page" in doc.metadata else ""
            st.write(f"### Reference [{i + 1}] {source_str} {page_str}")
            st.write(doc.page_content)

# Sidebar
with st.sidebar:
    st.title("💬 Open AI Embeddings")

    with st.form("Upload and Process"):
        selected_index = st.selectbox('Select an existing vector store or upload a file:', get_available_indexes())
        uploaded_file = st.file_uploader("Upload documents", type=SUPPORTED_FILE_TYPES)
        submitted = st.form_submit_button("Process")

        if submitted:
            try:
                process_selected_file(selected_index, uploaded_file)
                st.success("Vector db initialized.")
                st.balloons()
            except Exception as e:
                st.error(f"An error occurred while processing the file: {str(e)}")

# Chat
if 'messages' in st.session_state:
    display_chat_history()

if user_input := st.chat_input():
    if 'docChatBot' not in st.session_state:
        st.error("Please upload a document and click the 'Process' button.")
        st.stop()

    # Process user input and fetch response
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Show a spinner while waiting for response
    with st.spinner("Processing..."):
        result_answer, result_source = get_llm_response(user_input)
    
    st.chat_message("assistant").markdown(result_answer)

    # Display references
    display_references(result_source)

    # Save the assistant's message to session state
    st.session_state.messages.append({"role": "assistant", "content": result_answer})
