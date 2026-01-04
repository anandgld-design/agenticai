import os
import boto3
import streamlit as st
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

# Initialize AWS S3 client with explicit session
session = boto3.Session(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
)
s3_client = session.client('s3')

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def read_file_from_s3(bucket_name: str, file_key: str) -> str:
    """Read file content from AWS S3"""
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        file_content = response['Body'].read()
        
        # Check if it's a PDF file
        if file_key.lower().endswith('.pdf'):
            # Extract text from PDF
            pdf_file = BytesIO(file_content)
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        else:
            # For other files, try to decode as UTF-8 text
            return file_content.decode('utf-8')
    except Exception as e:
        st.error(f"Error reading from S3: {str(e)}")
        return ""

def get_openai_response(messages: list) -> str:
    """Get response from OpenAI API"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    # Page configuration
    st.set_page_config(
        page_title="AWS S3 OpenAI Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for beautiful UI
    st.markdown("""
    <style>
        .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .stChatMessage { border-radius: 10px; padding: 15px; }
        .stButton>button { 
            background-color: #667eea; 
            color: white; 
            border-radius: 8px;
            font-weight: bold;
        }
        h1 { color: white; text-align: center; }
        .sidebar .sidebar-content { background-color: #f0f2f6; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🤖 AWS S3 OpenAI Chatbot")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        bucket_name = st.text_input("S3 Bucket Name", value=os.getenv('S3_BUCKET', 'amzn-s4-anand'), placeholder="your-bucket-name")
        file_key = st.text_input("S3 File Path", value=os.getenv('S3_KEY', 'Rechargeable.pdf'), placeholder="documents/file.txt")
        system_prompt = st.text_area("System Prompt", value="You are a helpful assistant.")
        
        if st.button("Load Document from S3"):
            if bucket_name and file_key:
                with st.spinner("Loading..."):
                    st.session_state.s3_content = read_file_from_s3(bucket_name, file_key)
                    st.success("Document loaded successfully!")
            else:
                st.warning("Please enter bucket name and file path")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "s3_content" not in st.session_state:
        st.session_state.s3_content = ""
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Prepare messages for OpenAI
        messages = [{"role": "system", "content": system_prompt}]
        
        if st.session_state.s3_content:
            messages.append({
                "role": "system",
                "content": f"Document content:\n{st.session_state.s3_content}"
            })
        
        messages.extend(st.session_state.messages)
        
        # Get OpenAI response
        with st.spinner("Thinking..."):
            response = get_openai_response(messages)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()