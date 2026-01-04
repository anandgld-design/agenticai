from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
import gradio as gr
import os
import boto3
from botocore.config import Config
import requests

load_dotenv(override=True)

client = OpenAI()

# --- Step 1: Download and read document PDF from S3 into a single string ---
url = os.getenv("S3_URL")
try:
    response = requests.get(url)
    response.raise_for_status()
    with open('temp.pdf', 'wb') as f:
        f.write(response.content)
except Exception as e:
    print(f"Error downloading file from S3 URL: {e}")
    print("Please check the S3_URL in .env.")
    exit(1)

reader = PdfReader("temp.pdf")
document = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        document += text

# --- Step 2: Define chat function ---
def chat_with_document(message, history):
    """
    Chat with document knowledge base using OpenAI Responses API.
    message: user input
    history: previous chat messages (not needed for stateless, but Gradio passes it)
    """
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            # Pass only the first 4k of document words, ~ 1000 tokens
            # Avoids overloading: Earlier models supported just 4k tokens, GPT-4o-mini supports ~128k
            {"role": "system", "content": f"You are a helpful assistant. You can answer based only on the following text:\n\n{document[:4000]}"},
            {"role": "user", "content": message},
        ]
    )
    return response.output_text

# --- Step 3: Build Gradio UI ---

# Blocks is a layout system in Gradio, more advanced than 
#  gradio.Interface
# With creates a context manager, so that all components defined next
#  will get added to this demo app
with gr.Blocks() as demo: # start defining the UI
    gr.Markdown("# Ask about the Document") # means big header
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask a question about the document...")
    clear = gr.Button("Clear")
    
    def respond(user_message, chat_history):
        answer = chat_with_document(user_message, chat_history)
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": answer})
        return "", chat_history

    # Syntax: msg.submit(function, inputs, outputs)
    # When the user clicks ENTER, pass inputs [msg, chatbot] to the chat function
    # Then take the function's output and add it to the chat history [msg, chatbot]
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    
    # When the user clicks the "Clear" button, clear the chat history
    # Run the function lambda: None, which means do nothing
    # Function takes no inputs (None) and sends its output (None) to the chatbot
    # This resets the chat history
    # queue=False means run the function immediately
    clear.click(lambda: None, None, chatbot, queue=False)
    
demo.launch()
