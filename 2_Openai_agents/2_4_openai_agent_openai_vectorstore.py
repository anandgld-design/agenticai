# This will use OpenAI's vector store
# Nothing will get created on our local machine
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
client = OpenAI()

vector_store = client.vector_stores.create(name="faqs_store")
print(f"Vector store created: {vector_store.id}")

faq_file = client.files.create(
    file=open("c:\\code\\agenticai\\2_openai_agents\\faq.txt", "rb"),
    purpose="assistants"
)

client.vector_stores.files.create(
    vector_store_id=vector_store.id,
    file_id=faq_file.id
)

response = client.responses.create(
    model="gpt-4o-mini",
    input="Tell me about the warranty policy."
)

print("\nBot Answer:", response.output_text)
