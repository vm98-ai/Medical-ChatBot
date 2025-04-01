from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

extracted_data = load_pdf('Data/')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"
print(f"Index Name: {index_name}")
re = PineconeVectorStore(
    embedding=embeddings,
    index_name=index_name  
)
docsearch = re.from_texts([t.page_content for t in text_chunks], embedding=embeddings, index_name=index_name)

