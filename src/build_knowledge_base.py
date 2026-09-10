import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

knowledge_base_path = os.path.join(
    os.path.dirname(__file__), "..", "data", "knowledge_base"
)

loader = DirectoryLoader(
    knowledge_base_path,
    glob="*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)

documents = loader.load()
print(f"📄 {len(documents)} documentos carregados")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)
print(f"✂️  {len(chunks)} chunks gerados")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

persist_directory = os.path.join(
    os.path.dirname(__file__), "..", "chroma_db"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)

print(f"✅ Banco vetorial criado em: {persist_directory}")
print(f"✅ {len(chunks)} chunks indexados com sucesso")