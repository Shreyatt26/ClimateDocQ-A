from langchaincommunity.document_loaders import PyPDFDirectoryLoader
from langchaincommunity.text_splitters import RecursiveCharacterTextSplitter
from langchaincommunity.embeddings import OpenAIEmbeddings
from langchaincommunity.vectorstores import Chroma

## Load all PDFs from docs
loader = PyPDFDirectoryLoader("docs/")
documents = loader.load()

## Chunk documents (overlap prevents context from being cut mid-sentence)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

## Create embeddings and store in ChromaDB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="vectorstore")
print(f"Ingested {len(chunks)} chunks into vector store.")

