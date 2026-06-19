## Runtime RAG chain

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


## Load the vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory="vectorstore", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) # top 4 relevant chunks

## Custom prompt to provide more context to the LLM - forces model to stay grounded in retrieved chunks
prompt = PromptTemplate.from_template("""
    You are an assistant for water quality and climate questions.
    Use only the following retrieved information to answer the question. 
    If you don't know the answer, say you don't know. 
    Always use all available information to provide the best answer possible.

    Context: {context}

    Question: {question}
    Answer:
    """
)

# Temperature controls the randomnes of the model's output. 
# Lower values make the output more deterministic, while higher values make it more random.
llm = ChatOllama(model="llama3.2", temperature=0)

chain = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=retriever, 
    return_source_documents=True, 
    chain_type = "stuff", # 'stuff' means all retrieved chunks are stuffed into one prompt.
    chain_type_kwargs={"prompt": prompt}
)

## Interactive loop to ask questions
while True:
    question = input("\nAsk a question about water quality and climate (or 'exit' to quit): ")
    if question.lower() == "exit":
        break
    result = chain.invoke({"query": question})
    print(f"\nAnswer: {result['result']}")
    print(f"\nSources used:")
    for doc in result["source_documents"]:
        print(f" - {doc.metadata.get('source', 'unknown')} (page {doc.metadata.get('page', '?')})")
