from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings

DB_PATH = "vectorstores/db_faiss"

def load_db():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

def create_prompt():
    template = """Bạn là trợ lý AI.

Dựa vào context sau để trả lời.
Nếu không biết thì nói không biết.

Context:
{context}

Câu hỏi:
{question}

Trả lời:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def main():
    db = load_db()

    llm = ChatOllama(
        model="llama3",
        temperature=0.1
    )

    prompt = create_prompt()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt}
    )

    while True:
        query = input(">> ")
        if query == "exit":
            break

        result = qa.invoke({"query": query})
        print(result["result"])

if __name__ == "__main__":
    main()