from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def quick_search(query, index_path="faiss_index", k=5):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    vectorstore = FAISS.load_local(
        folder_path=index_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    
    results = vectorstore.similarity_search(query, k=k)
    
    print(f"Результаты для запроса: '{query}'\n")
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.page_content[:150]}...")
        print(f"   Метаданные: {doc.metadata}\n")
    
    return results

if __name__ == "__main__":
    # Простой поиск
    results = quick_search("Cursed Sword", k=3)
    
    # Или несколько запросов
    queries = ["Cedric Shadowbane", "Baron Brutus", "Arcane Scepter"]
    for query in queries:
        quick_search(query, k=2)