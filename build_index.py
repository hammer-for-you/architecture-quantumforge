import glob
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
import time

KNOWLEDGE_BASE_DIR = "knowledge_base"
CHUNK_SIZE = 400
OVERLAP = 50
INDEX_DIR = "faiss_index"
MODEL_NAME = "all-MiniLM-L6-v2"

def load_knowledge_base(base_dir):
    """
    Загружаем базу знаний

    Args:
        base_dir: Каталог с базой знаний
    """
    print("Загружаем базу знаний...")
    loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob = "*.txt", show_progress = True)
    docs = loader.load()
    print(f"Загружено {len(docs)} документов\n")

    return docs

def create_chunks(docs):
    """
    Создаём чанки

    Args:
        docs: Загруженные документы, которые будут нарезаны на чанки
    """
    print("Разбиваем документы на чанки...")

    splitter = RecursiveCharacterTextSplitter(chunk_size = CHUNK_SIZE, chunk_overlap = OVERLAP)
    chunks = splitter.split_documents(docs)
    print(f"Получено {len(chunks)} чанков. Размер чанка: {CHUNK_SIZE}, перекрытие: {OVERLAP}")

    print("Добавляем метаданные к чанкам...")
    for index, chunk in enumerate(chunks):
        source = chunk.metadata['source']
        filename = os.path.basename(source)
        chunk.metadata['filename'] = filename
        chunk.metadata['chunk_id'] = f"chunk::{index:08d}"
    print("Метаданные добавлены\n")

    return chunks

def create_index(chunks):
    """
    Создаём индекс

    Args:
        chunks: Чанки, полученные из документов
    """

    print("Создание индекса...")

    start_time = time.perf_counter()

    print("Инициализация языковой модели")
    model = SentenceTransformer(MODEL_NAME)

    print("Создание эмбеддингов")
    embeddings = model.encode(chunks, show_progress_bar = True)

    print("Начато построение индекса")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    index_path = INDEX_DIR
    vectorstore.save_local(INDEX_DIR)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"Построение индекса завершено за {elapsed_time:.6f} секунд")

def main():
    """
    Основная функция скрипта
    """
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        raise FileNotFoundError(f"Каталог {KNOWLEDGE_BASE_DIR} не найден!")
    
    docs = load_knowledge_base(KNOWLEDGE_BASE_DIR)

    chunks = create_chunks(docs)

if __name__ == "__main__":
    main()