import os
import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Gemini embeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_milvus import Milvus
from langchain.schema import Document
from dotenv import load_dotenv
from uuid import uuid4
from crawl import crawl_web

load_dotenv()

def load_data_from_local(filename: str, directory: str) -> tuple:
    """
    Hàm đọc dữ liệu từ file JSON local
    Args:
        filename (str): Tên file JSON cần đọc (ví dụ: 'data.json')
        directory (str): Thư mục chứa file (ví dụ: 'data_v3')
    Returns:
        tuple: Trả về (data, doc_name) trong đó:
            - data: Dữ liệu JSON đã được parse
            - doc_name: Tên tài liệu đã được xử lý (bỏ đuôi .json và thay '_' bằng khoảng trắng)
    """
    file_path = os.path.join(directory, filename)
    
    # Kiểm tra file có tồn tại không
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File không tồn tại: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print(f'✅ Data loaded from {file_path}')
        # Chuyển tên file thành tên tài liệu (bỏ đuôi .json và thay '_' bằng khoảng trắng)
        doc_name = filename.rsplit('.', 1)[0].replace('_', ' ')
        return data, doc_name
    except json.JSONDecodeError as e:
        raise ValueError(f"Lỗi parse JSON từ file {file_path}: {str(e)}")
    except Exception as e:
        raise Exception(f"Lỗi đọc file {file_path}: {str(e)}")

def get_embeddings_model(use_ollama: bool = False):
    """
    Khởi tạo model embeddings dựa trên lựa chọn
    Args:
        use_ollama (bool): True để sử dụng Ollama, False để sử dụng Gemini
    Returns:
        Embeddings model object
    """
    if use_ollama:
        print("🦙 Khởi tạo Ollama embeddings...")
        return OllamaEmbeddings(
            model="llama3.2:3b",  # Hoặc model khác bạn đã cài đặt
            base_url="http://localhost:11434"
        )
    else:
        print("🔍 Khởi tạo Gemini embeddings...")
        # Sử dụng Gemini embeddings
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",  # Model embedding của Gemini
            google_api_key=gemini_api_key
        )

def connect_to_milvus(URI_link: str, collection_name: str, use_ollama: bool = False) -> Milvus:
    """
    Kết nối đến Milvus collection đã có sẵn để query
    Args:
        URI_link (str): Đường dẫn kết nối đến Milvus
        collection_name (str): Tên collection cần kết nối
        use_ollama (bool): Sử dụng Ollama embeddings thay vì Gemini
    Returns:
        Milvus: Object Milvus để query
    """
    try:
        embeddings = get_embeddings_model(use_ollama)
        
        vectorstore = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": URI_link},
            collection_name=collection_name,
            drop_old=False  # Không xóa data đã có
        )
        
        print(f"✅ Đã kết nối với collection '{collection_name}' trong Milvus")
        return vectorstore
    except Exception as e:
        print(f"❌ Lỗi khi kết nối Milvus: {str(e)}")
        raise

def seed_milvus(URI_link: str, collection_name: str, filename: str, directory: str, use_ollama: bool = False) -> Milvus:
    """
    Hàm tạo và lưu vector embeddings vào Milvus từ dữ liệu local
    Args:
        URI_link (str): Đường dẫn kết nối đến Milvus
        collection_name (str): Tên collection trong Milvus để lưu dữ liệu
        filename (str): Tên file JSON chứa dữ liệu nguồn
        directory (str): Thư mục chứa file dữ liệu
        use_ollama (bool): Sử dụng Ollama embeddings thay vì Gemini
    Returns:
        Milvus: Object Milvus đã được seed dữ liệu
    """
    try:
        # Khởi tạo model embeddings
        embeddings = get_embeddings_model(use_ollama)
        
        # Đọc dữ liệu từ file local
        local_data, doc_name = load_data_from_local(filename, directory)

        # Kiểm tra dữ liệu có hợp lệ không
        if not local_data:
            raise ValueError("Dữ liệu từ file JSON trống")

        # Chuyển đổi dữ liệu thành danh sách các Document với giá trị mặc định cho các trường
        documents = []
        for i, doc in enumerate(local_data):
            try:
                # Xử lý trường hợp doc là dict hoặc có cấu trúc khác nhau
                if isinstance(doc, dict):
                    page_content = doc.get('page_content', '') or doc.get('content', '') or str(doc)
                    metadata = doc.get('metadata', {})
                else:
                    page_content = str(doc)
                    metadata = {}
                
                # Tạo Document với metadata đầy đủ
                document = Document(
                    page_content=page_content,
                    metadata={
                        'source': metadata.get('source', f'{filename}#{i}'),
                        'content_type': metadata.get('content_type', 'text/plain'),
                        'title': metadata.get('title', ''),
                        'description': metadata.get('description', ''),
                        'language': metadata.get('language', 'vi'),
                        'doc_name': doc_name,
                        'start_index': metadata.get('start_index', i),
                        'doc_id': i
                    }
                )
                documents.append(document)
            except Exception as e:
                print(f"⚠️ Bỏ qua document {i} do lỗi: {str(e)}")
                continue

        if not documents:
            raise ValueError("Không có document hợp lệ nào được tạo")

        print(f'📚 Loaded {len(documents)} documents from {filename}')

        # Tạo ID duy nhất cho mỗi document
        uuids = [str(uuid4()) for _ in range(len(documents))]

        # Khởi tạo và cấu hình Milvus
        vectorstore = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": URI_link},
            collection_name=collection_name,
            drop_old=True  # Xóa data đã tồn tại trong collection
        )
        
        # Thêm documents vào Milvus theo batch để tránh lỗi
        batch_size = 100
        total_batches = (len(documents) - 1) // batch_size + 1
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_ids = uuids[i:i+batch_size]
            
            try:
                vectorstore.add_documents(documents=batch_docs, ids=batch_ids)
                current_batch = i // batch_size + 1
                print(f'✅ Added batch {current_batch}/{total_batches} ({len(batch_docs)} docs)')
            except Exception as e:
                print(f"❌ Lỗi khi thêm batch {i//batch_size + 1}: {str(e)}")
                # Thử thêm từng document một nếu batch bị lỗi
                for j, (doc, doc_id) in enumerate(zip(batch_docs, batch_ids)):
                    try:
                        vectorstore.add_documents(documents=[doc], ids=[doc_id])
                    except Exception as doc_error:
                        print(f"❌ Bỏ qua document {i+j}: {str(doc_error)}")
        
        print(f'🎉 Successfully seeded {len(documents)} documents to collection: {collection_name}')
        return vectorstore
        
    except Exception as e:
        print(f"❌ Lỗi trong quá trình seed dữ liệu: {str(e)}")
        raise

def seed_milvus_live(URL: str, URI_link: str, collection_name: str, doc_name: str, use_ollama: bool = False) -> Milvus:
    """
    Hàm crawl dữ liệu trực tiếp từ URL và tạo vector embeddings trong Milvus
    Args:
        URL (str): URL của trang web cần crawl dữ liệu
        URI_link (str): Đường dẫn kết nối đến Milvus
        collection_name (str): Tên collection trong Milvus
        doc_name (str): Tên định danh cho tài liệu được crawl
        use_ollama (bool): Sử dụng Ollama embeddings thay vì Gemini
    Returns:
        Milvus: Object Milvus đã được seed dữ liệu
    """
    try:
        # Khởi tạo model embeddings
        embeddings = get_embeddings_model(use_ollama)
        
        print(f'🌐 Starting to crawl data from: {URL}')
        documents = crawl_web(URL)
        
        if not documents:
            raise ValueError(f"Không crawl được dữ liệu từ URL: {URL}")
            
        print(f'📚 Crawled {len(documents)} documents')

        # Cập nhật metadata cho mỗi document với giá trị mặc định
        for i, doc in enumerate(documents):
            metadata = {
                'source': doc.metadata.get('source') or URL,
                'content_type': doc.metadata.get('content_type') or 'text/html',
                'title': doc.metadata.get('title') or f'Page {i+1}',
                'description': doc.metadata.get('description') or '',
                'language': doc.metadata.get('language') or 'en',
                'doc_name': doc_name,
                'start_index': doc.metadata.get('start_index') or i,
                'url': URL,
                'doc_id': i
            }
            doc.metadata = metadata

        # Tạo ID duy nhất cho mỗi document
        uuids = [str(uuid4()) for _ in range(len(documents))]

        # Khởi tạo Milvus
        vectorstore = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": URI_link},
            collection_name=collection_name,
            drop_old=True
        )
        
        # Thêm documents vào Milvus theo batch
        batch_size = 50  # Giảm batch size cho crawled data
        total_batches = (len(documents) - 1) // batch_size + 1
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_ids = uuids[i:i+batch_size]
            
            try:
                vectorstore.add_documents(documents=batch_docs, ids=batch_ids)
                current_batch = i // batch_size + 1
                print(f'✅ Added batch {current_batch}/{total_batches} ({len(batch_docs)} docs)')
            except Exception as e:
                print(f"❌ Lỗi khi thêm batch {i//batch_size + 1}: {str(e)}")
                # Thử thêm từng document một nếu batch bị lỗi
                for j, (doc, doc_id) in enumerate(zip(batch_docs, batch_ids)):
                    try:
                        vectorstore.add_documents(documents=[doc], ids=[doc_id])
                    except Exception as doc_error:
                        print(f"❌ Bỏ qua document {i+j}: {str(doc_error)}")
        
        print(f'🎉 Successfully seeded {len(documents)} documents from {URL} to collection: {collection_name}')
        return vectorstore
        
    except Exception as e:
        print(f"❌ Lỗi trong quá trình crawl và seed dữ liệu: {str(e)}")
        raise

def test_connection(URI_link: str = 'http://localhost:19530'):
    """
    Test kết nối với Milvus server
    Args:
        URI_link (str): Đường dẫn kết nối đến Milvus
    """
    try:
        from pymilvus import connections, utility
        
        # Thử kết nối
        connections.connect("default", uri=URI_link)
        
        # Liệt kê các collection
        collections = utility.list_collections()
        print(f"✅ Kết nối Milvus thành công!")
        print(f"📋 Các collection có sẵn: {collections}")
        
        # Đóng kết nối
        connections.disconnect("default")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi kết nối Milvus: {str(e)}")
        print("💡 Hướng dẫn khắc phục:")
        print("1. Kiểm tra Milvus server đã chạy: docker ps")
        print("2. Kiểm tra port 19530 có mở không")
        print("3. Restart Milvus: docker-compose restart")
        return False

# Test function
# if __name__ == "__main__":
#     # Test kết nối Milvus
#     print("🧪 Testing Milvus connection...")
#     test_connection()
    
#     # Test load dữ liệu local (nếu có file)
#     try:
#         data, doc_name = load_data_from_local("stack.json", "data")
#         print(f"✅ Test load data thành công: {len(data)} items, doc_name: {doc_name}")
#     except Exception as e:
#         print(f"⚠️ Test load data: {str(e)}")

if __name__ == "__main__":
    # 1) kiểm tra Milvus
    test_connection()

    # 2) thực sự seed dữ liệu
    seed_milvus(
        URI_link="http://localhost:19530",
        collection_name="stack_docs",          # tên bạn muốn
        filename="stack.json",
        directory="data",
        use_ollama=False                       # hoặc True nếu dùng Ollama
    )
