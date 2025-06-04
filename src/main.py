"""
File chính để chạy ứng dụng Chatbot AI
Chức năng: 
- Tạo giao diện web với Streamlit
- Xử lý tương tác chat với người dùng
- Kết nối với AI model để trả lời (Gemini & Ollama)
"""

# === IMPORT CÁC THƯ VIỆN CẦN THIẾT ===
import streamlit as st  # Thư viện tạo giao diện web
from dotenv import load_dotenv 
import asyncio, nest_asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
nest_asyncio.apply()
from seed_data import seed_milvus, seed_milvus_live  # Hàm xử lý dữ liệu
from agent import get_retriever, get_llm_and_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# === THIẾT LẬP GIAO DIỆN TRANG WEB ===
def setup_page():
    """
    Cấu hình trang web cơ bản
    """
    st.set_page_config(
        page_title="AI Assistant",  # Tiêu đề tab trình duyệt
        page_icon="💬",  # Icon tab
        layout="wide"  # Giao diện rộng
    )

# === KHỞI TẠO ỨNG DỤNG ===
def initialize_app():
    """
    Khởi tạo các cài đặt cần thiết:
    - Đọc file .env chứa API key
    - Cấu hình trang web
    """
    load_dotenv()  # Đọc API key từ file .env
    setup_page()  # Thiết lập giao diện

# === THANH CÔNG CỤ BÊN TRÁI ===
def setup_sidebar():
    """
    Tạo thanh công cụ bên trái với các tùy chọn
    """
    with st.sidebar:
        st.title("⚙️ Cấu hình")
        
        # Phần 1: Chọn Embeddings Model
        st.header("🔤 Embeddings Model")
        embeddings_choice = st.radio(
            "Chọn Embeddings Model:",
            ["Gemini", "Ollama"]
        )
        use_ollama_embeddings = (embeddings_choice == "Ollama")
        
        # Phần 2: Cấu hình Data
        st.header("📚 Nguồn dữ liệu")
        data_source = st.radio(
            "Chọn nguồn dữ liệu:",
            ["File Local", "URL trực tiếp"]
        )
        
        # Xử lý nguồn dữ liệu dựa trên embeddings đã chọn
        if data_source == "File Local":
            handle_local_file(use_ollama_embeddings)
        else:
            handle_url_input(use_ollama_embeddings)
            
        # Thêm phần chọn collection để query
        st.header("🔍 Collection để truy vấn")
        collection_to_query = st.text_input(
            "Nhập tên collection cần truy vấn:",
            "data_test",
            help="Nhập tên collection bạn muốn sử dụng để tìm kiếm thông tin"
        )
        
        # Phần 3: Chọn Model để trả lời
        st.header("🤖 Model AI")
        model_choice = st.radio(
            "Chọn AI Model để trả lời:",
            ["Google Gemini", "Ollama (Local)"]
        )
        
        # Phần 4: Cấu hình Ollama (nếu chọn Ollama)
        if model_choice == "Ollama (Local)":
            st.subheader("🦙 Cấu hình Ollama")
            ollama_model = st.selectbox(
                "Chọn model Ollama:",
                ["llama3.2:3b", "llama3.2:1b", "llama3.1:8b", "qwen2.5:7b", "mistral:7b"],
                help="Đảm bảo model đã được pull về local"
            )
            
            ollama_url = st.text_input(
                "URL Ollama server:",
                "http://localhost:11434",
                help="URL của Ollama server"
            )
            
            # Kiểm tra kết nối Ollama
            if st.button("🔍 Kiểm tra kết nối Ollama"):
                check_ollama_connection(ollama_url, ollama_model)
        else:
            ollama_model = None
            ollama_url = None
        
        return model_choice, collection_to_query, ollama_model, ollama_url

def check_ollama_connection(ollama_url, model_name):
    """
    Kiểm tra kết nối với Ollama server
    """
    try:
        import requests
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            
            if model_name in model_names:
                st.success(f"✅ Kết nối thành công! Model '{model_name}' đã sẵn sàng.")
            else:
                st.warning(f"⚠️ Kết nối thành công nhưng model '{model_name}' chưa được cài đặt.")
                st.info(f"Chạy lệnh: `ollama pull {model_name}` để cài đặt model.")
                st.info(f"Các model có sẵn: {', '.join(model_names) if model_names else 'Không có'}")
        else:
            st.error(f"❌ Không thể kết nối với Ollama server. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Lỗi kết nối với Ollama server: {str(e)}")
        st.info("Đảm bảo Ollama đang chạy. Chạy lệnh: `ollama serve`")

def handle_local_file(use_ollama_embeddings: bool):
    """
    Xử lý khi người dùng chọn tải file
    """
    collection_name = st.text_input(
        "Tên collection trong Milvus:", 
        "data_test",
        help="Nhập tên collection bạn muốn lưu trong Milvus"
    )
    filename = st.text_input("Tên file JSON:", "stack.json")
    directory = st.text_input("Thư mục chứa file:", "data")
    
    if st.button("📁 Tải dữ liệu từ file"):
        if not collection_name:
            st.error("Vui lòng nhập tên collection!")
            return
            
        with st.spinner("Đang tải dữ liệu..."):
            try:
                seed_milvus(
                    'http://localhost:19530', 
                    collection_name, 
                    filename, 
                    directory, 
                    use_ollama=use_ollama_embeddings
                )
                st.success(f"✅ Đã tải dữ liệu thành công vào collection '{collection_name}'!")
            except Exception as e:
                st.error(f"❌ Lỗi khi tải dữ liệu: {str(e)}")

def handle_url_input(use_ollama_embeddings: bool):
    """
    Xử lý khi người dùng chọn crawl URL
    """
    collection_name = st.text_input(
        "Tên collection trong Milvus:", 
        "data_test_live",
        help="Nhập tên collection bạn muốn lưu trong Milvus"
    )
    url = st.text_input("Nhập URL:", "https://tapchibitcoin.io")
    
    if st.button("🌐 Crawl dữ liệu"):
        if not collection_name:
            st.error("Vui lòng nhập tên collection!")
            return
            
        with st.spinner("Đang crawl dữ liệu..."):
            try:
                seed_milvus_live(
                    url, 
                    'http://localhost:19530', 
                    collection_name, 
                    'stack-ai', 
                    use_ollama=use_ollama_embeddings
                )
                st.success(f"✅ Đã crawl dữ liệu thành công vào collection '{collection_name}'!")
            except Exception as e:
                st.error(f"❌ Lỗi khi crawl dữ liệu: {str(e)}")

# === GIAO DIỆN CHAT CHÍNH ===
def setup_chat_interface(model_choice):
    st.title("💬 AI Assistant")
    
    # Caption động theo model
    if model_choice == "Google Gemini":
        st.caption("🚀 Trợ lý AI được hỗ trợ bởi LangChain và Google Gemini")
    else:
        st.caption("🚀 Trợ lý AI được hỗ trợ bởi LangChain và Ollama")
    
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Xin chào! Tôi có thể giúp gì cho bạn?"}
        ]
        msgs.add_ai_message("Xin chào! Tôi có thể giúp gì cho bạn?")

    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "human"
        st.chat_message(role).write(msg["content"])

    return msgs

# === XỬ LÝ TIN NHẮN NGƯỜI DÙNG ===
def handle_user_input(msgs, agent_executor):
    """
    Xử lý khi người dùng gửi tin nhắn:
    1. Hiển thị tin nhắn người dùng
    2. Gọi AI xử lý và trả lời
    3. Lưu vào lịch sử chat
    """
    if prompt := st.chat_input("Hãy hỏi tôi bất cứ điều gì về Stack AI!"):
        # Lưu và hiển thị tin nhắn người dùng
        st.session_state.messages.append({"role": "human", "content": prompt})
        st.chat_message("human").write(prompt)
        msgs.add_user_message(prompt)

        # Xử lý và hiển thị câu trả lời
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            
            # Lấy lịch sử chat
            chat_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages[:-1]
            ]

            try:
                # Gọi AI xử lý
                response = agent_executor.invoke(
                    {
                        "input": prompt,
                        "chat_history": chat_history
                    },
                    {"callbacks": [st_callback]}
                )

                # Lưu và hiển thị câu trả lời
                output = response["output"]
                st.session_state.messages.append({"role": "assistant", "content": output})
                msgs.add_ai_message(output)
                st.write(output)
                
            except Exception as e:
                error_msg = f"Đã xảy ra lỗi: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                msgs.add_ai_message(error_msg)

# === HÀM CHÍNH ===
def main():
    """
    Hàm chính điều khiển luồng chương trình
    """
    initialize_app()
    model_choice, collection_to_query, ollama_model, ollama_url = setup_sidebar()
    msgs = setup_chat_interface(model_choice)
    
    # Khởi tạo AI dựa trên lựa chọn model để trả lời
    try:
        retriever = get_retriever(collection_to_query)
        
        if model_choice == "Google Gemini":
            agent_executor = get_llm_and_agent(retriever, "gemini")
        else:  # Ollama
            # Cập nhật cấu hình Ollama nếu cần
            if ollama_model and ollama_url:
                # Có thể thêm logic để truyền cấu hình Ollama
                pass
            agent_executor = get_llm_and_agent(retriever, "ollama")
        
        handle_user_input(msgs, agent_executor)
        
    except Exception as e:
        st.error(f"❌ Lỗi khởi tạo hệ thống: {str(e)}")
        st.info("Vui lòng kiểm tra:")
        st.info("1. Milvus server đang chạy (localhost:19530)")
        st.info("2. API key được cấu hình đúng")
        st.info("3. Ollama server đang chạy (nếu sử dụng Ollama)")

# Chạy ứng dụng
if __name__ == "__main__":
    main()