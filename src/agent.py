# Import các thư viện cần thiết
from langchain.tools.retriever import create_retriever_tool  # Tạo công cụ tìm kiếm
from langchain_google_genai import ChatGoogleGenerativeAI  # Model Gemini từ Google
from langchain_ollama import ChatOllama  # Model Ollama local
from langchain.agents import AgentExecutor, create_openai_functions_agent  # Tạo và thực thi agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Xử lý prompt
from seed_data import seed_milvus, connect_to_milvus  # Kết nối với Milvus
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler  # Xử lý callback cho Streamlit
from langchain_community.chat_message_histories import StreamlitChatMessageHistory  # Lưu trữ lịch sử chat
from langchain.retrievers import EnsembleRetriever  # Kết hợp nhiều retriever
from langchain_community.retrievers import BM25Retriever  # Retriever dựa trên BM25
from langchain_core.documents import Document  # Lớp Document
from dotenv import load_dotenv
from langchain.agents import create_react_agent
from langchain.agents import initialize_agent
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

def get_retriever(collection_name: str = "data_test") -> EnsembleRetriever:
    """
    Tạo một ensemble retriever kết hợp vector search (Milvus) và BM25
    Args:
        collection_name (str): Tên collection trong Milvus để truy vấn
    """
    try:
        # Kết nối với Milvus và tạo vector retriever
        vectorstore = connect_to_milvus('http://localhost:19530', collection_name)
        milvus_retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 4}
        )

        # Tạo BM25 retriever từ toàn bộ documents
        documents = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in vectorstore.similarity_search("", k=100)
        ]
        
        if not documents:
            raise ValueError(f"Không tìm thấy documents trong collection '{collection_name}'")
            
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 4

        # Kết hợp hai retriever với tỷ trọng
        ensemble_retriever = EnsembleRetriever(
            retrievers=[milvus_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )
        return ensemble_retriever
        
    except Exception as e:
        print(f"Lỗi khi khởi tạo retriever: {str(e)}")
        # Trả về retriever với document mặc định nếu có lỗi
        default_doc = [
            Document(
                page_content="Có lỗi xảy ra khi kết nối database. Vui lòng thử lại sau.",
                metadata={"source": "error"}
            )
        ]
        return BM25Retriever.from_documents(default_doc)

# Tạo công cụ tìm kiếm cho agent
# tool = create_retriever_tool(
#     get_retriever(),
#     "find",
#     "Search for information of Stack AI."
# )
# --- MỚI: luôn trả về chuỗi ≠ "" để Gemini không lỗi 400 ---
from langchain.tools import Tool       # thêm import này

retriever = get_retriever()            # giữ retriever gốc

def run_retriever_safe(query: str) -> str:
    """
    Gọi retriever. Nếu KHÔNG tìm thấy tài liệu,
    trả về 'NO_RESULTS' (≥1 ký tự) thay vì chuỗi rỗng ''.
    """
    docs = retriever.invoke(query)
    # if not docs:
    #     return "NO_RESULTS"
    # return "\n\n".join(d.page_content for d in docs[:4])
    return "\n\n".join(d.page_content for d in docs) if docs else "NO_RESULTS"

tool = Tool(
    name="find",
    func=run_retriever_safe,
    description="Search Bitcoin knowledge base and return text."
)


def get_llm_and_agent(_retriever, model_choice="gemini") -> AgentExecutor:
    """
    Khởi tạo Language Model và Agent với cấu hình cụ thể
    Args:
        _retriever: Retriever đã được cấu hình để tìm kiếm thông tin
        model_choice: Lựa chọn model ("gemini" hoặc "ollama")
    """
    # Khởi tạo LLM dựa trên lựa chọn model
    if model_choice == "gemini":
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            api_key=GEMINI_API_KEY,
            convert_system_message_to_human=True  # Gemini cần convert system message
        )
    else:  # ollama
        llm = ChatOllama(
            model="llama3.2:3b",  # Hoặc model khác bạn đã cài đặt
            temperature=0,
            base_url="http://localhost:11434"  # URL mặc định của Ollama
        )
    
    tools = [tool]
    
    # Thiết lập prompt template cho agent
    system = """You are ChatchatAI, an expert assistant.
    """

    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{input}"),
        # MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),
    ])

    # Tạo và trả về agent
    try:
        # Thử tạo OpenAI functions agent trước
        agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True)
    except Exception as e:
        # print(f"Không thể tạo OpenAI functions agent: {e}")
        # # Fallback: Tạo agent cơ bản hơn
        # from langchain.agents import create_react_agent
        
        # # Điều chỉnh prompt cho ReAct agent
        # react_prompt = ChatPromptTemplate.from_messages([
        #     ("system", f"{system}\n\nYou have access to the following tools:\n{{tools}}\n\nUse the following format:\nQuestion: the input question\nThought: think about what to do\nAction: the action to take, should be one of [{{tool_names}}]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question"),
            
        #     ("human", "{input}"),
        #     # MessagesPlaceholder(variable_name="agent_scratchpad"),
        # ])
        
        # agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
        # return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
        agent_executor = initialize_agent(
            tools=tools,
            llm=llm,
            agent="zero-shot-react-description",   # không cần agent_scratchpad
            verbose=True,
            handle_parsing_errors=True,
        )
        return agent_executor
# Khởi tạo retriever và agent
retriever = get_retriever()
agent_executor = get_llm_and_agent(retriever)