import os
from dotenv import load_dotenv
from crewai_tools import JSONSearchTool
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# === LLM Provider Selection ===
llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()

if llm_provider == "nvidia":
    os.environ["MODEL"] = f"openai/{os.getenv('NVIDIA_MODEL_NAME', 'meta/llama-3.1-8b-instruct')}"
    os.environ["OPENAI_API_BASE"] = os.getenv("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1")
    os.environ["OPENAI_API_KEY"] = os.getenv("NVIDIA_API_KEY", "")
elif llm_provider == "groq":
    # 支援使用 Groq 呼叫與使用者提供的特製參數 openai/gpt-oss-120b
    # 透過加上 groq/ 前綴，LiteLLM(CrewAI底層) 會直接使用標準的 Groq 路由
    os.environ["MODEL"] = "groq/openai/gpt-oss-120b"
    # litellm 會自動取用 os.environ["GROQ_API_KEY"]
else:
    # Default to local Ollama Phi3
    os.environ["MODEL"] = "ollama/phi3"

# Workaround for early CrewAI-Tools versions that enforce OpenAI Key validation via Pydantic
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "NA")

def get_default_llm():
    """
    提供給 Agent 指定使用，若是在 groq 模式下，會套用使用者要求的特定溫度與 tokens 參數。
    """
    from crewai import LLM
    if llm_provider == "groq":
        return LLM(
            model="groq/openai/gpt-oss-120b",
            temperature=1,
            max_tokens=8192,
            top_p=1
            # reasoning_effort="medium" (Groq API 尚未支援此屬性，會跳錯因此隱藏)
        )
    elif llm_provider == "nvidia":
        return LLM(
            model=os.environ["MODEL"], 
            api_key=os.environ.get("NVIDIA_API_KEY"), 
            base_url=os.environ.get("OPENAI_API_BASE"),
            temperature=1,
            top_p=0.95,
            max_tokens=8192
        )
    else:
        return LLM(model=os.environ["MODEL"])


# Embedding Model for converting text to numerical representations
embedding_model = HuggingFaceEmbeddings(
    model_name='BAAI/bge-small-en-v1.5'
)

rag_config = {
    "embedding_model": {
        "provider": "sentence-transformer",
        "config": {
            "model_name": "BAAI/bge-small-en-v1.5"
        }
    }
}

# === Step 3: Configure RAG Tools (CrewAI RAG Tools) ===
def create_rag_tool(json_path: str, collection_name: str, config: dict, name: str, description: str) -> JSONSearchTool:
    from crewai.utilities.paths import db_storage_path
    from crewai_tools.tools.json_search_tool.json_search_tool import FixedJSONSearchToolSchema
    import sqlite3
    import os
    
    collection_exists = False
    db_file = os.path.join(db_storage_path(), "chroma.sqlite3")
    
    if os.path.exists(db_file):
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM collections WHERE name = ?", (collection_name,))
            if cursor.fetchone() is not None:
                collection_exists = True
            conn.close()
        except Exception:
            pass

    if collection_exists:
        tool = JSONSearchTool(collection_name=collection_name, config=config)
        tool.args_schema = FixedJSONSearchToolSchema
    else:
        tool = JSONSearchTool(json_path=json_path, collection_name=collection_name, config=config)
        
    tool.name = name
    tool.description = description
    return tool

user_rag_tool = create_rag_tool(
    json_path='data/filtered_user.json',
    collection_name='benchmark_true_fresh_index_Filtered_User_1',
    config=rag_config,
    name="search_user_profile_data",
    description=(
        "Searches the user profile database using semantic similarity. "
        "Input MUST be a natural language search_query string, e.g. "
        "'What are the review habits and average stars for user _BcWyKQL16?'. "
        "Do NOT pass raw user_id or JSON objects directly."
    )
)

item_rag_tool = create_rag_tool(
    json_path='data/filtered_item.json',
    collection_name='benchmark_true_fresh_index_Filtered_Item_1',
    config=rag_config,
    name="search_restaurant_feature_data",
    description=(
        "Searches the restaurant/business database using semantic similarity. "
        "Input MUST be a natural language search_query string, e.g. "
        "'What are the categories, location, and star rating for business abc123?'. "
        "Do NOT pass raw item_id or JSON objects directly."
    )
)

review_rag_tool = create_rag_tool(
    json_path='data/test_review.json',
    collection_name='benchmark_true_fresh_index_Filtered_Review_1',
    config=rag_config,
    name="search_historical_reviews_data",
    description=(
        "Searches historical review texts using semantic similarity. "
        "Input MUST be a natural language search_query string, e.g. "
        "'Find past reviews written by user _BcWyKQL16 about food quality and service'. "
        "Do NOT pass raw user_id, item_id, or JSON objects directly."
    )
)

# === Step 2: Inject Global Background Knowledge (CrewAI Knowledge) ===
schema_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'docs', 'Yelp Data Translation.md')
# Handle path resolution robustly from within tools depending on where run() is invoked from
schema_path = os.path.abspath(schema_path)

if os.path.exists(schema_path):
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema_content = f.read()
else:
    schema_content = "Fallback schema"

schema_knowledge = StringKnowledgeSource(
    content=schema_content,
    metadata={"source": "Yelp Schema Definition"}
)
