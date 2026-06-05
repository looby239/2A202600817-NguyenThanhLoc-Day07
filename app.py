import os
import sys
import re
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Ensure dotenv is loaded
load_dotenv()

from src import (
    Document,
    EmbeddingStore,
    KnowledgeBaseAgent,
    FixedSizeChunker,
    SentenceChunker,
    RecursiveChunker,
    OpenAIEmbedder,
    _mock_embed,
)

app = FastAPI(title="RAG Laboratory Backend")

# Global variables to store indexed vector store state
global_store: Optional[EmbeddingStore] = None
global_chapters: List[str] = []
global_embedding_backend: str = "Chưa nạp"
global_filename: str = ""

# Request models
class ChunkRequest(BaseModel):
    file_path: str
    strategy: str
    chunk_size: int
    overlap: int
    max_sentences_per_chunk: int

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    chapter_filter: Optional[str] = None

# Root endpoint serving index.html
@app.get("/", response_class=HTMLResponse)
def get_home():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

# API to list available files in the data directory
@app.get("/api/documents")
def get_documents():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(data_dir):
        return []
    
    docs = []
    for file in os.listdir(data_dir):
        if file.endswith((".txt", ".md")):
            docs.append({
                "name": file,
                "path": os.path.join("data", file)
            })
    # Sort so luat116.md and 81-btc.md are at the top
    docs.sort(key=lambda x: x["name"])
    return docs

# API to fetch current vector store indexing status
@app.get("/api/status")
def get_status():
    global global_store, global_chapters, global_embedding_backend, global_filename
    return {
        "is_indexed": global_store is not None,
        "embedding_backend": global_embedding_backend,
        "filename": global_filename,
        "chapters": global_chapters
    }

# API to preview chunking results without committing to the vector store
@app.post("/api/chunk")
def preview_chunk(req: ChunkRequest):
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    with open(req.file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Standardize carriage returns
    content = content.replace("\r\n", "\n")

    # Select chunker
    if req.strategy == "fixed_size":
        chunker = FixedSizeChunker(chunk_size=req.chunk_size, overlap=req.overlap)
    elif req.strategy == "by_sentences":
        chunker = SentenceChunker(max_sentences_per_chunk=req.max_sentences_per_chunk)
    else:
        chunker = RecursiveChunker(chunk_size=req.chunk_size)
        
    chunks = chunker.chunk(content)
    
    avg_len = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
    return {
        "filename": os.path.basename(req.file_path),
        "total_chunks": len(chunks),
        "avg_length": avg_len,
        "chunks": chunks
    }

# API to index a file, parse chapters, chunk, embed, and store in vector db
@app.post("/api/index")
def index_document(req: ChunkRequest):
    global global_store, global_chapters, global_embedding_backend, global_filename
    
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    with open(req.file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    content = content.replace("\r\n", "\n")
    
    # 1. Parse Metadata Schema by filename
    doc_id = "unknown"
    authority = "unknown"
    filename_lower = os.path.basename(req.file_path).lower()
    if "81-btc" in filename_lower:
        doc_id = "81/2025/TT-BTC"
        authority = "Bộ Tài chính"
    elif "luat116" in filename_lower:
        doc_id = "116/2025/QH15"
        authority = "Quốc hội"
    doc_id_clean = doc_id.replace("/", "-")

    # 2. Extract chapters dynamically
    chapters = []
    current_chapter_name = "Chương I"
    current_chapter_lines = []
    
    lines = content.split("\n")
    for line in lines:
        match = re.match(r'^(?:#+\s+)?(Chương\s+[IVXLCD]+|PHỤ\s+LỤC)', line.strip(), re.IGNORECASE)
        if match:
            if current_chapter_lines:
                chapters.append((current_chapter_name, "\n".join(current_chapter_lines)))
            current_chapter_name = match.group(1).strip().upper()
            current_chapter_lines = [line]
        else:
            current_chapter_lines.append(line)
            
    if current_chapter_lines:
        chapters.append((current_chapter_name, "\n".join(current_chapter_lines)))

    # 3. Select chunker
    if req.strategy == "fixed_size":
        chunker = FixedSizeChunker(chunk_size=req.chunk_size, overlap=req.overlap)
    elif req.strategy == "by_sentences":
        chunker = SentenceChunker(max_sentences_per_chunk=req.max_sentences_per_chunk)
    else:
        chunker = RecursiveChunker(chunk_size=req.chunk_size)

    # 4. Create document chunks
    documents = []
    chunk_global_idx = 0
    unique_chapters = set()
    
    for chapter_name, chapter_text in chapters:
        unique_chapters.add(chapter_name)
        chapter_chunks = chunker.chunk(chapter_text)
        for chunk_text in chapter_chunks:
            doc = Document(
                id=f"{doc_id_clean}-chunk-{chunk_global_idx}",
                content=chunk_text,
                metadata={
                    "source": req.file_path,
                    "doc_id": doc_id,
                    "authority": authority,
                    "chapter": chapter_name,
                    "chunk_index": chunk_global_idx
                }
            )
            documents.append(doc)
            chunk_global_idx += 1

    # 5. Initialize embedder engine (OpenAI if key is present, else mock fallback)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key.startswith("sk-"):
        try:
            embedder = OpenAIEmbedder()
            backend_name = "OpenAI API (text-embedding-3-small)"
        except Exception as e:
            embedder = _mock_embed
            backend_name = f"Mock Fallback (Lỗi khởi tạo OpenAI: {e})"
    else:
        embedder = _mock_embed
        backend_name = "Mock Embeddings (Không có API Key)"

    # 6. Index documents into Vector Store
    store = EmbeddingStore("web_lab_store", embedding_fn=embedder)
    store.add_documents(documents)
    
    # Update global state
    global_store = store
    global_chapters = sorted(list(unique_chapters))
    global_embedding_backend = backend_name
    global_filename = os.path.basename(req.file_path)
    
    return {
        "message": f"Đã nạp thành công {len(documents)} chunks từ {len(chapters)} chương bằng backend: {backend_name}!",
        "total_chunks": len(documents),
        "embedding_backend": backend_name
    }

# API to retrieve context and generate answer using KnowledgeBaseAgent
@app.post("/api/query")
def query_agent(req: QueryRequest):
    global global_store
    if not global_store:
        raise HTTPException(status_code=400, detail="Vector Store chưa được khởi tạo. Hãy nạp tài liệu trước.")
        
    # 1. Retrieve chunks (filtered or unfiltered)
    if req.chapter_filter:
        results = global_store.search_with_filter(
            req.query, 
            top_k=req.top_k, 
            metadata_filter={"chapter": req.chapter_filter}
        )
    else:
        results = global_store.search(req.query, top_k=req.top_k)
        
    # 2. Build RAG LLM completing function
    # Uses real OpenAI model if key is active, otherwise falls back to static demo message
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key.startswith("sk-"):
        try:
            from openai import OpenAI
            client = OpenAI()
            def real_llm(prompt: str) -> str:
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Bạn là một trợ lý pháp lý chuyên nghiệp. Hãy trả lời câu hỏi dựa trên văn bản ngữ cảnh được cung cấp. Chỉ trả lời dựa trên thông tin có sẵn, nếu không có hãy nói 'Tôi không biết'."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                return completion.choices[0].message.content.strip()
            llm_fn = real_llm
        except Exception as e:
            def fallback_llm(prompt: str) -> str:
                return f"[Lỗi kết nối OpenAI completion: {e}]\n\nPrompt nhận được:\n{prompt}"
            llm_fn = fallback_llm
    else:
        def mock_llm(prompt: str) -> str:
            # Create a mock response summarizing the prompt context
            return (
                "[MOCK LLM RESPONSE] Bạn đang chạy ở chế độ offline (Không có API Key).\n"
                "Nếu có API Key, LLM sẽ nhận được các ngữ cảnh sau và trả lời:\n\n"
                f"Câu hỏi: {req.query}\n"
                f"Đã tìm thấy {len(results)} đoạn văn bản khớp nhất."
            )
        llm_fn = mock_llm

    # 3. Ask Agent
    agent = KnowledgeBaseAgent(global_store, llm_fn=llm_fn)
    answer = agent.answer(req.query, top_k=req.top_k)
    
    return {
        "results": results,
        "answer": answer
    }
