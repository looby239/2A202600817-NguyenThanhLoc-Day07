import os
import sys
import re

# Reconfigure stdout to use UTF-8 to prevent UnicodeEncodeError on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
# Load environment variables (such as OPENAI_API_KEY) from .env file
load_dotenv()

from src import Document, EmbeddingStore, RecursiveChunker, KnowledgeBaseAgent, OpenAIEmbedder

def main():
    # 1. Khởi tạo Chunker để cắt nhỏ tài liệu
    chunker = RecursiveChunker(chunk_size=300)
    
    # 2. Xác định file tài liệu mới
    # Nếu có tham số thứ 2, dùng file đó, ngược lại mặc định dùng luat116.md
    file_path = "data/luat116.md"
    if len(sys.argv) > 2:
        file_path = sys.argv[2]
        
    if not os.path.exists(file_path):
        print(f"Không tìm thấy file: {file_path}. Vui lòng tạo file này trước.")
        return
        
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Xác định siêu dữ liệu (Metadata Schema) dựa trên file
    doc_id = "unknown"
    authority = "unknown"
    filename_lower = os.path.basename(file_path).lower()
    
    if "81-btc" in filename_lower:
        doc_id = "81/2025/TT-BTC"
        authority = "Bộ Tài chính"
    elif "luat116" in filename_lower:
        doc_id = "116/2025/QH15"
        authority = "Quốc hội"

    doc_id_clean = doc_id.replace("/", "-")
    
    # 3. Phân tách văn bản thành các Chương để gán metadata chính xác
    chapters = []
    current_chapter_name = "Chương I"
    current_chapter_lines = []
    
    lines = content.split("\n")
    for line in lines:
        # Phát hiện tiêu đề Chương hoặc Phụ lục
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

    # 4. Chia nhỏ (Chunking) từng chương và tạo Document objects kèm Metadata
    documents = []
    chunk_global_idx = 0
    for chapter_name, chapter_text in chapters:
        chapter_chunks = chunker.chunk(chapter_text)
        for chunk_text in chapter_chunks:
            doc = Document(
                id=f"{doc_id_clean}-chunk-{chunk_global_idx}",
                content=chunk_text,
                metadata={
                    "source": file_path,
                    "doc_id": doc_id,
                    "authority": authority,
                    "chapter": chapter_name,
                    "chunk_index": chunk_global_idx
                }
            )
            documents.append(doc)
            chunk_global_idx += 1

    print(f"Đã phân tích tài liệu '{file_path}' ({authority}) thành {len(chapters)} chương.")
    print(f"Tổng số chunks được tạo ra: {len(documents)}")
    
    # 5. Khởi tạo Vector Store với OpenAI Embedder
    try:
        embedder = OpenAIEmbedder()
    except Exception as e:
        print(f"Lỗi khởi tạo OpenAIEmbedder: {e}")
        print("Vui lòng kiểm tra đã cài đặt thư viện openai và cấu hình OPENAI_API_KEY chưa.")
        return

    store = EmbeddingStore("custom_search_store", embedding_fn=embedder)
    store.add_documents(documents)
    print(f"Đã nạp {store.get_collection_size()} chunks vào EmbeddingStore.")
    
    # 6. Nhận diện câu hỏi từ tham số dòng lệnh
    if len(sys.argv) > 1:
        query = sys.argv[1].strip()
    else:
        query = "Phạm vi điều chỉnh"

    # Demo 1: Tìm kiếm KHÔNG dùng bộ lọc (Unfiltered Search)
    print(f"\n=== [DEMO 1] TÌM KIẾM KHÔNG BỘ LỌC cho câu hỏi: '{query}' ===")
    results_unfiltered = store.search(query, top_k=2)
    for index, res in enumerate(results_unfiltered, start=1):
        print(f"{index}. [Score: {res['score']:.3f}] Chapter: {res['metadata'].get('chapter')}")
        print(f"   Nội dung: {res['content'][:250].strip()}...\n")

    # Demo 2: Tìm kiếm CÓ bộ lọc Metadata (Filtered Search)
    # Ví dụ chúng ta chỉ muốn tìm thông tin nằm ở "CHƯƠNG I"
    target_chapter = "CHƯƠNG I"
    print(f"=== [DEMO 2] TÌM KIẾM CÓ BỘ LỌC METADATA (chỉ tìm trong '{target_chapter}') ===")
    results_filtered = store.search_with_filter(query, top_k=2, metadata_filter={"chapter": target_chapter})
    
    if not results_filtered:
        print("Không tìm thấy kết quả nào khớp với bộ lọc.\n")
    for index, res in enumerate(results_filtered, start=1):
        print(f"{index}. [Score: {res['score']:.3f}] Chapter: {res['metadata'].get('chapter')}")
        print(f"   Nội dung: {res['content'][:250].strip()}...\n")

    # 7. Chạy thử nghiệm câu trả lời với KnowledgeBaseAgent
    print("=== [DEMO 3] Chạy thử với KnowledgeBaseAgent ===")
    def demo_llm(prompt):
        return f"[DEMO LLM] Đã nhận Prompt RAG. Preview:\n{prompt[:350]}..."
        
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
    answer = agent.answer(query, top_k=2)
    print(answer)

if __name__ == "__main__":
    main()
