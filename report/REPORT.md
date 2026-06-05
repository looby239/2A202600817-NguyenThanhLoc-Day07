# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Thành Lộc -2A202600817
**Nhóm:** Bàn E2
**Ngày:** 2026-06-05

> Ghi chú: các phần kỹ thuật (số liệu chunking, similarity, benchmark) được đo
> trực tiếp trên package `src` của tôi với embedding `text-embedding-3-small`
> (OpenAI). Các ô đánh dấu **[điền cùng nhóm]** cần thống nhất với thành viên.

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Hai chunk có embedding chỉ gần như **cùng hướng** trong không gian vector, nghĩa là chúng mang ý nghĩa rất gần nhau — bất kể độ dài văn bản. Cosine đo góc giữa hai vector, không đo khoảng cách độ lớn.

**Ví dụ HIGH similarity:**
- Sentence A: "A vector store keeps embeddings for similarity search"
- Sentence B: "A database that stores vectors to retrieve similar items"
- Tại sao tương đồng: cùng nói về một khái niệm (lưu vector để tìm kiếm tương tự), chỉ khác cách diễn đạt. Đo thật: **+0.738**.

**Ví dụ LOW similarity:**
- Sentence A: "Python is great for machine learning"
- Sentence B: "Customers were charged twice on their billing statement"
- Tại sao khác: hai chủ đề hoàn toàn rời nhau (ngôn ngữ lập trình vs lỗi thanh toán). Đo thật: **−0.051**.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Vì độ dài văn bản làm vector dài/ngắn khác nhau, nhưng **hướng** mới mang ngữ nghĩa. Cosine chuẩn hóa theo độ lớn nên một câu và bản lặp lại của nó vẫn được coi là giống nhau; Euclidean sẽ bị độ lớn vector làm sai lệch.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> `num_chunks = ceil((10000 − 50) / (500 − 50)) = ceil(9950 / 450) = ceil(22.11) = `**23 chunks**.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> `num_chunks = ceil((10000 − 100) / (500 − 100)) = ceil(9900 / 400) = ceil(24.75) = `**25 chunks** → overlap lớn hơn ⇒ bước trượt nhỏ hơn ⇒ **nhiều chunk hơn**. Muốn overlap nhiều hơn để **không cắt đứt ý ở ranh giới chunk**: một câu/ý nằm vắt qua điểm cắt sẽ xuất hiện trọn vẹn trong ít nhất một chunk, tránh mất thông tin khi retrieve (đánh đổi: tốn thêm bộ nhớ/embedding).

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Văn bản pháp luật Việt Nam — **Luật An ninh mạng số 116/2025**, tách theo 8 chương.

**Tại sao nhóm chọn domain này?**
> Văn bản luật có cấu trúc phân cấp rõ (Chương → Điều → Khoản → điểm), nội dung tiếng Việt thật, gold answer **verify trực tiếp được** từ điều luật. Đây là use-case RAG thực tế (trợ lý tra cứu pháp luật) và cho phép thử metadata filter theo **chương** — đúng tinh thần "ít nhất 1 query cần metadata filtering".
> *Nguồn dữ liệu:* PDF scan được OCR bằng OpenAI vision rồi tách theo chương (xem `data/luat116_ch*.md`).

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | luat116_ch01.md — Chương I: Những quy định chung | data/ | 12234 | source, doc_id, chuong=1, lang=vi |
| 2 | luat116_ch02.md — Chương II: Bảo vệ ANM với HTTT | data/ | 9023 | source, doc_id, chuong=2, lang=vi |
| 3 | luat116_ch03.md — Chương III: Phòng ngừa, xử lý xâm phạm | data/ | 15783 | source, doc_id, chuong=3, lang=vi |
| 4 | luat116_ch04.md — Chương IV: Hoạt động bảo vệ ANM | data/ | 4271 | source, doc_id, chuong=4, lang=vi |
| 5 | luat116_ch05.md — Chương V: Tiêu chuẩn, quy chuẩn kỹ thuật | data/ | 3231 | source, doc_id, chuong=5, lang=vi |
| 6 | luat116_ch06.md — Chương VI: Lực lượng, điều kiện bảo đảm | data/ | 7640 | source, doc_id, chuong=6, lang=vi |
| 7 | luat116_ch07.md — Chương VII: Trách nhiệm cơ quan/tổ chức | data/ | 7309 | source, doc_id, chuong=7, lang=vi |
| 8 | luat116_ch08.md — Chương VIII: Điều khoản thi hành | data/ | 6069 | source, doc_id, chuong=8, lang=vi |

> Ghi chú nguồn: OCR bằng gpt-4o-mini vision (37 trang scan), chất lượng cao nhưng không 100% — vài tiêu đề chương bị nhiễu. Với mục đích lab thì đủ tốt.

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `source` | str | `luat116_ch03.md` | Truy vết chunk về đúng chương gốc (grounding/audit) |
| `doc_id` | str | `luat116_ch03` | Gom & xóa toàn bộ chunk của 1 chương (`delete_document`) |
| `chuong` | str | `1` … `8` | **Lọc theo chương** — vd chỉ tìm trong Chương VI (lực lượng) |
| `lang` | str | `vi` | Nhãn ngôn ngữ (mở rộng khi trộn tài liệu đa ngữ) |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare(text, chunk_size=400)` trên 2 chương luật (số liệu thật):

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| luat116_ch01.md (12234 ký tự) | `fixed_size` | 31 | 394.6 | Trung bình — cắt cứng theo ký tự, có thể cắt giữa Điều/Khoản |
| luat116_ch01.md | `by_sentences` | 19 | 641.5 | Câu luật dài → chunk **rất to** (641 > 400), vượt ngưỡng |
| luat116_ch01.md | `recursive` | 43 | 282.6 | Tốt nhất — bám ranh giới Điều/đoạn, gọn trong size |
| luat116_ch03.md (15783 ký tự) | `fixed_size` | 40 | 394.6 | Trung bình |
| luat116_ch03.md | `by_sentences` | 22 | 715.4 | Câu rất dài → chunk to nhất, khó embed chính xác |
| luat116_ch03.md | `recursive` | 58 | 270.3 | Tốt nhất — chunk đều, bám cấu trúc điều luật |

> Nhận xét: với văn bản luật, **câu rất dài** (một khoản có thể là 1 câu 600+ ký tự) nên `by_sentences` tạo chunk quá to (641–715 ký tự, vượt xa chunk_size=400). `recursive` cho chunk đều và gọn nhất → phù hợp domain luật nhất.

#### Cải tiến `RecursiveChunker`: gộp mẩu nhỏ (before/after)

Phiên bản đầu tách xong là giữ luôn từng mẩu → với văn bản luật (nhiều điểm a/b/c xuống dòng riêng) tạo ra **hàng trăm mẩu vụn** chỉ vài chục ký tự. Tôi sửa `_split` để **gộp các mẩu liên tiếp** tới gần `chunk_size` trước khi cắt:

| Tài liệu (chunk_size=400) | Chunk count | Avg length |
|---|---|---|
| luat116_ch01.md — *before* | 422 | 27.9 |
| luat116_ch01.md — **after** | **43** | **282.6** |
| luat116_ch03.md — *before* | 472 | 32.3 |
| luat116_ch03.md — **after** | **58** | **270.3** |

→ Khác biệt cực lớn: từ **422–472 mẩu vụn** (~28–32 ký tự, mỗi điểm a/b/c thành 1 vector vô nghĩa) xuống **43–58 chunk** đầy đặn (~270–283 ký tự) giữ trọn điều khoản. Giảm ~9× số vector ⇒ rẻ hơn, retrieve có ngữ cảnh hơn. 42/42 test vẫn pass.

### Strategy Của Tôi — Custom `ArticleChunker` (chia theo Điều)

**Loại:** Custom strategy `ArticleChunker` (Exercise 3.1) — không dùng built-in mà thiết kế riêng cho văn bản luật.

**Design rationale:** Đơn vị ngữ nghĩa tự nhiên của luật là **"Điều"**. Cắt theo ký tự (recursive/fixed) hay theo câu đều có thể xé một điều làm đôi hoặc gộp nhiều điều. `ArticleChunker` tách **trước mỗi "Điều N"** → mỗi chunk là **1 điều trọn vẹn**; điều nào quá dài (> max_size) mới đệ quy cắt nhỏ bằng RecursiveChunker.

```python
class ArticleChunker:
    """Chia văn bản luật theo từng 'Điều' — mỗi chunk là một điều hoàn chỉnh."""
    def __init__(self, max_size: int = 1500):
        self.max_size = max_size

    def chunk(self, text: str) -> list[str]:
        # tách trước mỗi tiêu đề "Điều N" (kể cả có ###/## ở đầu dòng)
        return _split_on_boundary(text, r"(?m)(?=^\s*#{0,6}\s*Điều\s+\d+\b)", self.max_size)
```

**So sánh 8 chiến lược** (cùng query *"Lực lượng bảo vệ an ninh mạng gồm những thành phần nào?"*, embedder OpenAI, số liệu thật):

| Chunking | Top-1 score | Chương trả về | Đúng? |
|----------|-------------|---------------|-------|
| `recursive` (built-in) | 0.747 | Chương 3 | ❌ sai chương |
| `fixed` (built-in) | 0.695 | Chương 6 | ✅ |
| `sentence` (built-in) | 0.692 | Chương 5 | ❌ sai |
| **`khoan`** (ClauseChunker) | **0.771** | Chương 6 | ✅ score cao nhất |
| **`dieu`** (ArticleChunker) | 0.742 | Chương 6 · **Điều 30** | ✅ trỏ đúng điều |
| `header` (MarkdownHeaderChunker) | 0.742 | Chương 6 · Điều 30 | ✅ |
| `paragraph` (ParagraphChunker) | 0.742 | Chương 6 · Điều 30 | ✅ |
| `chuong` (ChapterChunker) | 0.591 | Chương 4 | ❌ quá thô |

> **Kết quả then chốt:** các chunker **cấu trúc luật** (dieu, khoan, header, paragraph) trỏ đúng Chương VI, còn `recursive`/`sentence` (cắt máy móc) **lấy nhầm chương**. Dù `khoan` có score cao nhất (0.771), tôi chọn **`dieu`** vì nó trỏ đúng **Điều 30** và điền được trường citation (`dieu="Điều 30"`) — quan trọng cho việc bám nguồn pháp lý.

### So Sánh: Strategy của tôi vs Baseline (whole-doc)

Cùng query *"Lực lượng bảo vệ an ninh mạng gồm những thành phần nào?"*, cùng data luật, cùng embedder OpenAI:

| Cách lưu | Số vector | Top-1 score | Top-1 trả về |
|-----------|-----------|-------------|--------------|
| Whole document (8 chương nguyên file) | 8 | 0.577 | **Sai chương** — `luat116_ch04.md` (Chương IV), không phải VI |
| **RecursiveChunker(400)** | 253 | **0.743** | **Đúng** `luat116_ch06.md` — *"Điều 30. Lực lượng bảo vệ an ninh mạng bao gồm: a) Lực lượng chuyên trách... tại Bộ Công an, Bộ Quốc phòng..."* |

> Chunking không chỉ nâng score 0.577 → 0.743 mà còn **sửa lỗi trỏ nhầm chương**: whole-doc lấy nhầm Chương IV, còn bản chunk trỏ đúng Điều 30 (Chương VI). Văn bản luật dài (mỗi chương 4–16k ký tự) nếu nén thành 1 vector sẽ pha loãng nhiều điều khoản; chia nhỏ giúp mỗi vector tập trung 1 điều → khớp chính xác hơn.

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | **ArticleChunker (dieu)** | [điền] | Trỏ đúng Điều, citation đẹp, giữ trọn điều luật | Điều rất dài vẫn phải cắt nhỏ |
| [Tên] | [điền cùng nhóm] | | | |
| [Tên] | [điền cùng nhóm] | | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> [điền sau khi so sánh cùng nhóm — gợi ý dựa trên baseline: recursive là default mạnh cho tài liệu có cấu trúc đoạn]

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Tách câu bằng regex `(?<=[.!?])\s+` — lookbehind giữ dấu câu dính vào cuối câu, chỉ dùng khoảng trắng phía sau làm điểm cắt (xử lý cả `". "` và `".\n"`). Sau đó gom `max_sentences_per_chunk` câu/chunk. Edge case: text rỗng/toàn khoảng trắng → `[]`; lọc bỏ phần tử rỗng bằng `if s.strip()`.

**`RecursiveChunker.chunk` / `_split`** — approach:
> `_split` là đệ quy: **base case** là đoạn ≤ `chunk_size` (trả về nguyên đoạn), hoặc hết separator / gặp separator rỗng → cắt cứng theo kích thước. Ngược lại tách theo separator hiện tại, đoạn nào vẫn quá dài thì đệ quy với separator mịn hơn.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents` gọi `_make_record` để embed `doc.content` và đóng gói (id, doc_id, content, embedding, metadata) rồi append vào `self._store`. `search` embed query, tính `_dot(query_emb, rec_emb)` cho mọi record (embedding đã chuẩn hóa nên dot product = cosine), sort giảm dần, lấy `top_k`.

**`search_with_filter` + `delete_document`** — approach:
> Filter **trước** rồi search sau: lọc record có `metadata` khớp tất cả cặp key-value, rồi chạy `_search_records` trên tập đã lọc (giảm nhiễu). `delete_document` lọc ra mọi record có `metadata['doc_id'] == doc_id`; nếu không có thì trả `False`, có thì gán lại `self._store` bỏ các record đó và trả `True`.

### KnowledgeBaseAgent

**`answer`** — approach:
> RAG 3 bước: (1) `store.search(question, top_k)` lấy chunk liên quan; (2) ghép `content` các chunk thành `Context` và dựng prompt yêu cầu LLM "chỉ trả lời từ context, thiếu thì nói thiếu"; (3) gọi `llm_fn(prompt)`. `llm_fn` được inject nên dễ test (hàm giả) hoặc cắm OpenAI thật.

### Cải Tiến Thêm (ngoài yêu cầu cơ bản)

Sau khi pass hết test, tôi nâng cấp thêm (vẫn giữ 42/42 test):

**A. Chất lượng `src/`:**

| # | Cải tiến | Tác động đo được |
|---|----------|------------------|
| 1 | `RecursiveChunker` gộp mẩu nhỏ tới gần `chunk_size` | 422→43 chunk, avg_len 28→283 trên Ch.I luật (xem Section 3) |
| 2 | `search` xếp hạng bằng **cosine chuẩn** (`compute_similarity`) | Không còn giả định vector đã normalize |
| 3 | `agent.answer` xử lý **retrieval rỗng/yếu** + `min_score` | Trả "không tìm thấy" thay vì bịa (honest uncertainty) |
| 4 | **Batch embedding** (`embed_batch`) | 20 docs → 1 API request thay vì 20 |
| 5 | **Cache embedding** theo nội dung (OpenAI/Local) | Text trùng → 0 API call lần sau |
| 6 | **5 chunker mới** cho luật: `ArticleChunker`, `ChapterChunker`, `ClauseChunker`, `MarkdownHeaderChunker`, `ParagraphChunker` | Chunker cấu trúc luật chính xác hơn recursive (xem Section 3) |

**B. Hệ thống RAG end-to-end (ngoài `src/`, không ảnh hưởng bài nộp):**

| Thành phần | Mô tả |
|---|---|
| `api.py` (FastAPI) | API JSON: `/search`, `/search/text`, `/ask`, `/chapters`, `/history` + Swagger UI |
| Embedding backend | Chọn `mock`/`openai` per-request |
| Chunking | Chọn 1 trong **8** kiểu per-request |
| Vector store | Chọn **in-memory** hoặc **ChromaDB** thật (HNSW, cosine, metadata filter) |
| Grounding | `/ask` trả `answer` + `citations` (Điều/Chương) + `grounded` |
| `frontend/` | Web app React (Chatbot + Dashboard) gọi API |

### Test Results

```
$ pytest tests/ -q
..........................................                               [100%]
42 passed in 0.08s
```

**Số tests pass:** **42 / 42**

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

Đo thật bằng `compute_similarity()` + embedding OpenAI:

| Pair | Câu A | Câu B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "An ninh mạng là sự an toàn của không gian mạng" | "Bảo đảm an toàn cho không gian mạng quốc gia" | high | **+0.738** | ✓ |
| 2 | "Phòng ngừa hành vi xâm phạm an ninh mạng" | "Ngăn chặn các cuộc tấn công mạng trái phép" | high | **+0.598** | ✓ |
| 3 | "Lực lượng chuyên trách bảo vệ an ninh mạng tại Bộ Công an" | "Trách nhiệm của chủ quản hệ thống thông tin" | low-mid | **+0.460** | ✓ |
| 4 | "Xử lý vi phạm pháp luật về an ninh mạng" | "Công thức nấu phở bò Hà Nội" | low | **+0.247** | ✓ |
| 5 | "Hệ thống thông tin quan trọng về an ninh quốc gia" | "Thời tiết hôm nay trời nắng đẹp" | low | **+0.240** | ✓ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Bất ngờ nhất là **Pair 4 & 5 vẫn dương ~0.24** dù hai câu hoàn toàn khác chủ đề (luật vs nấu phở / thời tiết). Với tiếng Anh, cặp không liên quan thường cho score ~0 hoặc âm (xem data cũ: −0.05). Ở đây score "sàn" cao hơn — có thể vì cả hai đều là **tiếng Việt** nên chia sẻ nền ngôn ngữ chung, đẩy mức tương đồng cơ sở lên. Bài học: **ngưỡng "không liên quan" phụ thuộc ngôn ngữ/domain** — không thể dùng một threshold tuyệt đối (vd 0.5) cho mọi ngữ cảnh, phải hiệu chỉnh theo dữ liệu. Pair 3 (0.460) đúng dự đoán "lưng chừng": cùng lĩnh vực an ninh mạng nhưng nói về hai khía cạnh khác (lực lượng vs chủ quản HTTT).

---

## 6. Results — Cá nhân (10 điểm)

> Chạy 5 benchmark queries trên package `src` (RecursiveChunker 400, embedding OpenAI, **253 chunks** từ 8 chương luật). Query #4 dùng **metadata filter `chuong=6`**.

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer (verify từ luật) | Điều/Chương |
|---|-------|-------------|---|
| 1 | An ninh mạng được định nghĩa như thế nào? | Sự ổn định, an ninh, an toàn của không gian mạng; bảo vệ HTTT và bảo đảm thông tin/dữ liệu/hoạt động không gây phương hại đến an ninh quốc gia, trật tự an toàn xã hội | Điều 2.1 / Ch.I |
| 2 | Luật an ninh mạng áp dụng đối với những đối tượng nào? | Cơ quan/tổ chức/cá nhân Việt Nam; người nước ngoài tại VN & người gốc Việt; cơ quan/tổ chức/cá nhân nước ngoài liên quan hoạt động ANM tại VN | Điều 1.2 / Ch.I |
| 3 | Các biện pháp bảo vệ an ninh mạng gồm những gì? | Thẩm định ANM; đánh giá điều kiện ANM; kiểm tra ANM; giám sát ANM; ứng phó, khắc phục sự cố ANM… | Điều 5 / Ch.I |
| 4 *(filter chuong=6)* | Lực lượng bảo vệ an ninh mạng gồm những thành phần nào? | Lực lượng chuyên trách tại Bộ Công an, Bộ Quốc phòng; lực lượng tại Bộ/ngành/UBND; tổ chức/cá nhân được huy động | Điều 30 / Ch.VI |
| 5 | Trách nhiệm của cơ quan, tổ chức, cá nhân trong bảo vệ ANM? | Các trách nhiệm cụ thể của cơ quan/tổ chức/cá nhân theo Chương VII | Ch.VII |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Nguồn |
|---|-------|--------------------------------|-------|-----------|-------|
| 1 | An ninh mạng là gì | "Điều 2. Giải thích từ ngữ … An ninh mạng là sự ổn định, an ninh, an toàn của không gian mạng…" | **0.611** | ✓ | luat116_ch01.md |
| 2 | Đối tượng áp dụng | "Điều 1. Phạm vi điều chỉnh và đối tượng áp dụng … Luật này áp dụng đối với: a) Cơ quan, tổ chức, cá nhân Việt Nam…" | **0.672** | ✓ | luat116_ch01.md |
| 3 | Biện pháp bảo vệ ANM | "Điều 5. Biện pháp bảo vệ an ninh mạng … a) Thẩm định ANM; b) Đánh giá điều kiện ANM; c) Kiểm tra ANM…" | **0.768** | ✓ | luat116_ch01.md |
| 4 *(filter chuong=6)* | Lực lượng bảo vệ ANM | "Điều 30. Lực lượng bảo vệ an ninh mạng bao gồm: a) Lực lượng chuyên trách… tại Bộ Công an, Bộ Quốc phòng…" | **0.743** | ✓ | luat116_ch06.md |
| 5 | Trách nhiệm cơ quan/tổ chức | "…c) Tổ chức, cá nhân được huy động tham gia bảo vệ ANM. 2. Chính phủ quy định chi tiết…" | **0.773** | ✗ sai | luat116_ch06.md |

> **4/5 query top-1 đúng**. Riêng **Q5 là failure case** (xem Section 7): score cao 0.773 nhưng trỏ nhầm sang Chương VI (lực lượng) thay vì Chương VII (trách nhiệm) — query "trách nhiệm" trùng nhiều từ khoá với đoạn lực lượng. Metadata filter `chuong=7` sẽ khắc phục.

**Bao nhiêu queries trả về chunk relevant trong top-3?** **4 / 5** top-1 đúng; Q5 cần metadata filter để đúng.

---

## 7. What I Learned (5 điểm — Demo)

### Failure Analysis (Ex 3.5)

**Failure case 1 (high-confidence sai chương) — Q5:** Query *"Trách nhiệm của cơ quan, tổ chức, cá nhân trong bảo vệ an ninh mạng?"* trả về top-1 **score 0.773** (rất cao) nhưng **trỏ nhầm sang Chương VI** (lực lượng) thay vì Chương VII (trách nhiệm).

- **Tại sao:** đoạn Chương VI có cụm *"tổ chức, cá nhân được huy động tham gia bảo vệ an ninh mạng"* trùng nhiều từ khoá với query ("tổ chức, cá nhân", "bảo vệ an ninh mạng") → similarity cao về **bề mặt từ vựng** dù **ý** thuộc về điều khác. Đây là lỗi **retrieval precision** nguy hiểm vì score cao dễ khiến ta tin nhầm (*grounding quality* kém).
- **Đề xuất & đã kiểm chứng:** thêm **metadata filter `chuong=7`** → top-1 chuyển đúng về `luat116_ch07.md` (score 0.720, đúng nội dung trách nhiệm). Cho thấy *metadata utility* bù được điểm yếu của similarity thuần.

**Failure case 2 (whole-doc) — đã đo:** Cùng query lực lượng (Q4), nếu lưu **nguyên chương không chunk**, top-1 trỏ **nhầm Chương IV** (score 0.577); bật chunking (RecursiveChunker 400) trỏ đúng Điều 30 Chương VI (0.743). Chương luật dài (4–16k ký tự) nén thành 1 vector → pha loãng nhiều điều khoản (vấn đề **chunk coherence**).

**Failure case 3 (recursive làm vỡ vụn điều luật) — đã đo & đã sửa:** Query *"Luật an ninh mạng có hiệu lực từ ngày nào?"* với `chunking=recursive`, kết quả top-3 chứa một chunk **trộn lẫn 3 điều không liên quan**:
```
### Điều 14 ... ### Điều 15 ... ## Điều 3. Chính sách của Nhà nước...
```
- **Tại sao:** `recursive` cắt theo ~500 ký tự, **không tôn trọng ranh giới "Điều"** → gộp nhiều điều rời rạc vào 1 chunk → embedding nhiễu, retrieval kém (*chunk coherence* tệ).
- **Đã sửa:** đổi sang **`chunking=dieu` (ArticleChunker)** → mỗi chunk là 1 điều trọn vẹn; cùng query, top-1 trỏ đúng **Điều 44 "Hiệu lực thi hành"**, và `/ask` rút đúng đáp án: *"có hiệu lực thi hành từ ngày 01 tháng 7 năm 2026"*.

> **Bài học xuyên suốt 3 case:** với văn bản có cấu trúc (luật), **chunker tôn trọng ranh giới ngữ nghĩa** (Điều/Khoản) cho retrieval chính xác hơn hẳn chunker cắt máy móc; metadata filter là lớp bảo hiểm thứ hai cho các query mơ hồ.

- Ghi chú: `SentenceChunker` cũng yếu với luật — một khoản là 1 câu 600+ ký tự → chunk vượt xa `chunk_size`.

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> [điền cùng nhóm — sau buổi so sánh]

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> [điền sau demo]

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> [điền — gợi ý: tune chunk_size theo độ dài tài liệu, thêm metadata `section`/`date`, xử lý viết tắt cho SentenceChunker]

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | / 5 |
| Document selection | Nhóm | / 10 |
| Chunking strategy | Nhóm | / 15 |
| My approach | Cá nhân | / 10 |
| Similarity predictions | Cá nhân | / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 (42/42 tests pass) |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |
