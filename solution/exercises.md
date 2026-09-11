# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature = 0.0, phản hồi mang tính tất định (deterministic), tập trung vào các sự thật phổ biến nhất với văn phong chuẩn mực và hoàn toàn giống nhau nếu gọi lại. Khi tăng temperature lên 0.5 - 1.0, câu trả lời đa dạng và giàu hình ảnh hơn nhưng vẫn giữ tính chính xác. Ở mức 1.5, mô hình dùng từ ngữ bay bổng, bất ngờ hơn nhưng bắt đầu có dấu hiệu lan man và tăng nguy cơ sinh thông tin sai lệch (hallucination).

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature thấp, trong khoảng từ 0.0 đến 0.2 (tối đa 0.3). Vì chatbot chăm sóc khách hàng ưu tiên tính chính xác tuyệt đối về chính sách, giá cả và thông tin hỗ trợ; mức temperature thấp giúp câu trả lời nhất quán, bám sát tài liệu nguồn và triệt tiêu nguy cơ mô hình tự "bịa" thông tin (hallucination).

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Với workload này (10,5 triệu token output/ngày), GPT-4o đắt hơn GPT-4o-mini khoảng 16.7 lần (tương đương ~$105/ngày so với ~$6.30/ngày).

Nên dùng GPT-4o: Các tác vụ phức tạp đòi hỏi suy luận sâu (deep reasoning), phân tích văn bản pháp lý/hợp đồng, hoặc sinh code hệ thống phức tạp nơi sai sót nhỏ gây tổn thất lớn.
Nên dùng GPT-4o-mini: Chatbot CSKH giải đáp FAQ thông thường, phân loại ý định người dùng (intent classification), tóm tắt tin nhắn ngắn hoặc các tác vụ xử lý lưu lượng lớn lặp đi lặp lại.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Bản dành cho học sinh tiểu học có độ dài ngắn, dùng từ ngữ mộc mạc và đưa ra hình ảnh so sánh trực quan (như cuốn sổ ghi chép chung cả lớp cùng xem và không ai xé được). Ngược lại, bản chuyên gia tài chính dùng nhiều thuật ngữ chuyên sâu (sổ cái phân tán DLT, mật mã học, cơ chế đồng thuận PoW/PoS, tính bất biến). System prompt đóng vai trò định hình không gian ngữ cảnh (conditioning), giúp nắn chỉnh phân phối xác suất token của mô hình để thích ứng chính xác với đối tượng độc giả mục tiêu và văn phong yêu cầu.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với đoạn văn tiếng Việt 100 từ, công thức ước lượng 100 / 0.75 cho ra khoảng 133 tokens, trong khi tiktoken đếm thực tế thường dao động từ 160 – 200 tokens (chênh lệch từ 20% đến 50%). Tiếng Việt tốn nhiều token hơn tiếng Anh vì bộ từ vựng của tokenizer (BPE) được huấn luyện chủ yếu trên tiếng Anh; các từ và âm tiết tiếng Việt có dấu thanh (ký tự Unicode nhiều byte) thường không có sẵn trong từ điển token nguyên từ mà bị băm nhỏ thành nhiều sub-words hoặc byte riêng rẽ.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các giao diện tương tác trực tiếp với người dùng (chatbot UI, trợ lý dòng lệnh CLI, công cụ sinh code), giúp giảm thời gian phản hồi ban đầu (Time-To-First-Token - TTFT); người dùng thấy chữ xuất hiện tức thì sau ~0.5s thay vì phải chờ cả câu trả lời hoàn tất. Ngược lại, non-streaming phù hợp hơn cho các tác vụ xử lý ngầm trong pipeline (batch processing, cron job), hoặc khi cần gọi API ở chế độ trả về dữ liệu có cấu trúc (như Tool Calling / JSON Mode) để parse toàn bộ đối tượng trước khi xử lý tiếp.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Nếu hàng nghìn client cùng retry với delay cố định (ví dụ đúng 1 giây), toàn bộ các request thất bại sẽ đồng loạt dội ngược lại server tại cùng một thời điểm, gây ra hiện tượng Thundering Herd (đàn trâu giẫm đạp) khiến server tiếp tục sụp đổ và không thể hồi phục. Exponential backoff giải quyết vấn đề này bằng cách kéo giãn thời gian chờ tăng dần theo cấp số nhân ($0.1s \to 0.2s \to 0.4s \dots$), giúp dàn trải mật độ request theo thời gian và tạo khoảng thời gian cần thiết để server xả tải và hồi phục tài nguyên.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> System Prompt: "Bạn là trợ giảng lập trình AI thân thiện, trả lời ngắn gọn bằng tiếng Việt và ưu tiên đưa ví dụ code minh họa dễ hiểu." Giải thích lựa chọn từ ngữ:

"trả lời ngắn gọn": Giúp kiểm soát chi phí token đầu ra và giảm độ trễ phản hồi, đặc biệt phù hợp khi hiển thị trên giao diện dòng lệnh (terminal) tránh tràn màn hình.
"bằng tiếng Việt": Cố định ngôn ngữ đầu ra, tránh trường hợp mô hình tự động chuyển sang tiếng Anh khi gặp các thuật ngữ kỹ thuật trong câu hỏi.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất: Bộ nhớ ngắn hạn cố định 3 lượt (history[-6:]) khiến trợ lý nhanh chóng quên ngữ cảnh quan trọng nếu cuộc trò chuyện kéo dài, đồng thời không có khả năng lưu trữ phiên làm việc khi đóng terminal. Cải thiện đề xuất: Triển khai cơ chế Tóm tắt ngữ cảnh (Context Summarization).

Cách triển khai: Khi lịch sử vượt quá 3 lượt, thay vì cắt bỏ trực tiếp các tin nhắn cũ, ta gọi một model nhỏ (như gpt-4o-mini) để tóm tắt các lượt trao đổi cũ thành một đoạn summary ngắn gọn và nhúng đoạn tóm tắt này vào đầu danh sách messages. Nhờ đó, trợ lý vừa duy trì được ngữ cảnh toàn diện, vừa giữ số lượng token đầu vào ở mức thấp.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
