# Phân tích các điểm sáng cốt lõi của Hệ thống Matching CV - Job

Tài liệu này tổng hợp các kỹ thuật đặc biệt được tìm thấy trong các file core (`NCKH.ipynb`, `Service_Recom.py`, `Milvus_Load.py`) nhằm nâng cao độ chính xác và tính thực tế cho hệ thống gợi ý việc làm.

## 1. Tìm kiếm Đa Vector (Multi-Vector Semantic Search)
Thay vì sử dụng một vector duy nhất cho toàn bộ tin tuyển dụng, hệ thống tách biệt thành 3 trường vector:
- **`title_vec`**: Tập trung vào chức danh công việc.
- **`tech_vec`**: Tập trung vào kỹ năng cứng và công nghệ.
- **`mota_vec`**: Tập trung vào mô tả chi tiết, trách nhiệm và quyền lợi.

> [!TIP]
> **Lợi ích:** Cho phép áp dụng các trọng số khác nhau cho từng phần của công việc, giúp điều chỉnh độ ưu tiên linh hoạt giữa "Kỹ năng" và "Vị trí".

## 2. Ma trận Khoảng cách Địa lý (Location Distance Matrix)
Hệ thống không chỉ lọc theo tên thành phố mà sử dụng một ma trận điểm số giữa các tỉnh lân cận (Hưng Yên, Hà Nội, Hải Dương, v.v.).
- Khớp chính xác: 1.0 điểm.
- Tỉnh lân cận (ví dụ Hà Nội - Hưng Yên): 0.92 điểm.
- Khác miền: 0.2 điểm.

> [!IMPORTANT]
> Đây là điểm cực kỳ thực tế cho thị trường Việt Nam, giúp gợi ý những công việc khả thi về mặt đi lại cho ứng viên.

## 3. Hàm Suy giảm Điểm Kinh nghiệm (Experience Decay Function)
Sử dụng logic "Soft Filter" thay vì loại bỏ thẳng tay các ứng viên thiếu kinh nghiệm:
- Nếu kinh nghiệm ứng viên $\ge$ yêu cầu: 1.0 điểm.
- Nếu thấp hơn: Điểm số giảm dần theo hàm số (Decay) dựa trên khoảng cách thiếu hụt, nhưng vẫn giữ một mức điểm sàn tối thiểu.

## 4. Mô hình Embedding Hiện đại (BGE-M3)
Sử dụng model `BAAI/bge-m3`, một trong những mô hình tốt nhất cho bài toán Retrieval đa ngôn ngữ:
- Hiểu sâu ngữ cảnh tiếng Việt.
- Xử lý tốt các từ khóa tiếng Anh chuyên ngành IT trộn lẫn trong văn bản tiếng Việt.

## 5. Quy trình xử lý dữ liệu (Data Cleaning Pipeline)
- **Loại bỏ HTML:** Sử dụng `BeautifulSoup` để làm sạch dữ liệu crawl từ web.
- **Chuẩn hóa kinh nghiệm:** Regex parser thông minh để chuyển đổi các chuỗi văn bản tự do thành khoảng giá trị `exp_min`, `exp_max`.

---
*Tài liệu này phục vụ cho việc nâng cấp hệ thống lên phiên bản Production.*
