# Core Concept: Thuật toán Matching CV - Job

## Mục tiêu
Tạo một điểm số tương thích cho mỗi job được đề xuất, kết hợp giữa:
- độ giống ngữ nghĩa (semantic similarity) từ vector search,
- và các yếu tố thực tế quan trọng như địa điểm và kinh nghiệm.

## Thành phần điểm
- `S_title`: similarity giữa chức danh/position trên CV và title của job.
- `S_tech`: similarity giữa kỹ năng CV và kỹ năng job.
- `S_mota`: similarity giữa phần mô tả CV và phần mô tả công việc.
- `S_loc`: điểm vị trí, đánh giá mức độ gần/xa giữa địa điểm ứng viên và job.
- `S_exp`: điểm kinh nghiệm, đánh giá mức độ đủ/thiếu năm kinh nghiệm so với yêu cầu job.

## Công thức chấm điểm
Công thức tổng hợp như sau:

```math
\text{Final Score} = W_{\text{title}} \cdot S_{\text{title}} + W_{\text{tech}} \cdot S_{\text{tech}} + W_{\text{mota}} \cdot S_{\text{mota}} + W_{\text{loc}} \cdot S_{\text{loc}} + W_{\text{exp}} \cdot S_{\text{exp}}
```

- Mỗi `S_*` nằm trong khoảng `[0,1]`.
- Mỗi `W_*` là trọng số cho biết yếu tố nào quan trọng hơn.
- Tổng `Final Score` sau khi tính được chuẩn hoá về `[0,1]` và nhân 100 để hiển thị phần trăm.

## Ví dụ minh hoạ
Giả sử trọng số:
- `W_title = 0.2`
- `W_tech = 0.4`
- `W_mota = 0.2`
- `W_loc = 0.1`
- `W_exp = 0.1`

Và điểm con của một candidate:
- `S_title = 0.8`
- `S_tech = 0.6`
- `S_mota = 0.7`
- `S_loc = 0.9`
- `S_exp = 1.0`

Khi đó:
- `Final Score = 0.2*0.8 + 0.4*0.6 + 0.2*0.7 + 0.1*0.9 + 0.1*1.0 = 0.73`
- Tương đương `73%` phù hợp.

## Vì sao không thể dùng kết quả vector search trực tiếp?
- Milvus chỉ trả về job có nội dung hơi giống CV, nhưng không đánh giá chính xác:
  - job đó có yêu cầu bao nhiêu năm kinh nghiệm?
  - job đó có gần nơi ứng viên không?
  - job đó có phù hợp với kỹ năng chính của ứng viên không?
- Do đó, hệ thống cần bước chấm điểm bổ sung để đưa ra đề xuất thực tế hơn.

## Lưu ý kỹ thuật
- Với Milvus COSINE, client có thể trả về giá trị `distance` thay vì `similarity`.
- Trong trường hợp đó, phải chuyển sang similarity bằng:
  - `S = 1 - distance`
- Sau khi chuyển, các giá trị `S_*` mới có thể gộp được trong công thức.

## Kết luận
Phần ``Weighted Scoring`` giúp dịch chuyển kết quả từ “đúng ngữ nghĩa” sang “phù hợp thực tế”.
Nó cho phép hệ thống vừa giữ được lợi thế của vector search, vừa bổ sung thông tin về vị trí, kinh nghiệm và kỹ năng để trả về đề xuất chuẩn hơn.
