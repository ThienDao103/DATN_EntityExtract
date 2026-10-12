# Các điểm cần chỉnh trong báo cáo đồ án năm 2025

Đối chiếu notebook và bản báo cáo DATN_ThienDao.docx ở thư mục ThienDao bên cạnh. Bản báo cáo đó không bị ghi đè. Tài liệu này ghi rõ các mâu thuẫn để ứng viên sửa trước khi sử dụng lại báo cáo hoặc bảo vệ portfolio.

| Nội dung trong báo cáo | Bằng chứng đối chiếu | Cách diễn đạt chính xác |
| --- | --- | --- |
| F1 khoảng 93%, 94.2% hoặc 96.55% ở các đoạn khác nhau | Notebook gốc và phép chạy lại cho weighted token F1 0.965502 trên 3.000 shortened silver sequences | Chỉ ghi đúng protocol, metric, tập dữ liệu và run; không trộn số của các run hoặc gọi là entity F1 |
| BERT vượt BiLSTM-CRF, CRF, TextCNN với các con số cụ thể | Không có implementation/run của các baseline này trong EntityExtract | Loại các kết quả chưa có bằng chứng; dùng rule baseline mới chạy trên cùng split và nêu hạn chế silver |
| Có stratify khi train_test_split | Cell 9 chỉ dùng test_size và random_state, không dùng stratify | Mô tả đúng chia ngẫu nhiên 70/15/15 với seed 42; bổ sung kiểm tra phân phối nhãn thay vì khẳng định stratification |
| Test hoàn toàn độc lập và chưa tiếp xúc | Có exact shortened-sequence overlap train-test; mô tả gốc trùng nhiều hơn | Không khẳng định independent gold. Dùng pipeline loại trùng mới cho các run tương lai |
| 20.000 mẫu đều là train examples hoặc gold annotations | Raw CSV có 20.000 rows; 19.994 shortened labeled sequences; train 13.995 | Phân biệt raw rows, labeled sequences, train/validation/test và nguồn nhãn tự động |
| Dimensions 9.658/20.000 = 19.9%; Weight 8.965/20.000 = 18.5% | Các phép chia này không đúng; catalog hiện tại còn là một artifact khác bộ token labels | 9.658/20.000 = 48.29%, 8.965/20.000 = 44.825% nếu tử số thực sự đúng; ưu tiên số tính lại trong data card và ghi đúng đơn vị đếm |
| I-Weight đạt F1 trên 0.90 | Test report cell 10 ghi I-Weight khoảng 0.862408 | Dùng per-label table thực tế; không suy rộng từ điểm tổng hợp |
| CNN được gọi là mạng truyền thẳng | CNN là convolutional neural network | Sửa thuật ngữ thành mạng nơ-ron tích chập |
| Toàn bộ label data giữ nguyên ngữ cảnh sản phẩm | Cell 4 giữ ngẫu nhiên khoảng 20% token O | Nêu rõ đây là rút ngắn câu, có ảnh hưởng ngữ cảnh; pipeline mới không xóa O |

Tài liệu tham khảo phải có tác giả/tên/nguồn thật. Không giữ một nghiên cứu hoặc tỷ lệ so sánh chỉ vì đã có trong bản cũ. BERT có nguồn xác minh [Devlin et al. 2019](https://aclanthology.org/N19-1423/). Nguồn dữ liệu phù hợp với [Flipkart Products của PromptCloud](https://www.kaggle.com/PromptCloudHQ/flipkart-products/code); chưa kiểm tra byte-for-byte upstream hoặc giấy phép phân phối lại file local.

Các bổ sung thực hiện ngày 12/10/2026 là phần mở rộng portfolio sau đồ án. Không diễn đạt rằng toàn bộ API, tests và protocol mới đã nằm trong bản nộp năm 2025.
