# Quy trình đánh giá và gán nhãn

## Ba loại kết quả cần phân biệt

1. **Historical reproduction:** checkpoint gốc với split random_state 42 của data_preprocessed.pkl, 3.000 test sequences. Dữ liệu đã xóa nhiều token O và có nhãn tự động. Đây là phép tái lập notebook, không phải chứng minh khả năng tổng quát hóa trên mô tả gốc.
2. **Full-context silver diagnostic:** test.jsonl mới với câu nguyên vẹn và mô tả đã loại trùng. Chạy checkpoint cũ ở đây chỉ là chẩn đoán vì nhiều mô tả có thể đã nằm trong tập train cũ. Chạy checkpoint mới từ pretrained BERT mới tạo ra phép held-out silver có thể diễn giải; vẫn không phải gold.
3. **Independent gold:** người đánh giá sửa trực tiếp entity spans trên câu nguyên vẹn, không xem prediction của model. Chưa có kết quả loại này. Không gọi các candidate labels tự động là gold.

## Định nghĩa metric

Weighted token F1 tính theo từng nhãn BIO và trọng số support; bao gồm O. Các subword không tương ứng nhãn hợp lệ và padding -100 bị loại. Một điểm cao không đảm bảo đúng ranh giới entity.

Exact entity micro F1: entity đúng khi type, start và end trùng khớp hoàn toàn. TP là span khớp, FP là span dự đoán thừa/sai, FN là span bị thiếu. Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2PR/(P+R). O không đóng góp vào entity score. Strict IOB2 không coi orphan I là bắt đầu entity. Báo cáo riêng Color, Material, Dimensions, Weight và macro F1 bốn loại.

Với phép tái lập cũ, start/end là chỉ số **subword của câu đã rút ngắn**. Không được gọi 0.907216 là entity F1 trên mô tả nguyên vẹn. Inference UI có thể sửa orphan I thành entity để hiển thị, nhưng evaluator strict không dùng sửa này để tăng điểm.

## Thiết kế tập gold

artifacts/annotation_candidates.jsonl chứa 50 mô tả thuộc test mới và proposed spans. Tất cả có review_status pending. Đây là vật liệu để gán nhãn, không phải benchmark đã xác minh. Người gán nhãn nên dùng bản text, tự đánh dấu span trước rồi mới đối chiếu proposed spans; nếu cần đánh giá mù, loại proposed_entities khỏi file trước khi gửi cho người gán nhãn.

- Color: đánh dấu từ chỉ màu của sản phẩm; không dùng màu trong một ví dụ không mô tả sản phẩm hoặc phủ định rõ ràng.
- Material: đánh dấu vật liệu cấu thành sản phẩm. Không tự suy từ loại sản phẩm hoặc thêm vật liệu không có trong câu.
- Dimensions: đánh dấu biểu thức kích thước kèm đơn vị hoặc chiều được nêu rõ; phân biệt kích thước với model number.
- Weight: phải là khối lượng, ưu tiên có đơn vị; không biến pack of 3 thành 3 kg.
- Dùng offset start inclusive, end exclusive trên chuỗi text gốc. Bắt đầu/kết thúc chính xác; không thêm dấu câu ngoài thực thể.
- Gán nhãn ít nhất hai vòng hoặc hai người nếu có điều kiện; ghi disagreements và quyết định adjudication. Ghi ngày, người kiểm tra, hướng dẫn phiên bản và hash dữ liệu.
- Test phải chưa được dùng để sửa regex, lựa chọn tham số hoặc fine-tune. Cần checkpoint mới train từ pretrained base và split được đóng băng để loại ảnh hưởng model cũ.

## Baseline và phân tích lỗi

Rule baseline trong workflow dùng màu/vật liệu theo từ điển và regex đơn vị/kích thước. Nó có chung một phần logic với bộ tạo nhãn silver; kết quả không độc lập. So sánh BiLSTM-CRF, CRF hoặc TextCNN chỉ được thêm khi có implementation, run config và predictions thực tế trên cùng protocol.

Lưu metrics.json, rule_baseline.json, predictions.jsonl, checkpoint/data hashes, seed, phiên bản package, thời gian và biểu đồ. Xem false positives ở Dimensions/Weight, entity boundary, multiword materials, phủ định, đổi ngôn ngữ và catalog unknown values. Không chọn checkpoint theo test F1.

Tài liệu phương pháp: [Hugging Face token classification](https://huggingface.co/docs/transformers/tasks/token_classification), [scikit-learn data leakage](https://scikit-learn.org/stable/common_pitfalls.html).
