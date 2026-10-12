# Câu hỏi bảo vệ và phỏng vấn

1. **Bài toán là gì?** Chuyển mô tả sản phẩm tiếng Anh thành bốn nhóm thuộc tính để hỗ trợ tổ chức dữ liệu và gợi ý sản phẩm. Chưa đo được tác động đến doanh thu hoặc conversion.
2. **Tại sao dùng BERT?** Mô hình token classification dựa trên encoder pretrained; fine-tune head chín BIO labels. Đây là lựa chọn triển khai đã dùng, không có bằng chứng rằng nó tối ưu hơn mọi mô hình khác.
3. **Dữ liệu được gán nhãn thế nào?** Rules và specifications. Đây là weak supervision/silver, không phải 20.000 mẫu được người gán nhãn độc lập.
4. **BIO và WordPiece xử lý thế nào?** B mở đầu entity, I tiếp tục; subword đầu giữ nhãn, subword sau đổi B thành I; special/padding mang -100. Lưu map nhãn trong config.
5. **Hạn chế lớn của notebook cũ?** Xóa nhiều token O làm mất ngữ cảnh, chia ngẫu nhiên có duplicates, và weighted token F1 bao gồm O. Notebook vẫn được giữ để tái lập chứ không sửa rồi giả định kết quả cũ còn đúng.
6. **96.55% có ý nghĩa gì?** Weighted token F1 của 3.000 shortened silver sequences; không phải accuracy hoặc entity F1 trên dữ liệu thực tế độc lập.
7. **Entity F1 mới là bao nhiêu?** 0.907216 micro, 0.858847 macro, strict boundaries trên cùng subword sequences. Dimensions là nhóm yếu hơn. Xem protocol và output hashes trước khi trích con số.
8. **Class weights có gây leakage không?** Pipeline mới chỉ tính trên train. Validation dùng chọn checkpoint; test không dùng chọn tham số. Nhãn tự động vẫn có thiên lệch dù split không overlap.
9. **Chia dữ liệu mới cải thiện gì?** Loại 1.807 mô tả trùng sau khi bỏ hai mô tả thiếu, còn 18.191 unique descriptions. Giữ O và toàn câu; chưa retrain đầy đủ trên bộ này.
10. **Baseline nào thực sự chạy?** Regex/từ điển, trên cùng test cũ. Không tự nhận đã chạy CRF, BiLSTM-CRF hoặc TextCNN. Baseline có chung logic với weak labeler, cần gold set để so sánh độc lập.
11. **Flask được cải thiện gì?** Đường dẫn theo project/env, lazy load, validation, JSON API, hai form riêng, model.eval/inference_mode và xử lý mô tả dài.
12. **Recommendation có dùng BERT embeddings không?** Không. Nó dùng sparse one-hot trên thuộc tính và cosine similarity. Unknown category không tạo match; không có metric chất lượng recommendation được xác minh.
13. **Model confidence có đáng tin không?** Là mean subword probability chưa được calibration; không được gọi là xác suất đúng của entity.
14. **Đã kiểm thử gì?** Offset/WordPiece, entity boundaries, duplicate-safe splits, recommendation unknown/missing/duplicates, input validation, form independence, XSS escaping và inference thật trên mô tả dài. Kết quả 20 tests, thêm CLI train smoke hai bước.
15. **Còn gì cần làm về khoa học?** Gold annotations độc lập, train lại từ pretrained base trên split mới, đánh giá nhiều seed và phân tích lỗi trên text nguyên vẹn. Không biến kế hoạch thành thành tích.

Khi trình bày đóng góp, phân biệt công việc năm 2025 và phần bổ sung portfolio năm 2026. Hiểu và giải thích các thay đổi được hỗ trợ bởi AI trước khi đưa vào phỏng vấn; không tuyên bố kết quả chưa chạy hoặc quyền tác giả của tài liệu tham khảo.
