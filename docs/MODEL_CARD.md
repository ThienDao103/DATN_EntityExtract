# Model card

Model: bert-base-uncased encoder with BertForTokenClassification and nine BIO labels. Intended use: English product descriptions, local academic prototype. Owner-provided checkpoint remains unchanged. Parameter count verified during loading: 108,898,569. Inference loads local files without trust_remote_code.

Checkpoint SHA256: 4b5d506e5f574520e0e078b4474b0ad4eaf9b92d2e2ddccc788195a07711aecd

Original notebook configuration: 3 epochs, batch 8, AdamW learning rate 2e-5, max length 128, class-weighted cross-entropy. Training examples 13,995; validation 2,999; test 3,000. New run uses model.eval, inference_mode and explicit overlapping windows for user inputs. Confidence is not calibrated.

| Historical test reproduction | Value |
| --- | ---: |
| Test sequences | 3,000 |
| Evaluated subwords | 67,494 |
| Weighted token precision | 0.970077 |
| Weighted token recall | 0.964071 |
| Weighted token F1 | 0.965502 |
| Strict subword entity micro F1 | 0.907216 |
| Strict entity macro F1 | 0.858847 |
| Rule baseline entity micro F1 | 0.765786 |
| Invalid predicted I tags | 263 |

Important: metrics are on shortened, automatically labeled sequences, with historical duplicate overlap. They do not establish real-world performance on full descriptions, Vietnamese text, independent human gold, negation or unseen material vocabularies. Do not claim production accuracy or a business KPI.

Training pipeline bổ sung lựa chọn checkpoint theo validation exact entity F1, class weights chỉ từ train, seed/hashes/config logs. Đã chạy hai bước smoke continuation từ checkpoint cũ để xác minh forward/backward/save; không huấn luyện lại đầy đủ và không lấy smoke F1 làm kết quả nghiên cứu. Phép đánh giá khoa học tiếp theo cần pretrained base mới và gold set độc lập.
