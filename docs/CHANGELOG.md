# Nhật ký hoàn thiện ngày 12 tháng 10 năm 2026

- Bảo toàn bert.ipynb, raw/legacy datasets, label_map và checkpoint gốc. Backup app/template cũ trong docs/legacy.
- Refactor ứng dụng, sửa path model, eval/inference_mode, offset decoding, long-text windows, validation và tách hai form.
- Thêm JSON APIs và recommendation sparse có unknown/missing handling, deduplicate results và không trả zero similarity.
- Thêm duplicate-safe full-context corpus preparation, SHA256 manifest, training CLI với validation selection và evaluation strict entity/token/baseline.
- Đọc toàn bộ text notebook/code/config/vocabulary, profile CSV/tensor artifacts và kiểm tra các tensor shapes trong safetensors. Git internals chỉ đọc status/config/log cần thiết, không sửa lịch sử.
- Thay eda.ipynb trống bằng notebook đã execute và figures; bổ sung requirements, README, data/model cards, architecture, technical report, evaluation protocol, report corrections và defense guide.
- Tái lập toàn bộ 3.000 legacy test sequences. Chạy 20 automated tests gồm real checkpoint, service smoke thật và two-step training continuation smoke.
- Chuẩn bị 50 annotation candidates; chưa nhận là gold. Chưa full retrain trên corpus mới, chưa deployment hoặc đo business impact.
- Phần mới là portfolio supplement năm 2026; CV phải thể hiện thời gian 2025–2026 và mô tả các tính năng mới có thể giải thích được.
