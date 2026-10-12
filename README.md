# EntityExtract

Đồ án tốt nghiệp của **Dao Ngoc Binh Thien**, Trường Đại học Đại Nam: sử dụng BERT cho bài toán phân loại token để trích xuất **màu sắc (Color), vật liệu (Material), kích thước (Dimensions) và khối lượng (Weight)** từ mô tả sản phẩm thương mại điện tử bằng tiếng Anh. Dự án có ứng dụng minh họa bằng Flask và chức năng gợi ý sản phẩm theo thuộc tính.

Đồ án gốc được thực hiện năm 2025. Phần bổ sung kỹ thuật và kiểm chứng khả năng tái lập được thực hiện ngày 12/10/2026. Notebook, các bộ dữ liệu và checkpoint gốc được giữ nguyên; các phần bổ sung được trình bày trong [báo cáo kỹ thuật](docs/THESIS_TECHNICAL_REPORT.md).

## Chạy ứng dụng trên máy

Mở PowerShell tại thư mục gốc của repository và chạy:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
git lfs pull
.\.venv\Scripts\python.exe app.py
```

Truy cập http://127.0.0.1:5000. Chức năng trích xuất và gợi ý sản phẩm sử dụng hai biểu mẫu riêng. Bạn có thể nhập trực tiếp thuộc tính để nhận gợi ý mà không cần nhập mô tả vào biểu mẫu trích xuất. Khi checkpoint đi kèm đã được tải qua Git LFS, quá trình suy luận không cần tải thêm mô hình từ bên ngoài.

Môi trường đã được kiểm tra: Python 3.13.2 trên Windows, PyTorch 2.6 bản CPU và Transformers 5.18.0. Phiên bản cụ thể của các thư viện chính nằm trong requirements.txt. Máy có GPU, nhưng bản PyTorch dùng để kiểm tra chỉ hỗ trợ CPU. Việc cài bản hỗ trợ CUDA là tùy chọn và chưa được thực hiện trong lần bổ sung này.

Có thể cấu hình các biến môi trường ENTITYEXTRACT_MODEL_DIR, ENTITYEXTRACT_CATALOG_PATH, ENTITYEXTRACT_DEVICE và PORT. Đường dẫn mặc định được xác định theo vị trí của app.py, nên ứng dụng có thể chạy từ thư mục làm việc khác. Chế độ debug được tắt và ứng dụng chỉ lắng nghe trên localhost. Đây là bản minh họa chạy trên máy, chưa phải hệ thống đã triển khai trong môi trường production.

## Chạy lại quy trình và tạo các kết quả

```powershell
python -m entityextract audit
python -m entityextract prepare
python -m entityextract evaluate --protocol legacy
$env:RUN_MODEL_TESTS='1'
python -m pytest -q -p no:cacheprovider -p no:tmpdir
```

Quy trình chuẩn bị dữ liệu giữ nguyên mô tả đầy đủ và loại bỏ dữ liệu trùng trước khi chia tập theo tỷ lệ 70/15/15 với kết quả xác định theo seed. File `artifacts/prepared/manifest.json` ghi mã băm dữ liệu đầu vào, kích thước từng tập và seed. Nhãn của bộ dữ liệu này là **nhãn silver**: được tạo tự động bằng quy tắc và các giá trị có trong thông số sản phẩm, không phải nhãn gold được con người gán và kiểm tra độc lập.

Để huấn luyện một mô hình mới từ BERT đã được tiền huấn luyện, chạy:

```powershell
python -m entityextract train --output artifacts/trained-model --epochs 3
python -m entityextract evaluate --model artifacts/trained-model --protocol full-context-diagnostic --prepared artifacts/prepared --output artifacts/new-evaluation
```

Lệnh đầu có thể tải bert-base-uncased và cần nhiều thời gian xử lý bằng CPU hoặc một môi trường GPU tương thích. Quy trình chỉ tính trọng số lớp từ tập train và chọn checkpoint theo entity F1 trên tập validation; không dùng tập test để lựa chọn. **Chưa thực hiện huấn luyện lại đầy đủ** trên bộ dữ liệu mới trong lần bổ sung này. CLI đã được kiểm tra bằng một lần chạy thử huấn luyện tiếp hai bước trong artifacts/smoke-model; kết quả chạy thử đó không phải kết quả đánh giá hiệu năng mô hình.

## Kết quả đã kiểm chứng

| Chỉ số đánh giá | BERT | Mô hình cơ sở dùng quy tắc |
| --- | ---: | ---: |
| F1 theo token, trung bình có trọng số | 0.965502 | 0.920067 |
| F1 micro theo thực thể với ranh giới khớp chính xác và BIO hợp lệ | 0.907216 | 0.765786 |

Các kết quả trên được chạy lại trên toàn bộ **3.000 chuỗi thuộc tập test cũ**, với 67.494 vị trí subword được đánh giá. Ranh giới thực thể được tính trên các chuỗi subword đã rút ngắn mà notebook gốc sử dụng. Nhãn tham chiếu được tạo tự động; bộ dữ liệu cũ đã loại bỏ nhiều token O và còn có dữ liệu trùng giữa các tập. Đây là kết quả tái lập, **không phải hiệu năng trên mô tả đầy đủ chưa từng được mô hình nhìn thấy và có nhãn gold do người kiểm tra**. Mô hình cơ sở cũng dùng một phần quy tắc tương tự bộ tạo nhãn tham chiếu, nên phép so sánh này không phải đánh giá độc lập trên nhãn gold.

Dữ liệu gốc có 20.000 dòng, trong đó hai dòng thiếu mô tả. Sau khi loại bỏ 1.807 mô tả không thiếu nhưng bị trùng, còn **18.191 mô tả đầy đủ không trùng**, được chia thành 12.733 mẫu train, 2.728 mẫu validation và 2.730 mẫu test. Checkpoint hiện có chưa được huấn luyện lại trên cách chia tập mới này.

## Các endpoint trả về JSON

- GET /health: kiểm tra trạng thái tiến trình và cho biết các dịch vụ đã được tải hay chưa; không xác nhận mô hình đã sẵn sàng suy luận.
- POST /api/extract: nhận một đối tượng JSON có trường description là chuỗi; trả về thuộc tính, vị trí thực thể trong văn bản gốc và thông tin về phạm vi xử lý của các cửa sổ.
- POST /api/recommend: nhận một đối tượng JSON có trường attributes, trường description tùy chọn và top_k từ 1 đến 20.

Ví dụ yêu cầu trích xuất. Giữ mô tả bằng tiếng Anh vì mô hình được huấn luyện cho dữ liệu tiếng Anh:

```json
{"description":"This shirt is 30 mm x 50 mm with red color and weight 2 kg."}
```

Ví dụ yêu cầu gợi ý sản phẩm:

```json
{"attributes":{"color":"black"},"top_k":3}
```

Đầu vào không hợp lệ trả về mã 400, nội dung yêu cầu quá lớn trả về 413, dịch vụ trên máy không khả dụng trả về 503. Giá trị confidence là trung bình xác suất dự đoán của các subword và chưa được hiệu chỉnh xác suất. Chức năng gợi ý dùng thuộc tính được mã hóa one-hot dạng thưa và độ tương đồng cosine; không dùng BERT embeddings. Chưa có phép đo chất lượng gợi ý sản phẩm. Giá trị chưa biết và trường trống không được tính thành các lượt khớp giả.

## Cấu trúc repository

- bert.ipynb: notebook huấn luyện gốc và các kết quả cũ, được giữ nguyên.
- eda.ipynb: notebook phân tích dữ liệu đã chạy, có biểu đồ.
- entityextract/: xử lý dữ liệu, suy luận, gợi ý sản phẩm, các chỉ số đánh giá và quy trình CLI.
- app.py và templates/: ứng dụng chạy trên máy và API JSON.
- tests/: 20 kiểm thử, bao gồm suy luận với checkpoint thật, phạm vi xử lý văn bản dài, ranh giới thực thể, kiểm tra đầu vào và hành vi gợi ý sản phẩm.
- artifacts/: mã băm phục vụ kiểm tra, kết quả thực nghiệm, biểu đồ, bằng chứng chạy thử và các mẫu đề xuất để gán nhãn.
- docs/: báo cáo kỹ thuật, tài liệu mô hình và dữ liệu, kiến trúc, các điểm cần sửa trong báo cáo cũ, hướng dẫn bảo vệ và nhật ký thay đổi.

Đọc [quy trình đánh giá](docs/EVALUATION_PROTOCOL.md) trước khi diễn giải các chỉ số. Bước kiểm chứng khoa học tiếp theo là xây dựng tập gold được kiểm tra độc lập theo hướng dẫn gán nhãn đã cung cấp. Nguồn dữ liệu phù hợp với bộ Flipkart/PromptCloud được mô tả trong đồ án gốc; mã băm file trên máy đã được ghi lại, nhưng chưa xác minh file khớp từng byte với bản từ nguồn hoặc giấy phép phân phối lại. Chưa tự gán giấy phép cho mã nguồn của ứng viên hay dữ liệu của bên thứ ba.
