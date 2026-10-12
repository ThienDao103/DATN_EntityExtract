# Data card

Nguồn local: data.csv, các mô tả tiếng Anh và thông số sản phẩm Flipkart. Báo cáo gốc nói nguồn Kaggle/PromptCloud. Metadata và số dòng phù hợp nguồn [Flipkart Products](https://www.kaggle.com/PromptCloudHQ/flipkart-products/code), nhưng chưa kiểm tra byte-for-byte upstream hoặc quyền phân phối lại. Không tải lại hoặc công bố dữ liệu trong lần bổ sung này.

SHA256 raw CSV: b896acede72e2ae9ab8f9acfc655f191d47477a99066490bd35b5f1a6283bb10

| Thống kê | Số lượng |
| --- | ---: |
| Raw products | 20,000 |
| Missing descriptions | 2 |
| Duplicate nonmissing descriptions removed | 1,807 |
| Unique full descriptions | 18,191 |
| New train / validation / test | 12,733 / 2,728 / 2,730 |
| Legacy labeled / tensor sequences | 19,994 / 19,994 |
| Legacy train / validation / test | 13,995 / 2,999 / 3,000 |

Legacy labels được tạo bằng rule/specification rồi giữ ngẫu nhiên khoảng 20% token O. Legacy reference có 9 orphan I tags. Chia theo row với seed 42 tạo exact shortened-sequence overlap train-validation 1, train-test 2, validation-test 2. Đây là đếm overlap sau rút ngắn câu; không loại trừ leakage từ mô tả nguyên vẹn hoặc gần trùng.

Pipeline mới giữ nguyên text sau chuẩn hóa khoảng trắng, tạo weak spans từ rules/specifications, loại trùng theo hash mô tả casefold và chia unique descriptions. Tất cả split labels vẫn là silver. 50 annotation candidates có review_status pending, không phải 50 gold annotations.

Catalog recommendation vẫn là products_labeled.csv gốc: 20,000 rows; missing Color 3,321, Material 8,018, Dimensions 17,789, Weight 10,169. Có giá trị như số lượng trong package bị gán vào Weight và chuỗi màu/vật liệu chưa chuẩn; chất lượng catalog phải được xem xét riêng. Không lấy catalog attributes làm ground truth NER độc lập.
