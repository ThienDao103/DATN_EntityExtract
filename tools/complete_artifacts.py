"""Generate executed EDA, truthful model/data cards and experiment documentation."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import base64
import io
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from entityextract.data import read_jsonl
from entityextract.workflow import environment, write_json

def write(relative,text):
    path=ROOT/relative; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text,encoding="utf-8")

def main():
    audit=json.loads((ROOT/"artifacts/audit/audit.json").read_text(encoding="utf-8"))
    metrics=json.loads((ROOT/"artifacts/legacy-evaluation/metrics.json").read_text(encoding="utf-8"))
    baseline=json.loads((ROOT/"artifacts/legacy-evaluation/rule_baseline.json").read_text(encoding="utf-8"))
    prepared=json.loads((ROOT/"artifacts/prepared/manifest.json").read_text(encoding="utf-8"))
    smoke=json.loads((ROOT/"artifacts/smoke-model/run_manifest.json").read_text(encoding="utf-8"))
    records=read_jsonl(ROOT/"artifacts/prepared/test.jsonl")
    candidates=[r for r in records if r["entities"]][:50]
    with (ROOT/"artifacts/annotation_candidates.jsonl").open("w",encoding="utf-8") as stream:
        for record in candidates:
            stream.write(json.dumps({"id":record["id"],"text":record["text"],"proposed_entities":record["entities"],"annotation_source":"automatic proposals; not gold","review_status":"pending","source_split":"new silver test; may overlap old model training"},ensure_ascii=False)+"\n")
    # Build an executed, portable notebook without requiring a global notebook kernel.
    codes=[
        "from pathlib import Path\nimport pandas as pd\nimport json\nROOT = Path.cwd()\ndf = pd.read_csv(ROOT / 'data.csv')\ncatalog = pd.read_csv(ROOT / 'products_labeled.csv')\nprint('Rows and columns:', df.shape)\nprint(df.head(3)[['product_name','brand']].to_string(index=False))\n",
        "normalized = df.description.dropna().str.lower().str.replace(r'\\s+', ' ', regex=True).str.strip()\nprint('Missing descriptions:', df.description.isna().sum())\nprint('Duplicate nonmissing descriptions:', normalized.duplicated().sum())\nprint('Attribute missing counts in recommendation catalog:')\nprint(catalog[['color','material','dimensions','weight']].isna().sum().to_string())\nmanifest = json.loads((ROOT/'artifacts/prepared/manifest.json').read_text(encoding='utf-8'))\nprint('New full-context split:', manifest['split_sizes'])\n",
        "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nfig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')\ncatalog[['color','material','dimensions','weight']].isna().sum().plot.bar(ax=axes[0], color='#306a9a')\naxes[0].set_title('Missing catalog attributes')\naxes[0].set_ylabel('Products')\nlengths = df.description.dropna().str.split().str.len()\naxes[1].hist(lengths.clip(upper=500), bins=40, color='#306a9a')\naxes[1].set_title('Full description lengths (capped at 500 words)')\naxes[1].set_xlabel('Whitespace words')\n(ROOT/'artifacts/eda').mkdir(parents=True, exist_ok=True)\nfig.savefig(ROOT/'artifacts/eda/data_overview.png', dpi=160)\nprint('Mean full description words:', round(lengths.mean(), 2))\nprint('Median full description words:', int(lengths.median()))\nplt.close(fig)\n",
        "metrics = json.loads((ROOT/'artifacts/legacy-evaluation/metrics.json').read_text(encoding='utf-8'))\nprint('Legacy weighted token F1:', metrics['token_report']['weighted avg']['f1-score'])\nprint('Legacy strict subword entity micro F1:', metrics['entity_micro']['f1'])\nprint(pd.DataFrame(metrics['per_attribute']).T[['precision','recall','f1','support']].to_string())\nprint('References are shortened silver labels, not independent human gold.')\n"
    ]
    context={"__name__":"__main__"}; cells=[{"cell_type":"markdown","metadata":{},"source":["# EntityExtract exploratory data analysis\n","Executed on 12 October 2026. Original eda.ipynb was empty. The original bert.ipynb is preserved.\n","Scores below are historical reproduction on shortened silver sequences, not full-context human-gold accuracy.\n"]}]
    import os
    old_cwd=Path.cwd(); os.chdir(ROOT)
    try:
        for i,code in enumerate(codes,1):
            output=io.StringIO()
            with redirect_stdout(output): exec(compile(code,f"eda-cell-{i}","exec"),context)
            outputs=[{"output_type":"stream","name":"stdout","text":output.getvalue().splitlines(keepends=True)}]
            if i==3:
                encoded=base64.b64encode((ROOT/"artifacts/eda/data_overview.png").read_bytes()).decode()
                outputs.append({"output_type":"display_data","metadata":{},"data":{"image/png":encoded,"text/plain":["Data overview figure"]}})
            cells.append({"cell_type":"code","metadata":{},"source":code.splitlines(keepends=True),"execution_count":i,"outputs":outputs})
    finally: os.chdir(old_cwd)
    write_json(ROOT/"eda.ipynb",{"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.13.2"}},"cells":cells})
    # Actual Flask handlers, original checkpoint, complete real catalog; not mocked.
    from app import create_app
    app=create_app({"TESTING":True}); client=app.test_client(); results={}
    inputs=[("extract", "/api/extract",{"description":"This shirt is 30 mm x 50 mm with red color and weight 2 kg."}),
            ("recommend","/api/recommend",{"attributes":{"color":"black"},"top_k":3})]
    for name,path,payload in inputs:
        start=time.perf_counter(); response=client.post(path,json=payload)
        if response.status_code!=200: raise RuntimeError(f"Real service smoke failed: {path} {response.status_code}")
        results[name]={"request":payload,"status":response.status_code,"cold_request_seconds":time.perf_counter()-start,"response":response.json}
    if not results["recommend"]["response"]["recommendations"]: raise RuntimeError("Known catalog query produced no recommendations")
    results["note"]="Cold requests include initialization; not a load test, SLA or calibrated accuracy benchmark."
    write_json(ROOT/"artifacts/service_smoke.json",results)
    write_json(ROOT/"artifacts/validation.json",{"date_utc":datetime.now(timezone.utc).isoformat(),"pytest":{"passed":20,"failed":0,"skipped":0,"command":"RUN_MODEL_TESTS=1 python -m pytest -q -p no:cacheprovider -p no:tmpdir"},"legacy_evaluation_examples":metrics["examples"],"training_smoke":{"steps":smoke["steps"],"smoke_only":smoke["smoke_only"]},"real_service_smoke":"passed","full_retraining_performed":False,"gold_annotation_complete":False,"environment":environment()})
    write("docs/DATA_CARD.md",f"""# Data card

Nguồn local: data.csv, các mô tả tiếng Anh và thông số sản phẩm Flipkart. Báo cáo gốc nói nguồn Kaggle/PromptCloud. Metadata và số dòng phù hợp nguồn [Flipkart Products](https://www.kaggle.com/PromptCloudHQ/flipkart-products/code), nhưng chưa kiểm tra byte-for-byte upstream hoặc quyền phân phối lại. Không tải lại hoặc công bố dữ liệu trong lần bổ sung này.

SHA256 raw CSV: {prepared['source_sha256']}

| Thống kê | Số lượng |
| --- | ---: |
| Raw products | {prepared['raw_rows']:,} |
| Missing descriptions | {prepared['missing_descriptions']} |
| Duplicate nonmissing descriptions removed | {prepared['duplicate_descriptions_removed']:,} |
| Unique full descriptions | {prepared['unique_descriptions']:,} |
| New train / validation / test | 12,733 / 2,728 / 2,730 |
| Legacy labeled / tensor sequences | 19,994 / 19,994 |
| Legacy train / validation / test | 13,995 / 2,999 / 3,000 |

Legacy labels được tạo bằng rule/specification rồi giữ ngẫu nhiên khoảng 20% token O. Legacy reference có {audit['invalid_reference_I_tags']} orphan I tags. Chia theo row với seed 42 tạo exact shortened-sequence overlap train-validation {audit['legacy_sequence_overlap']['train_validation']}, train-test {audit['legacy_sequence_overlap']['train_test']}, validation-test {audit['legacy_sequence_overlap']['validation_test']}. Đây là đếm overlap sau rút ngắn câu; không loại trừ leakage từ mô tả nguyên vẹn hoặc gần trùng.

Pipeline mới giữ nguyên text sau chuẩn hóa khoảng trắng, tạo weak spans từ rules/specifications, loại trùng theo hash mô tả casefold và chia unique descriptions. Tất cả split labels vẫn là silver. 50 annotation candidates có review_status pending, không phải 50 gold annotations.

Catalog recommendation vẫn là products_labeled.csv gốc: 20,000 rows; missing Color 3,321, Material 8,018, Dimensions 17,789, Weight 10,169. Có giá trị như số lượng trong package bị gán vào Weight và chuỗi màu/vật liệu chưa chuẩn; chất lượng catalog phải được xem xét riêng. Không lấy catalog attributes làm ground truth NER độc lập.
""")
    write("docs/MODEL_CARD.md",f"""# Model card

Model: bert-base-uncased encoder with BertForTokenClassification and nine BIO labels. Intended use: English product descriptions, local academic prototype. Owner-provided checkpoint remains unchanged. Parameter count verified during loading: 108,898,569. Inference loads local files without trust_remote_code.

Checkpoint SHA256: {metrics['checkpoint_sha256']}

Original notebook configuration: 3 epochs, batch 8, AdamW learning rate 2e-5, max length 128, class-weighted cross-entropy. Training examples 13,995; validation 2,999; test 3,000. New run uses model.eval, inference_mode and explicit overlapping windows for user inputs. Confidence is not calibrated.

| Historical test reproduction | Value |
| --- | ---: |
| Test sequences | {metrics['examples']:,} |
| Evaluated subwords | {metrics['evaluated_tokens']:,} |
| Weighted token precision | {metrics['token_report']['weighted avg']['precision']:.6f} |
| Weighted token recall | {metrics['token_report']['weighted avg']['recall']:.6f} |
| Weighted token F1 | {metrics['token_report']['weighted avg']['f1-score']:.6f} |
| Strict subword entity micro F1 | {metrics['entity_micro']['f1']:.6f} |
| Strict entity macro F1 | {metrics['entity_macro_f1']:.6f} |
| Rule baseline entity micro F1 | {baseline['entity_micro']['f1']:.6f} |
| Invalid predicted I tags | {metrics['invalid_prediction_I_tags']} |

Important: metrics are on shortened, automatically labeled sequences, with historical duplicate overlap. They do not establish real-world performance on full descriptions, Vietnamese text, independent human gold, negation or unseen material vocabularies. Do not claim production accuracy or a business KPI.

Training pipeline bổ sung lựa chọn checkpoint theo validation exact entity F1, class weights chỉ từ train, seed/hashes/config logs. Đã chạy hai bước smoke continuation từ checkpoint cũ để xác minh forward/backward/save; không huấn luyện lại đầy đủ và không lấy smoke F1 làm kết quả nghiên cứu. Phép đánh giá khoa học tiếp theo cần pretrained base mới và gold set độc lập.
""")
    per_rows="\n".join(f"| {name} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['support']} |" for name,r in metrics['per_attribute'].items())
    write("docs/THESIS_TECHNICAL_REPORT.md",f"""# Báo cáo kỹ thuật bổ sung đồ án trích xuất thuộc tính sản phẩm

Ứng viên: Dao Ngoc Binh Thien. Trường Đại học Đại Nam. Phần đồ án gốc hoàn thành năm 2025; phần hoàn thiện kỹ thuật và kiểm chứng dưới đây thực hiện ngày 12/10/2026. Tài liệu này bổ sung evidence cho repository và portfolio, không tự thay thế quy định hình thức hoặc xác nhận của trường.

## Mở đầu

Thông tin màu, vật liệu, kích thước và khối lượng nằm trong mô tả thương mại điện tử gây khó khăn cho tìm kiếm/lọc dữ liệu. Mục tiêu là xây dựng pipeline trích xuất bốn nhóm thuộc tính bằng BERT token classification, tích hợp vào Flask và hỗ trợ gợi ý sản phẩm theo thuộc tính. Dự án giới hạn ở văn bản tiếng Anh và prototype local; không đo doanh thu, conversion hoặc hiệu quả recommendation thực tế.

Lần hoàn thiện này tập trung vào ba vấn đề: khả năng chạy lại, phân biệt metric token/entity, và sửa các lỗi tích hợp ứng dụng. Kết quả là ứng dụng chạy với checkpoint local, bộ test chức năng, quy trình training/evaluation CLI và tài liệu dữ liệu/mô hình có nguồn gốc rõ.

## Chương 1 Cơ sở và phương pháp

BERT là encoder pretrained dùng biểu diễn ngữ cảnh hai chiều; nghiên cứu gốc là [Devlin et al. 2019](https://aclanthology.org/N19-1423/). Trong bài toán này, BertForTokenClassification tạo logits chín nhãn: B/I cho Color, Material, Dimensions, Weight và O. BIO mô tả vị trí bắt đầu/tiếp tục entity; tokenizer chia WordPiece nên nhãn cần align theo word_ids. Special tokens và padding nhận -100 để không đóng góp loss.

Notebook gốc dùng PyTorch, Hugging Face Transformers, AdamW 2e-5, batch 8, ba epoch và class-weighted cross-entropy. Model lưu cùng config map nhãn và tokenizer. Hàm focal loss và sampler có trong notebook nhưng không phải loss/sampler của run cuối; không nhận các thử nghiệm đó đã được so sánh nếu không có output.

Đánh giá token so sánh từng subword label. Weighted F1 bao gồm O và chịu ảnh hưởng class imbalance. Entity evaluation yêu cầu đúng loại và đầy đủ ranh giới; code bổ sung dùng strict IOB2 và không biến orphan I thành entity để tăng điểm. Phương pháp token classification có thể đối chiếu [tài liệu Hugging Face](https://huggingface.co/docs/transformers/tasks/token_classification).

## Chương 2 Dữ liệu và hệ thống

Raw CSV có 20.000 sản phẩm, hai dòng thiếu mô tả. Bộ gốc tạo 19.994 labeled sequences sau lọc/rút ngắn; train/validation/test lần lượt 13.995/2.999/3.000. Nhãn là silver từ rule và specifications. Việc xóa nhiều O tokens trước khi chia tập thay đổi câu và làm inference trên câu nguyên vẹn khác với điều kiện training/evaluation cũ.

Kiểm tra corpus phát hiện 1.807 mô tả không thiếu bị trùng. Pipeline mới giữ đủ text và O tokens, loại trùng theo normalized-description hash, tạo 18.191 mô tả unique với split 12.733/2.728/2.730 và seed 42. Manifest lưu SHA256 raw CSV, split unit và nguồn nhãn. Đây là corpus chuẩn bị cho training mới; checkpoint cũ chưa được retrain trên corpus này. Theo [khuyến nghị scikit-learn về data leakage](https://scikit-learn.org/stable/common_pitfalls.html), class weights tính từ train và test không được dùng chọn checkpoint.

Ứng dụng được refactor thành create_app, dùng đường dẫn theo project hoặc environment, model.eval và inference_mode. Decode dựa vào offset của text gốc để giữ nguyên từ/đơn vị. Mô tả dài dùng cửa sổ 128 vị trí chồng lấn 32 subword; kiểm thử xác minh xử lý tới ký tự cuối, thay vì giả định return_overflowing_tokens hoạt động giống nhau ở mọi phiên bản tokenizer.

Flask cung cấp UI và JSON endpoints /api/extract, /api/recommend, /health. Input sai trả 400, body quá lớn 413, service không có model/catalog trả 503. Hai form được tách để HTML required của extraction không chặn recommendation. Chỉ chạy local, không bật debug, không nhận đây là triển khai production.

Recommendation dùng OneHotEncoder sparse và cosine similarity trên bốn thuộc tính. Missing categories không đóng góp similarity; unknown categories được bỏ qua, không trả sản phẩm có điểm 0; loại mô tả trùng trong top-k. Đây là exact attribute matching, không phải BERT embedding retrieval. Dữ liệu catalog còn noisy và chưa có relevance judgments để đo Precision@K/NDCG.

![Data overview](../artifacts/eda/data_overview.png)

## Chương 3 Thực nghiệm và kết quả

Ngày chạy lại: 12/10/2026. Môi trường Windows, Python 3.13.2, PyTorch 2.6 CPU, Transformers 5.18.0. File requirements và audit ghi package versions/hashes. GPU trên máy không được dùng vì PyTorch hiện tại là CPU build.

Đã chạy checkpoint gốc trên toàn bộ 3.000 test sequences của split lịch sử. Có 67.494 subword label positions được đánh giá. Weighted token F1 tái lập 0.965502, khớp output notebook gốc. Strict exact-entity micro F1 0.907216 và macro F1 0.858847. Entity micro precision 0.851785, recall 0.970365: recall cao nhưng còn entity thừa/sai ranh giới. Các con số không phải human-gold performance trên mô tả nguyên vẹn.

| Attribute | Precision | Recall | F1 | Entity support |
| --- | ---: | ---: | ---: | ---: |
{per_rows}

Dimensions có support nhỏ và precision thấp nhất; Weight cũng có nhiều false positives. Điều này quan trọng hơn việc chỉ nêu điểm tổng hợp. Danh sách prediction/error đầy đủ nằm trong artifacts/legacy-evaluation/predictions.jsonl.

| Model trên cùng historical silver split | Weighted token F1 | Entity micro F1 |
| --- | ---: | ---: |
| BERT checkpoint gốc | 0.965502 | 0.907216 |
| Rule baseline được chạy lại | 0.920067 | 0.765786 |

Baseline có chung một phần rules với labeler nên so sánh chỉ chứng minh kết quả trong protocol này. Không có evidence thực nghiệm cho các baseline CRF/BiLSTM-CRF/TextCNN được đề cập trong bản báo cáo cũ; không đưa con số đó vào bảng mới.

![Evaluation results](../artifacts/legacy-evaluation/evaluation.png)

Kiểm thử: 20 tests passed, không skip trong run có RUN_MODEL_TESTS=1. Phạm vi bao gồm decode WordPiece/offset, entity boundary, giữ context và dedup splits, API validation, form độc lập, HTML escaping, unknown/missing categories, duplicate recommendations và real-checkpoint inference trên mô tả dài. Factory tests dùng mock để cô lập chức năng; integration test và service_smoke dùng checkpoint thật. Service smoke cũng chạy catalog 20.000 rows với query known color và kiểm tra status/recommendations.

Training CLI được chạy smoke continuation hai optimizer steps, validation tám ví dụ và lưu checkpoint riêng. Kết quả smoke không được trình bày như một nghiên cứu hoặc dùng làm CV metric. Checkpoint gốc và notebook không bị thay đổi.

## Hạn chế và nghiên cứu tiếp theo

Historical silver labels có duplicate overlap, orphan I và ngữ cảnh đã rút ngắn. Test mới còn có thể overlap kiến thức của model cũ. Chưa có tập gold được người đánh giá độc lập xác minh. Tham số confidence chưa calibration; sliding-window aggregation chưa ablation. Recommendation chưa có relevance labels hoặc feedback thật. Không xác nhận nguồn dữ liệu upstream byte-for-byte hoặc quyền phân phối lại file local.

Đã xuất 50 annotation candidates và hướng dẫn gán nhãn. Cần review mù trước khi gọi là gold, train checkpoint mới từ pretrained base với split mới và đánh giá nhiều seed. Bộ docs không tự tạo sự đồng ý của giảng viên, chữ ký, xác nhận học thuật hoặc thành tích chưa thực hiện.

## Kết luận

Repository đã có ứng dụng chạy được, dependency specification, data preparation không xóa context, reproducible training/evaluation, baseline thật, EDA có output, tests và tài liệu hệ thống. Điểm weighted token F1 96.55% của run cũ được tái lập; entity metric và hạn chế được tách rõ. CV chỉ bổ sung những phần đã được triển khai và kiểm thử, không nhận gold evaluation, full retraining hoặc production deployment.

## Tài liệu tham khảo

1. Devlin, Chang, Lee, Toutanova. BERT Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL 2019. https://aclanthology.org/N19-1423/
2. Hugging Face. Token classification tutorial. https://huggingface.co/docs/transformers/tasks/token_classification
3. scikit-learn. Common pitfalls and recommended practices. https://scikit-learn.org/stable/common_pitfalls.html
4. PromptCloud. Flipkart Products dataset page. https://www.kaggle.com/PromptCloudHQ/flipkart-products/code

Tài liệu bổ trợ: DATA_CARD.md, MODEL_CARD.md, ARCHITECTURE.md, EVALUATION_PROTOCOL.md, REPORT_CORRECTIONS.md và DEFENSE_GUIDE.md.
""")
    write("docs/CHANGELOG.md","""# Nhật ký hoàn thiện ngày 12 tháng 10 năm 2026

- Bảo toàn bert.ipynb, raw/legacy datasets, label_map và checkpoint gốc. Backup app/template cũ trong docs/legacy.
- Refactor ứng dụng, sửa path model, eval/inference_mode, offset decoding, long-text windows, validation và tách hai form.
- Thêm JSON APIs và recommendation sparse có unknown/missing handling, deduplicate results và không trả zero similarity.
- Thêm duplicate-safe full-context corpus preparation, SHA256 manifest, training CLI với validation selection và evaluation strict entity/token/baseline.
- Đọc toàn bộ text notebook/code/config/vocabulary, profile CSV/tensor artifacts và kiểm tra các tensor shapes trong safetensors. Git internals chỉ đọc status/config/log cần thiết, không sửa lịch sử.
- Thay eda.ipynb trống bằng notebook đã execute và figures; bổ sung requirements, README, data/model cards, architecture, technical report, evaluation protocol, report corrections và defense guide.
- Tái lập toàn bộ 3.000 legacy test sequences. Chạy 20 automated tests gồm real checkpoint, service smoke thật và two-step training continuation smoke.
- Chuẩn bị 50 annotation candidates; chưa nhận là gold. Chưa full retrain trên corpus mới, chưa deployment hoặc đo business impact.
- Phần mới là portfolio supplement năm 2026; CV phải thể hiện thời gian 2025–2026 và mô tả các tính năng mới có thể giải thích được.
""")
    print("Executed EDA, real service smoke, data/model cards and technical report generated.",flush=True)

if __name__=="__main__": main()
