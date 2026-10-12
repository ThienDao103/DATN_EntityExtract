"""Save actual boundary errors from the completed historical evaluation."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from entityextract.workflow import read_conll,write_json
from entityextract.metrics import bio_entities
samples=read_conll(ROOT/"data_labeled.txt")
selected={name:0 for name in ("Color","Material","Dimensions","Weight")}
cases=[]
for line in (ROOT/"artifacts/legacy-evaluation/predictions.jsonl").read_text(encoding="utf-8").splitlines():
    row=json.loads(line); gold=bio_entities(row["reference"]); predicted=bio_entities(row["prediction"])
    for name in selected:
        actual={e for e in gold if e[0]==name}; guessed={e for e in predicted if e[0]==name}
        if actual!=guessed and selected[name]<3:
            cases.append({"legacy_row":row["id"],"attribute":name,"shortened_input":" ".join(samples[row["id"]][0]),
                          "false_positive_subword_spans":sorted(guessed-actual),"false_negative_subword_spans":sorted(actual-guessed)})
            selected[name]+=1
    if all(count==3 for count in selected.values()): break
write_json(ROOT/"artifacts/error_cases.json",cases)
lines=["# Phân tích lỗi thực nghiệm", "", "Các ví dụ được lấy từ run 3.000 historical test sequences trên câu đã rút ngắn và nhãn tự động. Chọn ba trường hợp boundary/type mismatch cho mỗi loại. Ranh giới là chỉ số subword, không phải offset ký tự hoặc gold annotation.", ""]
for case in cases:
    lines.extend([f"## {case['attribute']} tại legacy row {case['legacy_row']}","",case["shortened_input"][:600],"",
        f"False positives (type, start, end): {case['false_positive_subword_spans']}","",
        f"False negatives (type, start, end): {case['false_negative_subword_spans']}",""])
lines.extend(["## Lỗi và giới hạn trên câu nguyên vẹn", "", "Service smoke dự đoán Weight thành 2 kg. gồm dấu chấm cuối câu. Đây là lỗi boundary cụ thể, được giữ trong service_smoke.json thay vì sửa evidence. Một mô tả dài có thể được xử lý đủ tới cuối nhưng model vẫn bỏ sót thuộc tính; kiểm thử coverage không được diễn giải là chứng minh semantic accuracy.","", "Hướng kiểm chứng: annotator độc lập sửa span trên mô tả nguyên vẹn, không xem prediction trước; chốt hướng dẫn về punctuation, units, phủ định, multiword material và catalog noisy. Không khắc phục bằng cách báo cáo weighted token F1 như entity F1."])
(ROOT/"docs/ERROR_ANALYSIS.md").write_text("\n".join(lines),encoding="utf-8")
print(f"Saved {len(cases)} real error cases")
