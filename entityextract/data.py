"""Full-context weak labels and duplicate-safe, deterministic splits."""
import hashlib
import json
import random
import re
from pathlib import Path
from . import LABELS

COLORS = "black white red blue green brown pink silver gold navy yellow grey gray purple beige orange maroon".split()
MATERIALS = ["stainless steel", "cotton", "lycra", "fabric", "microfiber", "foam", "crystal", "pu", "leather", "wood", "plastic", "polyester", "canvas"]
UNIT = r"(?:cm|mm|m|inches|inch|in|ft|feet)"

def normalize_description(text):
    return re.sub(r"\s+", " ", str(text)).strip().casefold()

def description_key(text):
    return hashlib.sha256(normalize_description(text).encode()).hexdigest()

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def rule_spans(text):
    patterns = {
        "Color": r"\b(?:" + "|".join(COLORS) + r")\b",
        "Material": r"\b(?:" + "|".join(map(re.escape,MATERIALS)) + r")\b",
        "Weight": r"\b\d+(?:\.\d+)?\s*(?:kg|grams?|g|lbs?|oz)\b",
        "Dimensions": rf"\b\d+(?:\.\d+)?\s*(?:{UNIT})?\s*[x×]\s*\d+(?:\.\d+)?\s*(?:{UNIT})?(?:\s*[x×]\s*\d+(?:\.\d+)?\s*(?:{UNIT})?)?\b",
    }
    candidates = [{"label":label,"start":m.start(),"end":m.end(),"text":m.group()} for label,p in patterns.items() for m in re.finditer(p,text,re.I)]
    return nonoverlapping(candidates)

def nonoverlapping(spans):
    accepted=[]
    for s in sorted(spans,key=lambda s:(-(s["end"]-s["start"]),s["start"],s["label"])):
        if not any(s["start"]<t["end"] and t["start"]<s["end"] for t in accepted):
            accepted.append(s)
    return sorted(accepted,key=lambda s:s["start"])

def weak_spans(text,specification=""):
    spans=rule_spans(text)
    if not isinstance(specification,str):
        return spans
    try:
        parsed=json.loads(specification.replace("=>",":"))
    except (ValueError,TypeError):
        return spans
    entries=parsed.get("product_specification",[]) if isinstance(parsed,dict) else []
    if not isinstance(entries,list):
        return spans
    keys={"color":"Color","colour":"Color","material":"Material","fabric":"Material","dimension":"Dimensions","dimensions":"Dimensions","weight":"Weight"}
    for item in entries:
        if not isinstance(item,dict):
            continue
        label=keys.get(str(item.get("key","")).strip().casefold()); value=item.get("value")
        if label and isinstance(value,str) and value.strip():
            for m in re.finditer(r"(?<!\w)"+re.escape(value.strip())+r"(?!\w)",text,re.I):
                spans.append({"label":label,"start":m.start(),"end":m.end(),"text":m.group()})
    return nonoverlapping(spans)

def words_and_tags(text,spans):
    matches=list(re.finditer(r"\S+",text)); tags=["O"]*len(matches)
    for s in spans:
        indices=[i for i,m in enumerate(matches) if m.start()<s["end"] and s["start"]<m.end()]
        for position,i in enumerate(indices):
            tags[i]=("B-" if position==0 else "I-")+s["label"]
    return [m.group() for m in matches],tags

def align_labels(tokenizer,words,tags,max_length=128):
    if len(words)!=len(tags):
        raise ValueError("Each word must have one BIO label")
    mapping={label:i for i,label in enumerate(LABELS)}
    encoded=tokenizer(words,is_split_into_words=True,truncation=True,max_length=max_length)
    labels=[]; previous=None
    for word in encoded.word_ids():
        if word is None:
            labels.append(-100)
        else:
            tag=tags[word]
            if word==previous and tag.startswith("B-"):
                tag="I-"+tag[2:]
            labels.append(mapping[tag])
        previous=word
    return {"input_ids":encoded["input_ids"],"attention_mask":encoded["attention_mask"],"labels":labels}

def prepare_corpus(csv_path,output_dir,seed=42):
    import pandas as pd
    frame=pd.read_csv(csv_path); records=[]; seen=set(); missing=duplicates=0
    if "description" not in frame:
        raise ValueError("CSV requires description")
    for index,row in frame.iterrows():
        if pd.isna(row["description"]) or not str(row["description"]).strip():
            missing+=1; continue
        text=re.sub(r"\s+"," ",str(row["description"])).strip(); key=description_key(text)
        if key in seen:
            duplicates+=1; continue
        seen.add(key); spans=weak_spans(text,row.get("product_specifications","")); words,tags=words_and_tags(text,spans)
        records.append({"id":key,"source_row":int(index),"text":text,"words":words,"tags":tags,"entities":spans,"annotation_source":"rules_and_specifications_silver"})
    order=list(range(len(records))); random.Random(seed).shuffle(order)
    n_train=int(.7*len(order)); n_val=int(.15*len(order))
    splits={"train":order[:n_train],"validation":order[n_train:n_train+n_val],"test":order[n_train+n_val:]}
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    for name,indices in splits.items():
        with (output_dir/f"{name}.jsonl").open("w",encoding="utf-8") as stream:
            for i in indices:
                stream.write(json.dumps(records[i],ensure_ascii=False)+"\n")
    manifest={"source_file":Path(csv_path).name,"source_sha256":sha256_file(csv_path),"raw_rows":len(frame),"missing_descriptions":missing,"duplicate_descriptions_removed":duplicates,"unique_descriptions":len(records),"split_sizes":{k:len(v) for k,v in splits.items()},"seed":seed,"split_unit":"unique normalized full description","annotation":"silver","context_preserved":True,"warning":"Automatically generated labels, not human gold. Legacy checkpoint may have seen these new test descriptions."}
    (output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest

def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
