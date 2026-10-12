"""Audits, repeatable evaluation and training without overwriting the legacy model."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import random
import time
from . import LABELS
from .data import align_labels, description_key, prepare_corpus, read_jsonl, rule_spans, sha256_file, words_and_tags
from .metrics import evaluate_sequences, bio_entities

ROOT=Path(__file__).resolve().parents[1]

def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding="utf-8")

def read_conll(path):
    records=[]; words=[]; tags=[]
    for line in Path(path).read_text(encoding="utf-8").splitlines()+[""]:
        if not line.strip():
            if words: records.append((words,tags)); words=[]; tags=[]
        else:
            word,tag=line.rsplit(maxsplit=1)
            if tag not in LABELS: raise ValueError(f"Unknown label {tag}")
            words.append(word); tags.append(tag)
    return records

def environment():
    import platform, torch
    names=("torch","transformers","Flask","pandas","numpy","scikit-learn","matplotlib","safetensors","pytest")
    return {"python":platform.python_version(),"platform":platform.platform(),"cuda_available":torch.cuda.is_available(),
            "packages":{n:importlib.metadata.version(n) for n in names}}

def audit(output):
    import pandas as pd
    import torch
    from safetensors import safe_open
    from transformers import BertTokenizerFast
    from sklearn.model_selection import train_test_split
    frame=pd.read_csv(ROOT/"data.csv"); catalog=pd.read_csv(ROOT/"products_labeled.csv")
    notebook=json.loads((ROOT/"bert.ipynb").read_text(encoding="utf-8"))
    conll=read_conll(ROOT/"data_labeled.txt")
    tensors=pd.read_pickle(ROOT/"data_preprocessed.pkl")  # Owner-provided local artifact; never load untrusted pickle.
    label_map=torch.load(ROOT/"label_map.pt",weights_only=True)
    tokenizer=BertTokenizerFast.from_pretrained(ROOT/"bert_product_model",local_files_only=True)
    normalized=frame.description.dropna().map(description_key)
    sequences=[description_key(" ".join(words)) for words,_ in conll]
    train,temp=train_test_split(list(range(len(conll))),test_size=.3,random_state=42)
    val,test=train_test_split(temp,test_size=.5,random_state=42)
    keys=[{sequences[i] for i in split} for split in (train,val,test)]
    malformed=0
    for words,tags in conll:
        malformed+=sum(tag.startswith("I-") and (i==0 or tags[i-1] not in ("B-"+tag[2:],tag)) for i,tag in enumerate(tags))
    files=[]
    excluded={".git","__pycache__",".pytest_cache","artifacts",".venv"}
    for p in sorted(ROOT.rglob("*")):
        if p.is_file() and not any(part in excluded for part in p.relative_to(ROOT).parts):
            files.append({"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":sha256_file(p)})
    with safe_open(ROOT/"bert_product_model/model.safetensors",framework="pt",device="cpu") as weights:
        shapes={name:list(weights.get_slice(name).get_shape()) for name in weights.keys()}
        metadata=weights.metadata()
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    legacy_outputs=[]
    for i,cell in enumerate(notebook["cells"]):
        for item in cell.get("outputs",[]):
            text="".join(item.get("text",[]))
            if "F1:" in text or "Train size:" in text or "Tổng số mẫu:" in text:
                legacy_outputs.append({"cell":i,"text":text})
    result={"date_utc":datetime.now(timezone.utc).isoformat(),"environment":environment(),"raw_rows":len(frame),
            "missing_description":int(frame.description.isna().sum()),"duplicate_full_descriptions":int(normalized.duplicated().sum()),
            "catalog_rows":len(catalog),"catalog_missing":{c:int(catalog[c].isna().sum()) for c in ("color","material","dimensions","weight")},
            "legacy_labeled_sequences":len(conll),"legacy_tensor_rows":len(tensors),"legacy_split_sizes":{"train":len(train),"validation":len(val),"test":len(test)},
            "legacy_sequence_overlap":{"train_validation":len(keys[0]&keys[1]),"train_test":len(keys[0]&keys[2]),"validation_test":len(keys[1]&keys[2])},
            "legacy_label_counts":dict(Counter(tag for _,tags in conll for tag in tags)),"invalid_reference_I_tags":malformed,
            "label_map":label_map,"tokenizer_vocabulary_size":len(tokenizer),"model_tensor_shapes":shapes,"model_metadata":metadata,
            "files":files,"notebook_cells":len(notebook["cells"]),"original_eda_bytes":0,
            "limitations":["Legacy references are automatically labeled and O tokens were removed.","Historical splits are not guaranteed duplicate-free.","Model performance on full descriptions or human-annotated gold is not established."]}
    write_json(output/"audit.json",result); write_json(output/"historical_notebook_results.json",legacy_outputs)
    print(json.dumps({k:v for k,v in result.items() if k in ("raw_rows","duplicate_full_descriptions","legacy_split_sizes","legacy_sequence_overlap","invalid_reference_I_tags")},indent=2),flush=True)
    return result

def collate_samples(samples,device="cpu"):
    import torch
    length=max(len(s["input_ids"]) for s in samples)
    return {key:torch.tensor([list(s[key])+[(-100 if key=="labels" else 0)]*(length-len(s[key])) for s in samples],dtype=torch.long,device=device)
            for key in ("input_ids","attention_mask","labels")}

def predict_samples(model,samples,batch_size=8,device="cpu"):
    import torch
    references=[]; predictions=[]
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(samples),batch_size):
            batch=collate_samples(samples[start:start+batch_size],device)
            labels=batch.pop("labels"); guessed=model(**batch).logits.argmax(-1)
            for gold,pred in zip(labels.cpu().tolist(),guessed.cpu().tolist()):
                paired=[(a,b) for a,b in zip(gold,pred) if a!=-100]
                references.append([LABELS[a] for a,_ in paired]); predictions.append([LABELS[b] for _,b in paired])
            if start % (batch_size*25)==0: print(f"Evaluated {min(start+batch_size,len(samples))}/{len(samples)}",flush=True)
    return references,predictions

def plot_results(metrics,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
    names=list(metrics["per_attribute"]); values=[metrics["per_attribute"][n]["f1"] for n in names]
    axes[0].bar(names,values,color="#306a9a"); axes[0].set_ylim(0,1); axes[0].set_ylabel("Exact entity F1"); axes[0].set_title("Strict IOB2 by attribute")
    matrix=np.array(metrics["confusion_matrix"],dtype=float); sums=matrix.sum(axis=1,keepdims=True)
    matrix=np.divide(matrix,sums,out=np.zeros_like(matrix),where=sums!=0)
    axes[1].imshow(matrix,vmin=0,vmax=1,cmap="Blues")
    axes[1].set_xticks(range(9),LABELS,rotation=90); axes[1].set_yticks(range(9),LABELS)
    axes[1].set_xlabel("Predicted label"); axes[1].set_ylabel("Reference label"); axes[1].set_title("Row-normalized token confusion matrix")
    fig.savefig(path,dpi=160); plt.close(fig)

def evaluate(model_dir,output,limit=0,protocol="legacy",prepared=None):
    import pandas as pd
    import torch
    from sklearn.model_selection import train_test_split
    from transformers import BertForTokenClassification,BertTokenizerFast
    torch.set_num_threads(4)
    tokenizer=BertTokenizerFast.from_pretrained(model_dir,local_files_only=True)
    model=BertForTokenClassification.from_pretrained(model_dir,local_files_only=True).eval()
    if [model.config.id2label[i] for i in range(9)]!=LABELS: raise ValueError("Model label order mismatch")
    if protocol=="legacy":
        frame=pd.read_pickle(ROOT/"data_preprocessed.pkl"); conll=read_conll(ROOT/"data_labeled.txt")
        if len(frame)!=len(conll): raise ValueError("Legacy artifacts differ in size")
        _,temp=train_test_split(list(range(len(frame))),test_size=.3,random_state=42)
        _,indices=train_test_split(temp,test_size=.5,random_state=42)
        if limit: indices=indices[:limit]
        samples=[]; baseline=[]; ids=[]
        for i in indices:
            stored={k:v.tolist()[:128] for k,v in frame.iloc[i]["tensor_data"].items()}
            words,tags=conll[i]; encoded=align_labels(tokenizer,words,tags,max_length=100000)
            if encoded["input_ids"][:128]!=stored["input_ids"]:
                raise ValueError(f"Tokenizer/source mismatch at legacy row {i}")
            text=" ".join(words); _,rule_tags=words_and_tags(text,rule_spans(text))
            rule_encoded=align_labels(tokenizer,words,rule_tags,max_length=100000)
            # Legacy truncation may cut a sequence without SEP; use matching positions.
            raw_rule=rule_encoded["labels"]
            baseline.append([LABELS[raw_rule[j]] if j<len(raw_rule) and raw_rule[j]!=-100 else "O" for j,g in enumerate(stored["labels"]) if g!=-100])
            samples.append(stored); ids.append(int(i))
        disclaimer="Historical random split on shortened, automatically labeled sequences; duplicate leakage may exist. Entity boundaries are subword positions, not independently human-annotated gold."
        split_hash=sha256_file(ROOT/"data_preprocessed.pkl")
    else:
        if not prepared: raise ValueError("--prepared is required")
        records=read_jsonl(Path(prepared)/"test.jsonl")
        if limit: records=records[:limit]
        samples=[align_labels(tokenizer,r["words"],r["tags"]) for r in records]; ids=[r["id"] for r in records]
        baseline=[]
        for r,s in zip(records,samples):
            _,tags=words_and_tags(r["text"],rule_spans(r["text"])); b=align_labels(tokenizer,r["words"],tags)
            baseline.append([LABELS[p] for p,g in zip(b["labels"],s["labels"]) if g!=-100])
        disclaimer="Full-context silver labels. Legacy model may have seen these descriptions; this is a diagnostic, not an independent generalization benchmark. Rule baseline shares labeling rules with references."
        split_hash=sha256_file(Path(prepared)/"test.jsonl")
    started=time.perf_counter(); references,predictions=predict_samples(model,samples)
    result=evaluate_sequences(references,predictions); result.update(protocol=protocol,disclaimer=disclaimer,seconds=time.perf_counter()-started,
        dataset_sha256=split_hash,checkpoint_sha256=sha256_file(Path(model_dir)/"model.safetensors"),seed=42,max_length=128,device="cpu",environment=environment())
    baseline_result=evaluate_sequences(references,baseline)
    baseline_result["disclaimer"]="Rule extraction on the same selected inputs; references are weak labels. Not an independent gold benchmark."
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    write_json(output/"metrics.json",result); write_json(output/"rule_baseline.json",baseline_result)
    with (output/"predictions.jsonl").open("w",encoding="utf-8") as f:
        for key,gold,pred in zip(ids,references,predictions):
            f.write(json.dumps({"id":key,"reference":gold,"prediction":pred,"entity_match":bio_entities(gold)==bio_entities(pred)})+"\n")
    plot_results(result,output/"evaluation.png")
    print(json.dumps({"examples":result["examples"],"weighted_token_f1":result["token_report"]["weighted avg"]["f1-score"],"entity_micro_f1":result["entity_micro"]["f1"],"seconds":result["seconds"]},indent=2),flush=True)
    return result

def train(prepared,output,initial_model="bert-base-uncased",epochs=3,batch_size=8,smoke_steps=0,seed=42):
    import numpy as np
    import torch
    from transformers import BertForTokenClassification,BertTokenizerFast
    output=Path(output)
    if output.resolve()==(ROOT/"bert_product_model").resolve(): raise ValueError("Never overwrite the original checkpoint")
    if output.exists() and any(output.iterdir()): raise ValueError("Use an empty output directory")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(4)
    tokenizer=BertTokenizerFast.from_pretrained(initial_model)
    model=BertForTokenClassification.from_pretrained(initial_model,num_labels=9,id2label=dict(enumerate(LABELS)),label2id={s:i for i,s in enumerate(LABELS)},ignore_mismatched_sizes=True)
    if Path(initial_model).resolve()==(ROOT/"bert_product_model").resolve() and not smoke_steps:
        raise ValueError("Original fine-tuned checkpoint may contain test knowledge; use pretrained bert-base-uncased for a new benchmark")
    training=read_jsonl(Path(prepared)/"train.jsonl"); validation=read_jsonl(Path(prepared)/"validation.jsonl")
    if smoke_steps: training=training[:max(16,smoke_steps*batch_size)]; validation=validation[:8]
    samples=[align_labels(tokenizer,r["words"],r["tags"]) for r in training]
    validation_samples=[align_labels(tokenizer,r["words"],r["tags"]) for r in validation]
    counts=Counter(i for sample in samples for i in sample["labels"] if i!=-100)
    weights=torch.tensor([1/max(counts.get(i,0),1) for i in range(9)],dtype=torch.float)
    weights=weights/weights.sum()*9
    device="cuda" if torch.cuda.is_available() else "cpu"; model.to(device)
    loss_fn=torch.nn.CrossEntropyLoss(weight=weights.to(device),ignore_index=-100)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5)
    history=[]; best=-1.; steps=0; output.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,epochs+1):
        model.train(); order=list(range(len(samples))); random.Random(seed+epoch).shuffle(order); losses=[]
        for start in range(0,len(order),batch_size):
            batch=collate_samples([samples[i] for i in order[start:start+batch_size]],device); labels=batch.pop("labels")
            optimizer.zero_grad(); logits=model(**batch).logits; loss=loss_fn(logits.reshape(-1,9),labels.reshape(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); losses.append(loss.item()); steps+=1
            if smoke_steps and steps>=smoke_steps: break
        gold,pred=predict_samples(model,validation_samples,batch_size,device); metrics=evaluate_sequences(gold,pred)
        f1=metrics["entity_micro"]["f1"]; history.append({"epoch":epoch,"loss":sum(losses)/len(losses),"validation_entity_f1":f1,"steps":steps})
        if f1>best:
            best=f1; model.save_pretrained(output); tokenizer.save_pretrained(output)
        print(json.dumps(history[-1]),flush=True)
        if smoke_steps: break
    manifest={"seed":seed,"initial_model":initial_model,"smoke_only":bool(smoke_steps),"epochs_requested":epochs,"train_examples":len(samples),"validation_examples":len(validation_samples),"steps":steps,"learning_rate":2e-5,"batch_size":batch_size,"max_length":128,"class_weights_train_only":weights.tolist(),"history":history,"checkpoint_selection":"best validation exact entity F1; test not accessed","device":device,"prepared_manifest":json.loads((Path(prepared)/"manifest.json").read_text()),"environment":environment()}
    write_json(output/"run_manifest.json",manifest)
    return manifest

def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("audit"); p.add_argument("--output",default=str(ROOT/"artifacts/audit"))
    p=sub.add_parser("prepare"); p.add_argument("--csv",default=str(ROOT/"data.csv")); p.add_argument("--output",default=str(ROOT/"artifacts/prepared")); p.add_argument("--seed",type=int,default=42)
    p=sub.add_parser("evaluate"); p.add_argument("--model",default=str(ROOT/"bert_product_model")); p.add_argument("--output",default=str(ROOT/"artifacts/legacy-evaluation")); p.add_argument("--limit",type=int,default=0); p.add_argument("--protocol",choices=["legacy","full-context-diagnostic"],default="legacy"); p.add_argument("--prepared")
    p=sub.add_parser("train"); p.add_argument("--prepared",default=str(ROOT/"artifacts/prepared")); p.add_argument("--output",required=True); p.add_argument("--initial-model",default="bert-base-uncased"); p.add_argument("--epochs",type=int,default=3); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--smoke-steps",type=int,default=0)
    args=parser.parse_args()
    if args.command=="audit": audit(args.output)
    elif args.command=="prepare": print(json.dumps(prepare_corpus(args.csv,args.output,args.seed),indent=2))
    elif args.command=="evaluate": evaluate(args.model,args.output,args.limit,args.protocol,args.prepared)
    else: train(args.prepared,args.output,args.initial_model,args.epochs,args.batch_size,args.smoke_steps)
