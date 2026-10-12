"""Offline BERT extraction with original-text offsets and overlapping windows."""
from pathlib import Path
from threading import Lock
from . import ATTRIBUTES

def decode_spans(text,offsets,tags,confidence=None):
    if len(offsets)!=len(tags): raise ValueError("Mismatched offsets and tags")
    confidence=confidence if confidence is not None else [1.]*len(tags)
    entities=[]; active=None
    def finish():
        nonlocal active
        if active:
            active["text"]=text[active["start"]:active["end"]]
            values=active.pop("scores"); active["confidence"]=sum(values)/len(values)
            entities.append(active); active=None
    for (start,end),tag,prob in zip(offsets,tags,confidence):
        if start==end: continue
        prefix,_,label=tag.partition("-")
        if label not in ATTRIBUTES or prefix not in ("B","I"):
            finish(); continue
        contiguous=active and active["label"]==label and not text[active["end"]:start].strip()
        if prefix=="B" or not contiguous:
            finish(); active={"label":label,"start":int(start),"end":int(end),"scores":[float(prob)]}
        else:
            active["end"]=int(end); active["scores"].append(float(prob))
    finish()
    return entities  # Orphan I repaired for display only, not strict evaluation.

class EntityExtractor:
    def __init__(self,model_dir,device="cpu",max_length=128,stride=32):
        import torch
        from transformers import BertForTokenClassification,BertTokenizerFast
        self.torch=torch; torch.set_num_threads(min(4,max(1,torch.get_num_threads())))
        model_dir=Path(model_dir); weights=model_dir/"model.safetensors"
        if not weights.is_file(): raise FileNotFoundError(f"Checkpoint not found: {weights}")
        with weights.open("rb") as f:
            if f.read(40).startswith(b"version https://git-lfs"): raise RuntimeError("Run git lfs pull to download checkpoint")
        self.tokenizer=BertTokenizerFast.from_pretrained(model_dir,local_files_only=True)
        self.model=BertForTokenClassification.from_pretrained(model_dir,local_files_only=True).to(device).eval()
        self.id2label=self.model.config.id2label
        if set(self.id2label.values())!={"O"}|{p+a for a in ATTRIBUTES for p in ("B-","I-")}:
            raise ValueError("Checkpoint labels mismatch")
        if max_length>self.model.config.max_position_embeddings or not 0<=stride<max_length-2:
            raise ValueError("Invalid window length or overlap")
        self.device,self.max_length,self.stride=device,max_length,stride; self.lock=Lock()

    def extract(self,text):
        if not isinstance(text,str) or not text.strip(): raise ValueError("Nonempty description required")
        if len(text)>20000: raise ValueError("Description exceeds 20,000 characters")
        # Build windows explicitly: saved tokenizers and Transformers versions may
        # disagree about overflow handling. Offset coverage is tested end to end.
        with self.lock:
            full=self.tokenizer(text,add_special_tokens=False,truncation=False,return_offsets_mapping=True)
        capacity=self.max_length-2; step=capacity-self.stride; windows=[]; offsets=[]
        for start in range(0,len(full["input_ids"]),step):
            ids=full["input_ids"][start:start+capacity]
            positions=full["offset_mapping"][start:start+capacity]
            ids=[self.tokenizer.cls_token_id]+ids+[self.tokenizer.sep_token_id]
            positions=[(0,0)]+positions+[(0,0)]
            padding=self.max_length-len(ids)
            windows.append({"input_ids":ids+[self.tokenizer.pad_token_id]*padding,"attention_mask":[1]*len(ids)+[0]*padding})
            offsets.append(positions+[(0,0)]*padding)
            if start+capacity>=len(full["input_ids"]): break
        encoded={key:self.torch.tensor([w[key] for w in windows]) for key in ("input_ids","attention_mask")}
        predictions={}
        with self.lock,self.torch.inference_mode():
            for start in range(0,len(offsets),8):
                batch={k:v[start:start+8].to(self.device) for k,v in encoded.items()}
                probabilities=self.model(**batch).logits.softmax(-1).cpu(); scores,labels=probabilities.max(-1)
                for row in range(len(labels)):
                    for offset,label,prob in zip(offsets[start+row],labels[row].tolist(),scores[row].tolist()):
                        key=tuple(offset)
                        if key[0]==key[1]: continue
                        if key not in predictions or prob>predictions[key][1]: predictions[key]=(self.id2label[label],prob)
        ordered=sorted(predictions)
        entities=decode_spans(text,ordered,[predictions[k][0] for k in ordered],[predictions[k][1] for k in ordered])
        attributes={}
        for entity in entities:
            key=entity["label"].lower(); values=attributes.setdefault(key,[])
            if entity["text"].casefold() not in [v.casefold() for v in values]: values.append(entity["text"])
        return {"attributes":{k:", ".join(v) for k,v in attributes.items()},"entities":entities,"windows":len(offsets),"processed_characters":ordered[-1][1] if ordered else 0,"model":self.model.config.model_type}
