"""Exact entity boundaries under strict IOB2, distinct from token averages."""
from collections import Counter
from . import ATTRIBUTES,LABELS

def bio_entities(tags):
    found=set(); start=kind=None
    for i,tag in enumerate(list(tags)+["O"]):
        prefix,_,label=tag.partition("-")
        if kind is not None and not (prefix=="I" and label==kind):
            found.add((kind,start,i)); start=kind=None
        if prefix=="B" and label in ATTRIBUTES:
            start,kind=i,label
    return found  # An orphan I does not start a strict IOB2 entity.

def score(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0.; r=tp/(tp+fn) if tp+fn else 0.
    return {"precision":p,"recall":r,"f1":2*p*r/(p+r) if p+r else 0.,"tp":tp,"fp":fp,"fn":fn,"support":tp+fn}

def evaluate_sequences(references,predictions):
    from sklearn.metrics import classification_report,confusion_matrix
    if not references or len(references)!=len(predictions):
        raise ValueError("Nonempty paired sequences required")
    counts={label:Counter() for label in ATTRIBUTES}; flat_ref=[]; flat_pred=[]; invalid_i=0
    for actual,predicted in zip(references,predictions):
        if len(actual)!=len(predicted) or any(t not in LABELS for t in list(actual)+list(predicted)):
            raise ValueError("Invalid paired BIO sequences")
        gold,guessed=bio_entities(actual),bio_entities(predicted)
        for label in ATTRIBUTES:
            a={s for s in gold if s[0]==label}; b={s for s in guessed if s[0]==label}
            counts[label].update(tp=len(a&b),fp=len(b-a),fn=len(a-b))
        invalid_i+=sum(tag.startswith("I-") and (i==0 or predicted[i-1] not in ("B-"+tag[2:],tag)) for i,tag in enumerate(predicted))
        flat_ref.extend(actual); flat_pred.extend(predicted)
    per={label:score(c["tp"],c["fp"],c["fn"]) for label,c in counts.items()}; total=Counter()
    for c in counts.values(): total.update(c)
    return {"examples":len(references),"evaluated_tokens":len(flat_ref),"entity_micro":score(total["tp"],total["fp"],total["fn"]),"entity_macro_f1":sum(v["f1"] for v in per.values())/4,"per_attribute":per,"invalid_prediction_I_tags":invalid_i,"token_report":classification_report(flat_ref,flat_pred,labels=LABELS,output_dict=True,zero_division=0),"confusion_labels":LABELS,"confusion_matrix":confusion_matrix(flat_ref,flat_pred,labels=LABELS).tolist(),"entity_definition":"Exact type and token start/end positions, strict IOB2; O excluded from entity scores"}
