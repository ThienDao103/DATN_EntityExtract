"""Sparse content-based matching with normalized values and unknown handling."""
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
ATTRIBUTE_COLUMNS=("color","material","dimensions","weight")

def normalize_attribute(value):
    if pd.isna(value): return ""
    return re.sub(r"\s+"," ",str(value)).strip().casefold()

class ProductRecommender:
    def __init__(self,catalog):
        self.frame=pd.read_csv(catalog) if isinstance(catalog,(str,Path)) else catalog.copy()
        if not set(ATTRIBUTE_COLUMNS).issubset(self.frame.columns) or "description" not in self.frame:
            raise ValueError("Catalog requires description, color, material, dimensions and weight")
        self.frame=self.frame.dropna(subset=["description"]).reset_index(drop=True)
        if self.frame.empty: raise ValueError("Empty catalog")
        normalized=self.frame[list(ATTRIBUTE_COLUMNS)].map(normalize_attribute)
        self.encoder=OneHotEncoder(handle_unknown="ignore",sparse_output=True,dtype=np.float32)
        self.vectors=self.encoder.fit_transform(normalized).tocsc()
        # Zero empty-category columns without densifying the catalog.
        for j,categories in enumerate(self.encoder.categories_):
            base=sum(len(c) for c in self.encoder.categories_[:j]); empty=np.flatnonzero(categories=="")
            if len(empty):
                column=base+int(empty[0]); self.vectors.data[self.vectors.indptr[column]:self.vectors.indptr[column+1]]=0
        self.vectors.eliminate_zeros(); self.vectors=self.vectors.tocsr()

    def recommend(self,attributes,top_k=5):
        if isinstance(top_k,bool) or not isinstance(top_k,int) or not 1<=top_k<=20:
            raise ValueError("top_k must be an integer between 1 and 20")
        values={c:normalize_attribute(attributes.get(c,"")) for c in ATTRIBUTE_COLUMNS}
        if not any(values.values()): return []
        query=self.encoder.transform(pd.DataFrame([values],columns=ATTRIBUTE_COLUMNS)).tolil()
        for j,column in enumerate(ATTRIBUTE_COLUMNS):
            if not values[column]:
                base=sum(len(c) for c in self.encoder.categories_[:j]); query[:,base:base+len(self.encoder.categories_[j])]=0
        query=query.tocsr(); query.eliminate_zeros()
        if not query.nnz: return []
        scores=cosine_similarity(query,self.vectors).ravel(); ranked=np.argsort(-scores,kind="stable"); results=[]; seen=set()
        for i in ranked:
            if scores[i]<=0: break
            row=self.frame.iloc[i]; key=normalize_attribute(row["description"])
            if key in seen: continue
            seen.add(key); result={"description":str(row["description"]),"similarity":float(scores[i])}
            result.update({c:"" if pd.isna(row[c]) else str(row[c]) for c in ATTRIBUTE_COLUMNS}); results.append(result)
            if len(results)==top_k: break
        return results
