import pandas as pd
import pytest
from pathlib import Path
from uuid import uuid4
from app import create_app
from entityextract.data import rule_spans,words_and_tags,prepare_corpus,read_jsonl
from entityextract.inference import decode_spans
from entityextract.metrics import evaluate_sequences,bio_entities
from entityextract.recommendation import ProductRecommender

@pytest.fixture
def tmp_path():
    # Inherited Windows ACLs; pytest's 0700 temporary directories fail in this sandbox.
    path=Path(__file__).resolve().parents[1]/"artifacts/test-work"/uuid4().hex
    path.mkdir(parents=True)
    return path

def test_offset_decode_wordpieces_units():
    found=decode_spans("microfiber 2.5 kg",[(0,5),(5,10),(11,14),(15,17)],["B-Material","I-Material","B-Weight","I-Weight"])
    assert [s["text"] for s in found]==["microfiber","2.5 kg"]
    assert found[0]["start"]==0 and found[0]["end"]==10

def test_special_tokens_and_orphan_i():
    assert decode_spans("red",[(0,0),(0,3)],["O","I-Color"])[0]["text"]=="red"
    assert bio_entities(["I-Color"])==set()

def test_wrong_boundary_vs_token_accuracy():
    result=evaluate_sequences([["B-Weight","I-Weight","O","O"]],[["B-Weight","O","O","O"]])
    assert result["entity_micro"]["f1"]==0 and result["token_report"]["accuracy"]==.75

def test_rules_multicolors_dimensions_not_item_counts():
    text="Red and blue stainless steel box, 10 x 20 cm, weight 2.5 kg; pack of 3."
    spans=rule_spans(text)
    assert {s["text"] for s in spans if s["label"]=="Color"}=={"Red","blue"}
    assert [s["text"] for s in spans if s["label"]=="Weight"]==["2.5 kg"]
    words,tags=words_and_tags(text,spans)
    assert len(words)==len(tags) and "B-Dimensions" in tags and "I-Material" in tags

def test_dedup_and_full_context(tmp_path):
    path=tmp_path/"input.csv"
    pd.DataFrame({"description":["Red cotton shirt"," RED   cotton shirt "]+[f"Blue box {i}" for i in range(20)]}).to_csv(path,index=False)
    manifest=prepare_corpus(path,tmp_path/"prepared")
    assert manifest["duplicate_descriptions_removed"]==1
    splits=[read_jsonl(tmp_path/"prepared"/f"{s}.jsonl") for s in ("train","validation","test")]
    keys=[{r["id"] for r in split} for split in splits]
    assert not keys[0]&keys[1] and not keys[0]&keys[2] and not keys[1]&keys[2]
    red=next(r for split in splits for r in split if r["text"]=="Red cotton shirt")
    assert red["words"]==["Red","cotton","shirt"] and red["tags"]==["B-Color","B-Material","O"]

@pytest.fixture
def recommender():
    return ProductRecommender(pd.DataFrame([
        {"description":"red cotton shirt","color":"Red","material":"cotton","dimensions":"","weight":""},
        {"description":"blue plastic box","color":"blue","material":"plastic","dimensions":"","weight":""},
        {"description":"red cotton shirt","color":"Red","material":"cotton","dimensions":"","weight":""}]))

def test_sparse_unseen_missing_case_and_duplicate(recommender):
    assert recommender.recommend({"color":" RED "})[0]["description"]=="red cotton shirt"
    assert len(recommender.recommend({"color":"red"}))==1
    assert recommender.recommend({"color":"unseen-color"})==[] and recommender.recommend({})==[]
    assert recommender.vectors.nnz==6

class FakeExtractor:
    def extract(self,text): return {"attributes":{"color":"red"},"entities":[],"windows":1,"model":"test"}

@pytest.fixture
def client(recommender):
    return create_app({"TESTING":True},extractor=FakeExtractor(),recommender=recommender).test_client()

@pytest.mark.parametrize("payload",[None,{}, {"description":42},{"description":""},{"description":"a"*20001}])
def test_extract_validation(client,payload):
    assert client.post("/api/extract",json=payload).status_code==400

def test_api_and_recommendation_without_description(client):
    assert client.get("/health").json["status"]=="ok"
    assert client.post("/api/extract",json={"description":"red shirt"}).json["attributes"]=={"color":"red"}
    response=client.post("/api/recommend",json={"attributes":{"color":"red"}})
    assert response.status_code==200 and len(response.json["recommendations"])==1
    assert b"red cotton shirt" in client.post("/",data={"rec_submit":"","color":"red"}).data

@pytest.mark.parametrize("payload",[{"attributes":[]},{"attributes":{"bad":"red"}},{"attributes":{"color":1}},{"attributes":{},"top_k":True},{"attributes":{"color":"red"},"top_k":0},{"attributes":{"color":"red"},"top_k":21}])
def test_recommend_validation(client,payload):
    assert client.post("/api/recommend",json=payload).status_code==400

def test_two_forms_html_escape(client):
    assert client.get("/").data.count(b'<form method="post"')==2
    page=client.post("/",data={"desc_submit":"","description":"<script>alert(1)</script>"}).data
    assert b"&lt;script&gt;" in page and b"<script>alert(1)</script>" not in page
