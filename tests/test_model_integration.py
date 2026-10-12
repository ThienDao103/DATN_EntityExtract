"""Explicit real-checkpoint integration test: opt in with RUN_MODEL_TESTS=1."""
import os
from pathlib import Path
import pytest
from entityextract.inference import EntityExtractor
from app import create_app

@pytest.mark.skipif(os.getenv("RUN_MODEL_TESTS")!="1",reason="Set RUN_MODEL_TESTS=1 for checkpoint integration")
def test_real_model_offsets_windows_and_api():
    root=Path(__file__).resolve().parents[1]
    model=EntityExtractor(root/"bert_product_model")
    text="This shirt is 30 mm x 50 mm with red color and weight 2 kg."
    result=model.extract(text)
    assert not model.model.training
    assert result["attributes"] and result["windows"]==1
    for entity in result["entities"]:
        assert entity["text"]==text[entity["start"]:entity["end"]]
    long_text=("A normal product description with several details. "*45)+text
    long_result=model.extract(long_text)
    assert long_result["windows"]>1
    assert long_result["processed_characters"]==len(long_text)
    client=create_app({"TESTING":True},extractor=model).test_client()
    response=client.post("/api/extract",json={"description":text})
    assert response.status_code==200 and response.json["entities"]
