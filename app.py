"""Flask UI and JSON endpoints; paths are independent of the working directory."""
import logging
import os
from pathlib import Path
from threading import Lock
from flask import Flask,jsonify,render_template,request
from entityextract.inference import EntityExtractor
from entityextract.recommendation import ProductRecommender
ROOT=Path(__file__).resolve().parent

def create_app(config=None,extractor=None,recommender=None):
    app=Flask(__name__)
    app.config.update(MODEL_DIR=os.getenv("ENTITYEXTRACT_MODEL_DIR",str(ROOT/"bert_product_model")),CATALOG_PATH=os.getenv("ENTITYEXTRACT_CATALOG_PATH",str(ROOT/"products_labeled.csv")),DEVICE=os.getenv("ENTITYEXTRACT_DEVICE","cpu"),MAX_CONTENT_LENGTH=100_000,MAX_DESCRIPTION_LENGTH=20_000)
    if config: app.config.update(config)
    app.extensions.update(extractor=extractor,recommender=recommender); loading=Lock()
    def services(model=False,catalog=False):
        with loading:
            if model and app.extensions["extractor"] is None: app.extensions["extractor"]=EntityExtractor(app.config["MODEL_DIR"],device=app.config["DEVICE"])
            if catalog and app.extensions["recommender"] is None: app.extensions["recommender"]=ProductRecommender(app.config["CATALOG_PATH"])
        return app.extensions["extractor"],app.extensions["recommender"]
    def description(value,required=True):
        if not isinstance(value,str): raise ValueError("description must be a string")
        if required and not value.strip(): raise ValueError("Please enter a product description")
        if len(value)>app.config["MAX_DESCRIPTION_LENGTH"]: raise ValueError("Description exceeds 20,000 characters")
        return value.strip()
    def recommend(payload):
        attrs=payload.get("attributes",{})
        if not isinstance(attrs,dict) or any(not isinstance(v,str) for v in attrs.values()): raise ValueError("attributes must be an object of strings")
        if set(attrs)-{"color","material","dimensions","weight"}: raise ValueError("Unknown product attribute")
        if any(len(v)>500 for v in attrs.values()): raise ValueError("Attribute value exceeds 500 characters")
        text=description(payload.get("description",""),False); k=payload.get("top_k",5)
        if isinstance(k,bool) or not isinstance(k,int) or not 1<=k<=20: raise ValueError("top_k must be an integer between 1 and 20")
        if text:
            extractor,_=services(model=True); extracted=extractor.extract(text)["attributes"]
            attrs={key:attrs.get(key) or extracted.get(key,"") for key in ("color","material","dimensions","weight")}
        if not any(v.strip() for v in attrs.values()): raise ValueError("Enter a description or at least one attribute")
        _,recommender=services(catalog=True)
        return attrs,recommender.recommend(attrs,k)
    @app.get("/health")
    def health():
        return jsonify(status="ok",model_loaded=app.extensions["extractor"] is not None,catalog_loaded=app.extensions["recommender"] is not None)
    @app.post("/api/extract")
    def extract_api():
        try:
            payload=request.get_json(silent=True)
            if not isinstance(payload,dict): raise ValueError("Request body must be a JSON object")
            text=description(payload.get("description")); extractor,_=services(model=True)
            return jsonify(extractor.extract(text))
        except ValueError as exc: return jsonify(error=str(exc)),400
        except (OSError,RuntimeError):
            app.logger.exception("Model unavailable"); return jsonify(error="Model unavailable; check local configuration and server logs"),503
    @app.post("/api/recommend")
    def recommend_api():
        try:
            payload=request.get_json(silent=True)
            if not isinstance(payload,dict): raise ValueError("Request body must be a JSON object")
            attrs,results=recommend(payload); return jsonify(attributes=attrs,recommendations=results)
        except ValueError as exc: return jsonify(error=str(exc)),400
        except (OSError,RuntimeError):
            app.logger.exception("Recommendation unavailable"); return jsonify(error="Recommendation unavailable; check configuration and server logs"),503
    @app.errorhandler(413)
    def too_large(error): return jsonify(error="Request is too large"),413
    @app.route("/",methods=["GET","POST"])
    def index():
        extracted,recommendations,message={},[],""
        if request.method=="POST":
            try:
                if "desc_submit" in request.form:
                    text=description(request.form.get("description","")); extractor,_=services(model=True); extracted=extractor.extract(text)["attributes"]
                elif "rec_submit" in request.form:
                    extracted,recommendations=recommend({"description":request.form.get("rec_description",""),"attributes":{k:request.form.get(k,"") for k in ("color","material","dimensions","weight")}})
                    if not recommendations: message="No matching products found. Try an attribute present in the catalog."
                else: raise ValueError("Choose Extract or Recommend")
            except ValueError as exc: message=str(exc)
            except (OSError,RuntimeError):
                app.logger.exception("Service unavailable"); message="Service unavailable. Check the model and catalog configuration."
        return render_template("index.html",extracted=extracted,recommendations=recommendations,message=message,request=request)
    return app

app=create_app()
if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="127.0.0.1",port=int(os.getenv("PORT","5000")),debug=False)
