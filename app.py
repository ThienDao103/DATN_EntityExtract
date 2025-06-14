from flask import Flask, render_template, request
import pandas as pd
from transformers import BertTokenizerFast, BertForTokenClassification
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model_path = r"C:\Users\Admin\Desktop\ĐH\DATN\bert_7_2\bert_product_model"
model = BertForTokenClassification.from_pretrained(model_path, trust_remote_code=True)
tokenizer = BertTokenizerFast.from_pretrained(model_path)

# Label mapping
id2label = {
    0: 'B-Color', 1: 'B-Dimensions', 2: 'B-Material', 3: 'B-Weight',
    4: 'I-Color', 5: 'I-Dimensions', 6: 'I-Material', 7: 'I-Weight', 8: 'O'
}

# Load product data
df = pd.read_csv("products_labeled.csv")
attr_cols = ["color", "material", "dimensions", "weight"]
df[attr_cols] = df[attr_cols].fillna("")
encoder = OneHotEncoder()
attr_vectors = encoder.fit_transform(df[attr_cols]).toarray()

# Extract attributes from description
def predict_product_info(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    output = model(**tokens).logits
    preds = output.argmax(-1)[0].tolist()
    input_ids = tokens['input_ids'][0].tolist()
    tokens_decoded = tokenizer.convert_ids_to_tokens(input_ids)

    entities = {}
    current_entity = ""
    current_label = None

    for token, pred in zip(tokens_decoded, preds):
        label = id2label[pred]
        if label == "O":
            if current_label:
                entities.setdefault(current_label, []).append(current_entity.strip())
                current_entity, current_label = "", None
            continue
        tag, ent_type = label.split("-")
        if tag == "B":
            if current_label:
                entities.setdefault(current_label, []).append(current_entity.strip())
            current_entity = token.replace("##", "")
            current_label = ent_type
        elif tag == "I" and ent_type == current_label:
            current_entity += " " + token.replace("##", "")
    if current_entity and current_label:
        entities.setdefault(current_label, []).append(current_entity.strip())
    return {k.lower(): " ".join(v) for k, v in entities.items()}

# Recommend similar products
def recommend_from_attributes(color="", material="", dimensions="", weight="", top_k=5):
    input_df = pd.DataFrame([[color, material, dimensions, weight]], columns=attr_cols)
    try:
        input_vector = encoder.transform(input_df).toarray()
    except ValueError:
        return pd.DataFrame()  # Trả về rỗng nếu không vector hóa được (vd: giá trị không tồn tại)
    
    sims = cosine_similarity(input_vector, attr_vectors)[0]
    top_indices = sims.argsort()[::-1][:top_k]

    result_df = df.iloc[top_indices][["description"] + attr_cols].copy()
    result_df["similarity"] = sims[top_indices]
    return result_df

# Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    extracted = {}
    recommendations = []
    message = ""

    if request.method == "POST":
        if "desc_submit" in request.form:
            desc = request.form.get("description", "").strip()
            if not desc:
                message = "Please enter a product description."
            else:
                extracted = predict_product_info(desc)

        elif "rec_submit" in request.form:
            rec_description = request.form.get("rec_description", "").strip()
            color = request.form.get("color", "").strip()
            material = request.form.get("material", "").strip()
            dimensions = request.form.get("dimensions", "").strip()
            weight = request.form.get("weight", "").strip()

            if not any([rec_description, color, material, dimensions, weight]):
                message = "Please enter at least one attribute."
            else:
                # Nếu có mô tả sản phẩm, trích xuất thuộc tính từ đó
                if rec_description:
                    extracted_attr = predict_product_info(rec_description)
                    color = color or extracted_attr.get("color", "")
                    material = material or extracted_attr.get("material", "")
                    dimensions = dimensions or extracted_attr.get("dimensions", "")
                    weight = weight or extracted_attr.get("weight", "")

                rec_df = recommend_from_attributes(color, material, dimensions, weight)
                if rec_df.empty:
                    message = "No matching products found."
                else:
                    recommendations = rec_df.to_dict(orient="records")

    return render_template(
        "index.html",
        extracted=extracted,
        recommendations=recommendations,
        message=message,
        request=request  # giữ lại dữ liệu nhập trong form
    )

if __name__ == "__main__":
    app.run(debug=True)
