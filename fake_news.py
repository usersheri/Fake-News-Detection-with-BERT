import os
os.environ['TF_USE_LEGACY_KERAS'] = '0' 
import tensorflow as tf
from transformers import AutoTokenizer, TFBertModel
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn

# -------------------------
# BERT WRAPPER (SAME AS TRAINING)
# -------------------------
class BertWrapper(tf.keras.layers.Layer):
    def __init__(self, bert_model=None, **kwargs):
        super().__init__(**kwargs)
        self.bert = bert_model

    def call(self, inputs):
        outputs = self.bert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            training=False
        )
        return outputs.last_hidden_state[:, 0, :]

    @classmethod
    def from_config(cls, config):
        bert = TFBertModel.from_pretrained(
            "bert-base-uncased",
            use_safetensors=False
        )
        return cls(bert_model=bert, **config)


# -------------------------
# LOAD MODEL & TOKENIZER
# -------------------------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = tf.keras.models.load_model(
    "bert_fakenews_model.keras",
    custom_objects={
        "BertWrapper": BertWrapper,
        "TFBertModel": TFBertModel
    }
)

print("Model loaded successfully!")


# -------------------------
# FASTAPI APP
# -------------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")


# -------------------------
# HOME PAGE
# -------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# -------------------------
# PREDICTION API
# -------------------------
@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    text = data.get("text", "")

    if not text.strip():
        return JSONResponse({"error": "Empty text"})

    encodings = tokenizer(
        text,
        max_length=256,
        truncation=True,
        padding="max_length",
        return_tensors="tf"
    )

    inputs = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"]
    }

    prediction = model.predict(inputs, verbose=0)
    prob = float(prediction[0][0])
    label = "FAKE" if prob >= 0.5 else "REAL"

    return {
        "label": label,
        "probability": round(prob, 4)
    }


# -------------------------
# RUN LOCALLY
# -------------------------
if __name__ == "__main__":
    uvicorn.run("fake_news:app", host="127.0.0.1", port=8000, reload=True)
