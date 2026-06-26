from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
from nltk.corpus import stopwords

app = FastAPI(title="Email Lead Detection API")

model = joblib.load("email_classifier.pkl")

stop_words = set(stopwords.words("english"))

def process_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    tokens = text.split()

    tokens = [
        word
        for word in tokens
        if word not in stop_words
        and len(word) > 2
    ]

    return " ".join(tokens)


class EmailRequest(BaseModel):
    name: str = ""
    email: str = ""
    body: str


@app.get("/")
def home():
    return {
        "status": "running"
    }


@app.post("/predict")
def predict(request: EmailRequest):

    text = f"{request.name} {request.email} {request.body}"

    clean = process_text(text)

    prediction = model.predict([clean])[0]

    probabilities = model.predict_proba([clean])[0]

    score = round(max(probabilities) * 100)

    return {
        "classification": prediction,
        "score": score,
        "probabilities": {
            cls: round(float(prob), 4)
            for cls, prob in zip(model.classes_, probabilities)
        }
    }