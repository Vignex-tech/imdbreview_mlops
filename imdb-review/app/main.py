import logging
import time
import joblib
from fastapi import FastAPI
from pydantic import BaseModel , Field


logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(message)s')
log = logging.getLogger('imdb-review')

app = FastAPI(title = "IMDB Review Sentiment Analysis", description = "A simple API to predict the sentiment of IMDB reviews using a pre-trained model.", version = "1.0.0")

model = joblib.load("model/model.joblib")

class Review(BaseModel):
    text : str = Field(..., example = "This movie was fantastic! The acting was great and the plot was very engaging.", min_length = 1, max_length = 1000, description = "The text of the IMDB review to be analyzed.")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict_sentiment(review: Review):
    start = time.perf_counter()
    positive = model.predict_proba([review.text])[0][1]
    log.info(f"Predicted probability of positive sentiment: {positive:.4f}")
    if positive > 0.5:
        sentiment = "positive"
    else:
        sentiment = "negative"
    latency = (time.perf_counter() - start) * 1000
    log.info(f"Predicted sentiment: {sentiment} with probability {positive:.4f} in {latency:.2f} ms character length: {len(review.text)}")
    return {"sentiment": sentiment, "probability": positive, "latency_ms": latency}