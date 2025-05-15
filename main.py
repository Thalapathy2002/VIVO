from fastapi import FastAPI
from pydantic import BaseModel
from model import predict_emotion
from prompts import get_ai_therapist_response

app = FastAPI()

class JournalEntry(BaseModel):
    text: str

@app.post("/analyze")
def analyze(entry: JournalEntry):
    emotion = predict_emotion(entry.text)
    ai_response = get_ai_therapist_response(entry.text, emotion)
    return {
        "emotion": emotion,
        "therapist_reply": ai_response
    }
