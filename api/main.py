from fastapi import FastAPI
from pydantic import BaseModel
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
import sys

sys.path.append("B:/arabic-darija-chatbot")
from nlp.ner.medication_extractor import get_medication_info
from nlp.intent_detection.intent_classifier import detect_intent

# ── Charger Atlas depuis HuggingFace ─────────────────
print("Chargement du modèle Atlas...")
MODEL_NAME = "Chaymaae/darija-medical-bert"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
print("Modèle chargé ✅")

# ── Labels ───────────────────────────────────────────
LABELS = [
    "Rheumatology and Orthopedics", "Otorhinolaryngology",
    "Dermatology", "general practitioner", "Psychiatry",
    "ophthalmologist", "Cardiology", "Internal Medicine",
    "Hematology", "Respiratory diseases", "Dentistry",
    "Oncology", "Infectious Diseases", "Anesthesiology",
    "Cosmetic dermatologist", "Neurology", "Pediatric Medicine",
    "Diabetes mellitus", "Dietetics / Nutrition", "gynecologists",
    "Endocrinology", "Allergy and Immunology", "Mental Health",
    "General Medicine"
]

app = FastAPI(title="Arabic/Darija Medical Chatbot API")

class QuestionRequest(BaseModel):
    question:       str
    age:            str = ""
    allergy:        str = ""
    other_symptoms: str = ""
    duration:       str = ""
    ui_language:    str = "ar"

def detect_language(text):
    try:
        lang = detect(text)
        if lang == 'ar':
            return 'arabic'
        elif lang == 'fr':
            return 'french'
        else:
            return 'darija'
    except:
        return 'darija'

def translate_to_arabic(question, language):
        return question

def predict_specialty(question, language):
    # Mapping rapide français → symptôme arabe
    fr_ar_quick = {
        "fievre": "حمى", "fièvre": "حمى",
        "toux": "سعال", "douleur": "ألم",
        "nausee": "غثيان", "nausée": "غثيان",
        "allergie": "حساسية", "infection": "التهاب",
        "diabete": "سكر", "diabète": "سكر",
        "asthme": "ربو", "anxiete": "قلق",
    }
    
    q = question.lower()
    for fr, ar in fr_ar_quick.items():
        q = q.replace(fr, ar)
    
    inputs = tokenizer(
        q,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    confidence = round(conf.item(), 4)
    # Confiance trop faible → médecin généraliste plutôt qu'une spécialité erronée
    if confidence < 0.5:
        return "General Medicine", confidence
    return LABELS[pred.item()], confidence

@app.get("/")
def home():
    return {"message": "Chatbot API is running ✅"}

@app.post("/predict")
def predict(request: QuestionRequest):
    question = request.question

    # Enrichir le contexte avec les autres symptômes pour une meilleure prédiction
    full_context = question
    if request.other_symptoms:
        full_context += f" {request.other_symptoms}"

    language = detect_language(full_context)
    specialty, confidence = predict_specialty(full_context, language)
    intent = detect_intent(question)

    try:
        med_info = get_medication_info(
            question,
            age=request.age,
            allergy=request.allergy,
            other_symptoms=request.other_symptoms,
            language=request.ui_language
        )
        if not med_info:
            med_info = {"symptom_detected": None, "medications": []}
    except Exception:
        med_info = {"symptom_detected": None, "medications": []}

    return {
        "question":         question,
        "age":              request.age,
        "allergy":          request.allergy,
        "duration":         request.duration,
        "other_symptoms":   request.other_symptoms,
        "language":         language,
        "intent":           intent,
        "specialty":        specialty,
        "confidence":       confidence,
        "symptom_detected": med_info.get("symptom_detected"),
        "medications":      med_info.get("medications", [])
    }