from fastapi import FastAPI, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import urllib.parse
import requests
import smtplib
from email.message import EmailMessage
import re


app = FastAPI()

# Enable CORS if frontend is separate
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- LOAD ASSETS -----------------

# Keywords per aspect
aspect_keywords = {
    "doctor": ["dermatologist", "surgeon", "specialist", "doctors", "dentist", "endocrinology",
               "hospitalist", "neuro", "pulmonologist", "ctvs", "cardiac", "pediatrician", "intensivist",
               "md", "doctor", "physician", "immunologist", "dr", "practitioner", "consultant", "advisor",
               "registrar", "clinician", "resident", "urologist", "acupuncturist", "radiologist",
               "geriatrician", "pathologist", "anesthesiologist", "cardiologist", "cosmetologist",
               "endocrinologist", "gastroenterologist", "hepatologist", "gynecologist", "obstetrician",
               "oncologist", "neurologist", "ophthalmologist", "optometrist", "psychiatrist",
               "psychologist", "therapist", "rheumatologist", "nephrologist", "otolaryngologist",
               "hematologist", "toxicologist", "orthopedist", "podiatrist", "vaidyar", "rmo", "doc", "consultanting",
               "doctor", "consultation", "rude doctor", "good doctor", "bad doctor", "unprofessional",
               "treatment"],

    "staff": ["staff", "employee", "chemist", "nurse", "team", "staffs", "manager", "nursing",
              "technician", "guide", "employees", "providers", "workers", "professionals",
              "responders", "assistant", "housekeeping", "aayas", "reception", "admin",
              "receptionist", "matron", "groupd", "midwife", "pharmacist", "dietician",
              "nutritionist", "administrator", "executive", "clerk", "coordinator", "officer",
              "secretary", "engineer", "wardboy", "driver", "security", "personnel", "compounder",
              "perfusionist", "counselor", "guard", "nurses", "caregiver", "crew", "therapists",
              "facilitator", "faculty", "attendants", "rude", "nurse", "nurses", "icu", "behavior", "attitude", "staff",
              "rude", "impolite", "nurse", "bad service", "worst staff", "behavior", "irresponsible"],

    "cost": ["affordable", "expensive", "reasonable", "cost", "budget", "overpriced", "costeffective",
             "fare", "pricing", "fees", "payment", "insurance", "out-of-pocket", "costly", "premium",
             "luxurious", "billing", "pricey", "charges", "price", "free", "bills", "inexpensive",
             "cheap", "pocketfriendly", "value", "charge", "amount", "amounts", "bill", "expenditure",
             "finance", "penny", "coverage", "deductible", "negotiable", "overpaying", "clearance",
             "money", "economical", "rates", "minimum", "card", "cash", "costed", "priced", "rupees",
             "paid", "discount", "charging", "charged", "rs", "rps"],

    "waitingtime": ["delay", "stop", "wait", "remain", "long", "queue", "slow", "check-in", "hurried",
                    "crowded", "crowd", "busy", "late", "rush", "fast", "schedule", "line", "turnaround",
                    "waiting", "duration", "time", "speedy", "timely", "stagnation", "unhurried", "hurry",
                    "rapid", "pause", "scheduling", "sequence", "tardy", "hold", "backlog", "lag", "halt",
                    "stall", "quick", "postponed", "postpone", "response", "linger", "quickly", "hasty",
                    "minutes", "minute", "hour", "hours", "second", "seconds"],

    "cleanliness": ["clean", "aseptic", "antiseptic", "airy", "aromatic", "aesthtic", "decontamination",
                    "purity", "ablution", "neat", "tidy", "spotless", "infection", "sanitary", "hygienic",
                    "germ", "fumigation", "sterility", "autoclave", "oxidation", "chlorination", "fresh",
                    "immaculate", "sparkling", "pristine", "dust", "stains", "shiny", "linens", "bathroom",
                    "trash", "bins", "odor", "swept", "mopped", "spills", "windows", "walls", "polished",
                    "towels", "dustbin", "disposals", "cobwebs", "grime", "smell", "smudges", "disinfected",
                    "leakage", "mold", "surfaces", "sanitization", "dirt", "clear", "cluttered",
                    "uncluttered", "dirty", "scrubbed", "laundry", "foul", "odorless", "cleanliness",
                    "hygiene", "unhygienic", "rats", "room"],

    "facility": ["hospital", "office", "facility", "service", "hospital", "management", "accomodation", "people",
                 "experience", "convienent", "comfort", "air conditioning", "parking", "cafeteria",
                 "wheelchair", "elevators", "furniture", "lighting", "wi-fi", "seating", "signage",
                 "maintenance", "equipment", "spacious", "toiletries", "ventilation", "private", "shared",
                 "cabin", "garden", "area", "pharmacy", "lounge", "lab", "ward", "table",
                 "washbasins", "temperature", "lights", "storage", "disposal", "linen", "unit", "stay",
                 "care", "place", "healthcare", "hospitality", "comfortable", "atmosphere",
                 "overall", "organisation", "worth", "process", "clinic", "advanced", "department",
                 "quality", "speciality", "premises", "ambience", "environment", "privacy", "digital",
                 "everything", "infrastructure", "infra", "canteen", "corporate", "system", "centre",
                 "administration", "center", "standards", "clinics", "washroom", "toilet", "bathroom", "restroom",
                 "icu", "room access"]
}
ASPECTS = ["doctor", "staff", "cost", "cleanliness", "waitingtime", "facility"]
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]
CONFIDENCE_THRESHOLD = 0.3

# Define aspect-specific model paths
aspect_model_paths = {
    "doctor": "fine_tuned_roberta_doctor",
    "staff": "fine_tuned_roberta_staff",
    "cost": "fine_tuned_roberta_cost",
    "waitingtime": "fine_tuned_roberta_waitingtime",
    "cleanliness": "fine_tuned_roberta_cleanliness",
    "facility": "fine_tuned_roberta_facility"
}

# Load aspect-specific models and tokenizers
aspect_models = {}
for aspect, path in aspect_model_paths.items():
    model = AutoModelForSequenceClassification.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    aspect_models[aspect] = {"model": model, "tokenizer": tokenizer}

# ----------------- SCHEMAS -----------------

class ReviewRequest(BaseModel):
    reviews: List[str]

class ScrapeRequest(BaseModel):
    hospital_name: str

# ----------------- UTILS -----------------

def predict_sentiment(text: str, aspect: str) -> str:
    aspect_key = aspect.lower().replace(" ", "")
    model_data = aspect_models.get(aspect_key)
    if not model_data:
        return "Neutral"

    tokenizer = model_data["tokenizer"]
    model = model_data["model"]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    sentiment_idx = torch.argmax(probs).item()
    confidence = probs[sentiment_idx].item()
    return "Neutral" if confidence < CONFIDENCE_THRESHOLD else SENTIMENT_LABELS[sentiment_idx]

def analyze_review(review: str) -> Dict[str, str]:
    review_lower = review.lower()
    aspect_sentiments = {}

    for aspect in ASPECTS:
        keywords = aspect_keywords.get(aspect.lower(), [])
        if any(kw in review_lower for kw in keywords):
            sentiment = predict_sentiment(review, aspect)
            aspect_sentiments[aspect] = sentiment
        else:
            aspect_sentiments[aspect] = ""

    return aspect_sentiments

# ----------------- API ROUTES -----------------

@app.post("/predict")
async def predict_sentiment_api(request: ReviewRequest):
    return {"results": [analyze_review(r) for r in request.reviews]}

@app.post("/scrape")
async def scrape_reviews_api(request: ScrapeRequest):
    encoded_name = urllib.parse.quote(request.hospital_name)
    url = f"https://api.app.outscraper.com/maps/reviews-v3?query={encoded_name}&reviewsLimit=10&async=false"
    headers = {
        "X-API-KEY": "YOUR_API_KEY"  # Replace with your actual Outscraper API key
    }

    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        if "data" not in data or not data["data"]:
            raise HTTPException(status_code=404, detail="No data found.")
        reviews = [item.get("review_text", "") for item in data["data"][0]["reviews_data"]]
        return {"hospital": request.hospital_name, "reviews": reviews}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/send_email/")
async def send_email_with_chart_api(
    to_email: EmailStr = Form(...),
    chart_file: UploadFile = Form(...)
):
    try:
        chart_bytes = await chart_file.read()

        msg = EmailMessage()
        msg["Subject"] = "Sentiment Analysis Chart"
        msg["From"] = "akhilaa1326@gmail.com"  # Replace with your sender email
        msg["To"] = to_email
        msg.set_content("Attached is the sentiment analysis chart from Pratikriya HCQ.")
        msg.add_attachment(chart_bytes, maintype="image", subtype="png", filename="sentiment_chart.png")

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login("your_email@gmail.com", "your_app_password")  # Use your real Gmail credentials or app password
            smtp.send_message(msg)

        return {"status": "success", "message": "Email sent successfully"}

    except Exception as e:
        return {"status": "error", "message": str(e)}
