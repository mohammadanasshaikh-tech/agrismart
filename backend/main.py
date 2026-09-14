"""
AgriSmart AI - FastAPI Backend v2

Routes:
    GET  /api/health                  - service status check
    GET  /api/classes                 - list all supported disease classes
    POST /api/predict                 - core task: upload leaf image (+ optional lat/lon) -> prediction
    GET  /api/predictions/{id}        - full detail of one stored prediction
    GET  /api/history                 - recent predictions (list view)
    GET  /api/map-points               - predictions that have location data, for map display
    GET  /api/dashboard-stats         - aggregate stats for the dashboard
    POST /api/sustainability-score    - bonus D: compute sustainability score
    GET  /api/weather-advisory        - bonus C: weather-based action advisory
    GET  /uploads/{filename}          - serves stored leaf images

Run with (from project root):
    uvicorn backend.main:app --reload --port 8000
"""

import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import requests

from backend.core.inference import predict_from_bytes, get_all_classes
from backend.core.sustainability import compute_sustainability_score
from backend.core import database

database.init_db()

app = FastAPI(title="AgriSmart AI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve stored leaf images at http://localhost:8000/uploads/<filename>
app.mount("/uploads", StaticFiles(directory=database.UPLOADS_DIR), name="uploads")


class SustainabilityRequest(BaseModel):
    soil_moisture_pct: float
    is_healthy: bool
    disease_confidence: float = 0.0
    fertilizer_pesticide_level: str = "medium"


def _image_url(image_filename):
    if not image_filename:
        return None
    return f"/uploads/{image_filename}"


@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "AgriSmart AI API"}


@app.get("/api/classes")
def list_classes():
    try:
        classes = get_all_classes()
        return {"count": len(classes), "classes": classes}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/api/predict")
async def predict_disease(
    file: UploadFile = File(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
):
    """Core task: accept a leaf image (+ optional GPS location), return + store the prediction."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    try:
        result = predict_from_bytes(image_bytes)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    image_filename = database.save_uploaded_image(image_bytes, file.filename or "upload.jpg")

    new_id = database.save_prediction(
        class_label=result["class_label"],
        display_name=result["display_name"],
        confidence=result["confidence"],
        is_healthy=result["is_healthy"],
        precaution=result["precaution"],
        image_filename=image_filename,
        latitude=latitude,
        longitude=longitude,
        top_k_json=json.dumps(result["top_k"]),
    )

    result["id"] = new_id
    result["image_url"] = _image_url(image_filename)
    result["latitude"] = latitude
    result["longitude"] = longitude
    return result


@app.get("/api/predictions/{pred_id}")
def prediction_detail(pred_id: int):
    """Full stored detail of a single prediction, for the frontend's click-to-expand view."""
    row = database.get_prediction_by_id(pred_id)
    if not row:
        raise HTTPException(status_code=404, detail="Prediction not found.")

    row["image_url"] = _image_url(row.get("image_filename"))
    row["is_healthy"] = bool(row["is_healthy"])
    row["top_k"] = json.loads(row["top_k_json"]) if row.get("top_k_json") else []
    return row


@app.get("/api/history")
def prediction_history(limit: int = Query(50, ge=1, le=500)):
    rows = database.get_prediction_history(limit=limit)
    for r in rows:
        r["image_url"] = _image_url(r.get("image_filename"))
        r["is_healthy"] = bool(r["is_healthy"])
    return {"predictions": rows}


@app.get("/api/map-points")
def map_points():
    """Predictions that have GPS location attached, for the dashboard map."""
    rows = database.get_predictions_with_location()
    for r in rows:
        r["image_url"] = _image_url(r.get("image_filename"))
        r["is_healthy"] = bool(r["is_healthy"])
    return {"points": rows}


@app.get("/api/dashboard-stats")
def dashboard_stats():
    return database.get_dashboard_stats()


@app.post("/api/sustainability-score")
def sustainability_score(req: SustainabilityRequest):
    result = compute_sustainability_score(
        soil_moisture_pct=req.soil_moisture_pct,
        is_healthy=req.is_healthy,
        disease_confidence=req.disease_confidence,
        fertilizer_pesticide_level=req.fertilizer_pesticide_level,
    )
    database.save_sustainability_score(result["final_score"], result["breakdown"])
    return result


@app.get("/api/weather-advisory")
def weather_advisory(
    lat: float = Query(..., description="Latitude of the farm"),
    lon: float = Query(..., description="Longitude of the farm"),
    disease_detected: bool = Query(False, description="Whether the core model flagged a disease"),
):
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "precipitation_probability_max,temperature_2m_max,temperature_2m_min",
                "forecast_days": 2,
                "timezone": "auto",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Weather source unavailable: {e}")

    rain_prob_today = data["daily"]["precipitation_probability_max"][0]
    rain_prob_tomorrow = data["daily"]["precipitation_probability_max"][1]
    temp_max = data["daily"]["temperature_2m_max"][0]

    actions = []
    if rain_prob_today >= 60 or rain_prob_tomorrow >= 60:
        actions.append("Delay irrigation -- rain likely in the next 24-48h.")
    else:
        actions.append("No rain expected soon -- irrigation as normally scheduled is fine.")

    if disease_detected and (rain_prob_today >= 50 or rain_prob_tomorrow >= 50):
        actions.append("Raised disease risk -- wet conditions plus an active detection call for closer monitoring.")

    if temp_max >= 35:
        actions.append("High temperature expected -- watch for heat stress, consider shade/extra watering timing.")

    return {
        "source": "Open-Meteo (api.open-meteo.com)",
        "rain_probability_today_pct": rain_prob_today,
        "rain_probability_tomorrow_pct": rain_prob_tomorrow,
        "temperature_max_c": temp_max,
        "actions": actions,
    }
