"""
Bonus Module D: Sustainability Score.

Rule-based, fully reproducible formula (per brief section 3.2 requirement:
"Publish the exact formula/rules so it is reproducible").

Formula (0-100 scale):
    score = 0.4 * water_efficiency_score
          + 0.3 * crop_health_score
          + 0.3 * resource_use_score

Where each sub-score is 0-100:

- water_efficiency_score: based on soil_moisture_pct vs an ideal band (40-60%).
  Inside the band = 100. Outside, score decays linearly, floor at 0.

- crop_health_score: 100 if model predicted "healthy", else scaled down by
  model confidence in the diseased prediction (higher confidence in disease
  detection = more certain the plant needs intervention = lower health score).

- resource_use_score: based on self-reported fertilizer/pesticide use level
  (low/medium/high), mapped to 100/60/20. This is a simple proxy -- a real
  deployment would use actual application records.
"""


def water_efficiency_score(soil_moisture_pct: float) -> float:
    IDEAL_LOW, IDEAL_HIGH = 40.0, 60.0
    if IDEAL_LOW <= soil_moisture_pct <= IDEAL_HIGH:
        return 100.0
    if soil_moisture_pct < IDEAL_LOW:
        deficit = IDEAL_LOW - soil_moisture_pct
        return max(0.0, 100.0 - deficit * 2.5)
    excess = soil_moisture_pct - IDEAL_HIGH
    return max(0.0, 100.0 - excess * 2.5)


def crop_health_score(is_healthy: bool, disease_confidence: float) -> float:
    if is_healthy:
        return 100.0
    # higher confidence in a disease diagnosis -> more certain intervention is needed
    return max(0.0, 100.0 - disease_confidence * 100.0)


RESOURCE_USE_MAP = {"low": 100.0, "medium": 60.0, "high": 20.0}


def resource_use_score(fertilizer_pesticide_level: str) -> float:
    return RESOURCE_USE_MAP.get(fertilizer_pesticide_level.lower(), 60.0)


def compute_sustainability_score(soil_moisture_pct, is_healthy, disease_confidence,
                                  fertilizer_pesticide_level="medium"):
    w = water_efficiency_score(soil_moisture_pct)
    c = crop_health_score(is_healthy, disease_confidence)
    r = resource_use_score(fertilizer_pesticide_level)

    final_score = round(0.4 * w + 0.3 * c + 0.3 * r, 1)

    suggestions = []
    if w < 70:
        if soil_moisture_pct < 40:
            suggestions.append("Soil moisture is low -- consider increasing irrigation frequency.")
        else:
            suggestions.append("Soil moisture is high -- reduce irrigation to avoid waterlogging and runoff.")
    if c < 70:
        suggestions.append("Disease detected with meaningful confidence -- address it to protect yield and reduce input waste.")
    if r < 70:
        suggestions.append("Consider reducing fertilizer/pesticide use where possible, or switch to targeted application.")
    if not suggestions:
        suggestions.append("Current practices look sustainable -- keep monitoring.")

    return {
        "final_score": final_score,
        "breakdown": {
            "water_efficiency_score": round(w, 1),
            "crop_health_score": round(c, 1),
            "resource_use_score": round(r, 1),
        },
        "formula": "0.4*water_efficiency + 0.3*crop_health + 0.3*resource_use",
        "suggestions": suggestions,
    }
