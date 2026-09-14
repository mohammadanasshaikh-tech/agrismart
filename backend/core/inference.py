"""
Core inference logic used by the API. Same model as model/predict.py,
but works on in-memory image bytes (from an uploaded file) instead of a path.
"""

import json
import os
import io

import torch
import timm
from torchvision import transforms
from PIL import Image

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "model")
MODEL_DIR = os.path.abspath(MODEL_DIR)
WEIGHTS_PATH = os.path.join(MODEL_DIR, "best_model.pt")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PRECAUTIONS = {
    "healthy": "No action needed. Continue regular monitoring.",
    "Apple_scab": "Remove and destroy fallen leaves. Apply fungicide at bud break if recurring.",
    "Cedar_apple_rust": "Remove nearby cedar/juniper hosts if possible. Apply fungicide in early spring.",
    "Cercospora_leaf_spot Gray_leaf_spot": "Rotate crops, avoid overhead irrigation, apply fungicide if severe.",
    "Common_rust_": "Plant resistant hybrids next season. Fungicide rarely needed unless severe.",
    "Northern_Leaf_Blight": "Rotate crops, till residue, use resistant hybrids.",
    "Black_rot": "Prune out infected canes/leaves. Apply fungicide during wet spring weather.",
    "Esca_(Black_Measles)": "Remove and destroy infected wood. No effective chemical cure -- prevention is key.",
    "Bacterial_spot": "Avoid overhead watering. Copper-based bactericide can slow spread.",
    "Early_blight": "Remove affected lower leaves, avoid overhead watering, rotate crops.",
    "Late_blight": "Remove and destroy infected plants immediately -- spreads fast. Apply fungicide preventively in humid weather.",
    "Powdery_mildew": "Improve air circulation, avoid overhead watering, apply sulfur-based fungicide.",
    "Leaf_Mold": "Improve greenhouse ventilation, reduce humidity, avoid wetting leaves.",
    "Septoria_leaf_spot": "Remove infected lower leaves, mulch to prevent soil splash, rotate crops.",
    "Spider_mites Two-spotted_spider_mite": "Spray with water to dislodge mites, use insecticidal soap if severe.",
    "Tomato_Yellow_Leaf_Curl_Virus": "Remove infected plants, control whitefly vectors, use resistant varieties.",
    "Tomato_mosaic_virus": "Remove infected plants, sanitize tools, avoid tobacco contact before handling plants.",
}


def get_precaution(class_name):
    disease_part = class_name.split("___", 1)[-1] if "___" in class_name else class_name
    if "healthy" in disease_part.lower():
        return PRECAUTIONS["healthy"]
    return PRECAUTIONS.get(disease_part, "Consult a local agricultural extension office for guidance.")


_model_cache = {}


def load_model():
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["class_names"]

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"No trained weights at {WEIGHTS_PATH}. Run model/train.py first.")

    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)

    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(class_names))
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    _model_cache["model"] = model
    _model_cache["class_names"] = class_names
    return model, class_names


_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_from_bytes(image_bytes, top_k=3):
    """Same as predict.py's predict(), but takes raw image bytes (from an upload)."""
    model, class_names = load_model()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = _transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, k=min(top_k, len(class_names)))
    top_probs = top_probs.cpu().numpy()
    top_idxs = top_idxs.cpu().numpy()

    predicted_class = class_names[top_idxs[0]]
    confidence = float(top_probs[0])

    return {
        "class_label": predicted_class,
        "display_name": predicted_class.replace("___", " - ").replace("_", " "),
        "confidence": confidence,
        "is_healthy": "healthy" in predicted_class.lower(),
        "precaution": get_precaution(predicted_class),
        "top_k": [
            {
                "class_label": class_names[i],
                "display_name": class_names[i].replace("___", " - ").replace("_", " "),
                "confidence": float(p),
            }
            for i, p in zip(top_idxs, top_probs)
        ],
    }


def get_all_classes():
    _, class_names = load_model()
    return class_names
