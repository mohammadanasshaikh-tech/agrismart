"""
Step 6: Required predict interface (Section 4.1 of the brief).

CLI usage:
    python model/predict.py --image path/to/leaf.jpg

Also exposes predict(image_path) -> class_label as a plain Python function,
importable from the Streamlit app.
"""

import argparse
import json
import os

import torch
import timm
from torchvision import transforms
from PIL import Image

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(MODEL_DIR, "best_model.pt")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# basic precaution text per disease -- shown to the farmer alongside the prediction
# (Section 5 of the brief: "surface a clear result plus basic precautionary guidance")
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
    # class_name looks like "Tomato___Early_blight" -- take part after "___"
    disease_part = class_name.split("___", 1)[-1] if "___" in class_name else class_name
    if "healthy" in disease_part.lower():
        return PRECAUTIONS["healthy"]
    return PRECAUTIONS.get(disease_part, "Consult a local agricultural extension office for guidance.")


_model_cache = {}


def load_model():
    """Loads and caches the model + class list. Call once, reuse across predictions."""
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["class_names"]

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"No trained weights found at {WEIGHTS_PATH}. Run model/train.py first."
        )
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


def predict(image_path, top_k=3):
    """
    Required interface per Section 4.1: predict(image_path) -> class_label
    Also returns confidence and top-k alternatives + precaution text for the UI.
    """
    model, class_names = load_model()

    img = Image.open(image_path).convert("RGB")
    x = _transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, k=min(top_k, len(class_names)))
    top_probs = top_probs.cpu().numpy()
    top_idxs = top_idxs.cpu().numpy()

    predicted_class = class_names[top_idxs[0]]
    confidence = float(top_probs[0])

    result = {
        "class_label": predicted_class,
        "confidence": confidence,
        "precaution": get_precaution(predicted_class),
        "top_k": [
            {"class_label": class_names[i], "confidence": float(p)}
            for i, p in zip(top_idxs, top_probs)
        ],
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict crop disease from a leaf image.")
    parser.add_argument("--image", required=True, help="Path to a leaf/crop image")
    args = parser.parse_args()

    result = predict(args.image)

    print(f"\nPredicted class : {result['class_label']}")
    print(f"Confidence      : {result['confidence']*100:.1f}%")
    print(f"Precaution      : {result['precaution']}")
    print(f"\nTop-{len(result['top_k'])} predictions:")
    for r in result["top_k"]:
        print(f"  {r['class_label']:60s} {r['confidence']*100:5.1f}%")