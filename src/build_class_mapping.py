"""
Step 4 v2: Smarter class matching.
Fixes the v1 bug where "healthy" classes got matched to diseased classes,
and where classes from different crops got matched due to keyword overlap
on words like "leaf" or "spot".

Rule: two classes only match if
  (a) they belong to the SAME crop, AND
  (b) they are BOTH "healthy" or BOTH "diseased", AND
  (c) if diseased, they share at least one disease-specific keyword
"""

import os
import json
import re

PLANTVILLAGE_ROOT = r"D:\yash project\data\PlantVillage-Dataset\raw\color"
PLANTDOC_ROOT_TRAIN = r"D:\yash project\data\PlantDoc-Dataset\train"
PLANTDOC_ROOT_TEST = r"D:\yash project\data\PlantDoc-Dataset\test"
OUTPUT_JSON = r"D:\yash project\src\class_mapping.json"

# canonical crop name -> list of substrings that identify it in either dataset's naming
CROP_ALIASES = {
    "apple": ["apple"],
    "blueberry": ["blueberry"],
    "cherry": ["cherry"],
    "corn": ["corn", "maize"],
    "grape": ["grape"],
    "orange": ["orange", "citrus"],
    "peach": ["peach"],
    "pepper": ["pepper", "bell_pepper", "bell pepper"],
    "potato": ["potato"],
    "raspberry": ["raspberry"],
    "soybean": ["soybean", "soyabean"],
    "squash": ["squash"],
    "strawberry": ["strawberry"],
    "tomato": ["tomato"],
}

STOPWORDS = {"leaf", "leaves", "the", "a", "including", "sour", "spot", "spots",
             "disease", "including_sour"}


def normalize_words(name):
    name = name.lower().replace("___", " ").replace("__", " ").replace("_", " ").replace(",", " ")
    name = re.sub(r'[^a-z0-9\s]', ' ', name)
    return [w for w in name.split() if w and w not in STOPWORDS]


def detect_crop(name_lower):
    for crop, aliases in CROP_ALIASES.items():
        for alias in aliases:
            if alias.replace("_", " ") in name_lower or alias.replace(" ", "_") in name_lower:
                return crop
    return None


def is_healthy(name_lower):
    return "healthy" in name_lower


def safe_listdir(path):
    if not os.path.isdir(path):
        print(f"WARNING: path not found: {path}")
        return []
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])


def main():
    pv_classes = safe_listdir(PLANTVILLAGE_ROOT)
    pd_train = safe_listdir(PLANTDOC_ROOT_TRAIN)
    pd_test = safe_listdir(PLANTDOC_ROOT_TEST)
    pd_classes = sorted(set(pd_train) | set(pd_test))

    print(f"PlantVillage classes: {len(pv_classes)}")
    print(f"PlantDoc classes: {len(pd_classes)}")

    if not pv_classes or not pd_classes:
        print("STOPPING -- fix the paths at top of script first.")
        return

    # pre-compute crop + healthy state + keyword set for every pd class
        # pre-compute crop + healthy state + keyword set for every pd class
    pd_info = []
    for pd_c in pd_classes:
        low = pd_c.lower().replace("_", " ")
        crop = detect_crop(low)
        words = set(normalize_words(pd_c)) - set(CROP_ALIASES.get(crop, []))
        # PlantDoc convention: a healthy leaf is just "<crop>_leaf" with NO extra
        # disease words -- it usually doesn't literally say "healthy".
        healthy = is_healthy(low) or len(words) == 0
        pd_info.append({
            "name": pd_c,
            "crop": crop,
            "healthy": healthy,
            "words": words,
        })

    mapping = {}
    unmatched_pv = []

    for pv_c in pv_classes:
        low = pv_c.lower().replace("_", " ").replace(",", " ")
        pv_crop = detect_crop(low)
        pv_healthy = is_healthy(low)
        pv_words = set(normalize_words(pv_c)) - set(CROP_ALIASES.get(pv_crop, []))

        candidates = [p for p in pd_info if p["crop"] == pv_crop and p["healthy"] == pv_healthy]

        if not candidates:
            unmatched_pv.append((pv_c, "no same-crop/same-health-state class in PlantDoc"))
            continue

        if pv_healthy:
            # healthy: just take the (usually only) healthy candidate for that crop
            best = candidates[0]
            mapping[pv_c] = best["name"]
        else:
            # diseased: require actual disease keyword overlap, not just crop match
            best = None
            best_score = 0
            for cand in candidates:
                cand_words = cand["words"] - set(CROP_ALIASES.get(pv_crop, []))
                overlap = len(pv_words & cand_words)
                if overlap > best_score:
                    best_score = overlap
                    best = cand
            if best and best_score >= 1:
                mapping[pv_c] = best["name"]
            else:
                unmatched_pv.append((pv_c, f"same crop found but no disease-keyword overlap (candidates: {[c['name'] for c in candidates]})"))

    print(f"\n{'='*70}")
    print(f"CONFIRMED MATCHES ({len(mapping)})")
    print('='*70)
    for pv_c, pd_c in mapping.items():
        print(f"  {pv_c}  <-->  {pd_c}")

    print(f"\n{'='*70}")
    print(f"EXCLUDED / UNMATCHED ({len(unmatched_pv)})")
    print('='*70)
    for pv_c, reason in unmatched_pv:
        print(f"  {pv_c}  --  {reason}")

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"\nSaved {len(mapping)} verified class mappings to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()