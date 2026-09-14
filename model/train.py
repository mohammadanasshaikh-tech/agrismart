"""
Step 5: Train the crop-disease classifier.

- TRAIN/VAL: PlantVillage (lab images), split 85/15
- TEST (held-out, never trained on): PlantDoc (field images)
- Model: EfficientNet-B0, ImageNet-pretrained, fine-tuned
- Reports: macro-F1, confusion matrix, per-class precision/recall on BOTH
  the PlantVillage val split and the PlantDoc field test set.

Run: python model/train.py
"""

import os
import json
import time
import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

torch.backends.cudnn.benchmark = True  # speeds up conv training once input size is fixed

# ---------------- CONFIG ----------------
PLANTVILLAGE_ROOT = r"D:\yash project\data\PlantVillage-Dataset\raw\color"
PLANTDOC_ROOT_TRAIN = r"D:\yash project\data\PlantDoc-Dataset\train"
PLANTDOC_ROOT_TEST = r"D:\yash project\data\PlantDoc-Dataset\test"
CLASS_MAPPING_JSON = r"D:\yash project\src\class_mapping.json"

MODEL_OUT_DIR = r"D:\yash project\model"
REPORT_OUT_DIR = r"D:\yash project\report"

IMG_SIZE = 224
BATCH_SIZE = 16          # drop to 8 if you hit CUDA out-of-memory
NUM_EPOCHS = 15
LR = 3e-4
VAL_SPLIT = 0.15
NUM_WORKERS = 2          # Windows: keep this low (0-2) to avoid multiprocessing issues
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(MODEL_OUT_DIR, exist_ok=True)
os.makedirs(REPORT_OUT_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")


# ---------------- DATASET ----------------
class ImageListDataset(Dataset):
    """Simple dataset: list of (image_path, label_idx) pairs + a transform."""
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


def list_images(folder):
    exts = (".jpg", ".jpeg", ".png")
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(exts)]


def build_datasets():
    with open(CLASS_MAPPING_JSON) as f:
        mapping = json.load(f)  # {pv_class_name: pd_class_name}

    class_names = sorted(mapping.keys())
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    # ---- PlantVillage samples (train/val pool) ----
    pv_samples = []
    for pv_class in class_names:
        folder = os.path.join(PLANTVILLAGE_ROOT, pv_class)
        if not os.path.isdir(folder):
            print(f"  !! missing PlantVillage folder: {folder}")
            continue
        label = class_to_idx[pv_class]
        for img_path in list_images(folder):
            pv_samples.append((img_path, label))

    print(f"Total PlantVillage images across {len(class_names)} classes: {len(pv_samples)}")

    # ---- PlantDoc samples (held-out field test, from BOTH its train+test dirs) ----
    pd_samples = []
    for pv_class, pd_class in mapping.items():
        label = class_to_idx[pv_class]
        for root in (PLANTDOC_ROOT_TRAIN, PLANTDOC_ROOT_TEST):
            folder = os.path.join(root, pd_class)
            if os.path.isdir(folder):
                for img_path in list_images(folder):
                    pd_samples.append((img_path, label))

    print(f"Total PlantDoc (held-out field test) images: {len(pd_samples)}")

    return pv_samples, pd_samples, class_names


# ---------------- TRANSFORMS ----------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------- TRAIN LOOP ----------------
def train_model(model, train_loader, val_loader, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_f1 = 0.0
    best_state = None

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)

        # ---- validation ----
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [val]", leave=False):
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        val_f1 = f1_score(all_labels, all_preds, average="macro")
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{num_epochs} | train_loss={train_loss:.4f} | "
              f"val_macro_f1={val_f1:.4f} | {elapsed:.1f}s")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    print(f"\nBest validation macro-F1: {best_val_f1:.4f}")
    return model


# ---------------- EVALUATION ----------------
def evaluate(model, loader, class_names, split_name):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(all_labels, all_preds, target_names=class_names,
                                    zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*70}\n{split_name} RESULTS\n{'='*70}")
    print(f"Macro-F1: {macro_f1:.4f}\n")
    print(report)

    # save confusion matrix plot
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {split_name} (macro-F1={macro_f1:.3f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    out_path = os.path.join(REPORT_OUT_DIR, f"confusion_matrix_{split_name.replace(' ', '_')}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix -> {out_path}")

    # save text report
    report_path = os.path.join(REPORT_OUT_DIR, f"classification_report_{split_name.replace(' ', '_')}.txt")
    with open(report_path, "w") as f:
        f.write(f"{split_name}\nMacro-F1: {macro_f1:.4f}\n\n{report}")
    print(f"Saved classification report -> {report_path}")

    return macro_f1


# ---------------- MAIN ----------------
def main():
    pv_samples, pd_samples, class_names = build_datasets()
    num_classes = len(class_names)

    if len(pv_samples) == 0:
        print("STOPPING: no PlantVillage images found. Check paths/class_mapping.json.")
        return
    if len(pd_samples) == 0:
        print("WARNING: no PlantDoc images found for held-out test -- check paths.")

    # split PlantVillage into train/val (deterministic shuffle)
    n_val = int(len(pv_samples) * VAL_SPLIT)
    n_train = len(pv_samples) - n_val

    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(pv_samples), generator=g).tolist()
    shuffled = [pv_samples[i] for i in perm]
    train_samples = shuffled[:n_train]
    val_samples = shuffled[n_train:]

    print(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Field test (PlantDoc): {len(pd_samples)}")

    train_ds = ImageListDataset(train_samples, train_transform)
    val_ds = ImageListDataset(val_samples, eval_transform)
    test_ds = ImageListDataset(pd_samples, eval_transform) if pd_samples else None

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True) if test_ds else None

    # ---- model: EfficientNet-B0, pretrained, new head for our num_classes ----
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
    model = model.to(DEVICE)

    model = train_model(model, train_loader, val_loader, NUM_EPOCHS)

    # ---- final evaluation ----
    evaluate(model, val_loader, class_names, "PlantVillage_Validation_Lab")
    if test_loader:
        evaluate(model, test_loader, class_names, "PlantDoc_Held_Out_Field_Test")

    # ---- save weights + class list (predict.py needs both) ----
    weights_path = os.path.join(MODEL_OUT_DIR, "best_model.pt")
    torch.save(model.state_dict(), weights_path)
    with open(os.path.join(MODEL_OUT_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"\nSaved model weights -> {weights_path}")
    print(f"Saved class names -> {os.path.join(MODEL_OUT_DIR, 'class_names.json')}")
    print("\nDONE. Next: Step 6 -- predict.py CLI + Streamlit app.")


if __name__ == "__main__":
    main()