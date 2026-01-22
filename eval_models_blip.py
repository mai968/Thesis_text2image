# Evaluate for model sinh caption

import os
import pandas as pd
import torch
from evaluate import load
from PIL import Image
import clip
from tqdm import tqdm

# ==========================================================
# STEP 1: Load dataset
# ==========================================================
CSV_PATH = "caption_results_3models.csv"  # file bạn vừa sinh ở bước trước
df = pd.read_csv(CSV_PATH)

print(f"📄 Loaded {len(df)} samples from {CSV_PATH}")
print("Columns:", list(df.columns), "\n")

# ground-truth caption (do bạn viết tay)
refs = [[r] for r in df["caption_detail"].astype(str).tolist()]

# ==========================================================
# STEP 2: Initialize metrics
# ==========================================================
bleu = load("bleu")
meteor = load("meteor")
bertscore = load("bertscore")

# --- load CLIP for image-text cosine similarity
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# ==========================================================
# STEP 3: Evaluate helper function
# ==========================================================
def evaluate_model(preds, name):
    print(f"Evaluating {name} ...")
    results = {}

    results["BLEU"] = bleu.compute(predictions=preds, references=refs)["bleu"]
    results["METEOR"] = meteor.compute(predictions=preds, references=refs)["meteor"]
    bert = bertscore.compute(predictions=preds, references=refs, lang="en")
    results["BERTScore_P"] = sum(bert["precision"]) / len(bert["precision"])
    results["BERTScore_R"] = sum(bert["recall"]) / len(bert["recall"])
    results["BERTScore_F1"] = sum(bert["f1"]) / len(bert["f1"])

    # --- CLIPScore ---
    scores = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"CLIP-{name}"):
        img_path = os.path.join("dataset_animals", row["animal"], row["image"])
        if not os.path.exists(img_path):
            alt_path = os.path.join("dataset_animals/subset/train_subset_50/images", row["image"])
            if os.path.exists(alt_path):
                img_path = alt_path
            else:
                scores.append(0)
                continue

        image = clip_preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        text = clip.tokenize([row[f"caption_{name.lower()}"]]).to(device)
        with torch.no_grad():
            img_feat = clip_model.encode_image(image)
            txt_feat = clip_model.encode_text(text)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
            score = (img_feat @ txt_feat.T).item()
        scores.append(score)

    results["CLIPScore"] = sum(scores) / len(scores)
    print(f"{name} done!\n")
    return results


# ==========================================================
# STEP 4: Run evaluation for all models
# ==========================================================
models = ["BLIP", "GIT", "VIT"]
metrics = []

for model in models:
    preds = df[f"caption_{model.lower()}"].astype(str).tolist()
    res = evaluate_model(preds, model)
    res["Model"] = model
    metrics.append(res)

# ==========================================================
# STEP 5: Save results
# ==========================================================
results_df = pd.DataFrame(metrics)[
    ["Model", "BLEU", "METEOR", "BERTScore_P", "BERTScore_R", "BERTScore_F1", "CLIPScore"]
]
results_df.to_csv("evaluation_summary.csv", index=False, encoding="utf-8-sig")

print("🎯 Evaluation completed!")
print(results_df)
print("\n📊 Results saved to 'evaluation_blip_summary.csv'")
 