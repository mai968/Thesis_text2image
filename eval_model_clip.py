#!/usr/bin/env python3
"""
Evaluate CLIP-based models on Text-to-Image retrieval
CPU/GPU compatible version for thesis experiments
"""
import datetime
import pandas as pd
import os
import numpy as np
import time
import argparse
from tqdm import tqdm
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch
import clip
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModel
from tabulate import tabulate

# ---------- Utility ----------
def l2_normalize(x):
    x = np.array(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def recall_at_k(sim, k):
    ranks = np.argsort(-sim, axis=1)
    hits = sum(1 for i in range(sim.shape[0]) if i in ranks[i, :k])
    return hits / sim.shape[0]

def mean_rr(sim):
    ranks = np.argsort(-sim, axis=1)
    rrs = []
    for i in range(sim.shape[0]):
        pos = np.where(ranks[i] == i)[0]
        rrs.append(1/(pos[0]+1) if pos.size>0 else 0)
    return np.mean(rrs)

# ---------- Encoding ----------
def load_text_encoder(name, model_name, device="cpu"):
    """
    Trả về hàm encode_texts(texts) phù hợp với từng loại model
    """
    if "siglip" in model_name:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        def encode_texts(texts):
            inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                embs = model.get_text_features(**inputs)
            return embs.cpu().numpy()
        return encode_texts

    elif "align" in model_name:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        def encode_texts(texts):
            inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                embs = model.get_text_features(**inputs)
            return embs.cpu().numpy()
        return encode_texts

    else:
        model = SentenceTransformer(model_name, device=device)
        def encode_texts(texts):
            return model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return encode_texts
    

def encode_images_clip(model, preprocess, paths, device, batch=32):
    model.eval()
    all_feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), batch), desc="Encoding images"):
            imgs = []
            for p in paths[i:i+batch]:
                try:
                    img = Image.open(p).convert("RGB")
                    imgs.append(preprocess(img).unsqueeze(0))
                except:
                    imgs.append(torch.zeros((1,3,224,224)))
            imgs = torch.cat(imgs).to(device)
            f = model.encode_image(imgs).cpu().numpy()
            all_feats.append(f)
    return np.vstack(all_feats)


def encode_texts_clip(model, texts, device, batch=64):
    model.eval()
    all_feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch), desc="Encoding texts (CLIP)"):
            tokens = clip.tokenize(texts[i:i+batch]).to(device)
            f = model.encode_text(tokens).cpu().numpy()
            all_feats.append(f)
    return np.vstack(all_feats)

# ---------- Evaluation ----------
def evaluate_pair(text_embs, image_embs, time_s=None):
    text_embs, image_embs = l2_normalize(text_embs), l2_normalize(image_embs)
    sim = cosine_similarity(text_embs, image_embs)
    return {
        "CosSim": float(np.mean(np.diag(sim))),
        "Recall@1": recall_at_k(sim, 1),
        "Recall@5": recall_at_k(sim, 5),
        "MRR": mean_rr(sim),
        "Time (s/query)": time_s or 0.0
    }

# ---------- Main ----------
def main(args):
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} samples.")
    image_paths = [os.path.join(args.image_dir, p) for p in df["image"]]
    captions = df["caption_detail"].fillna("").tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    models_to_eval = [
        ("CLIP_OpenAI", "ViT-B/32"),
        ("SigLIP", "google/siglip-so400m-patch14-384"),
        ("ALIGN_like", "kakaobrain/align-base"),
    ]

    results = []

    for label, model_name in models_to_eval:
        print(f"\n=== Evaluating model: {label} ===")

        # -------------------------------------------------
        # =========================================================
        # 🔹 1. Load & encode IMAGES per model (STANDARDIZED)
        # =========================================================

        if label == "CLIP_OpenAI":
            # ✅ ĐÚNG MODEL CLIP HỆ THỐNG
            image_model, preprocess = clip.load(args.clip_variant, device=device)
            image_model.eval()

            t0 = time.time()
            image_embs = encode_images_clip(image_model, preprocess, image_paths, device)
            t_img = (time.time() - t0) / len(df)


        elif label == "SigLIP":
            from transformers import AutoProcessor, AutoModel
            processor = AutoProcessor.from_pretrained(model_name)
            image_model = AutoModel.from_pretrained(model_name).to(device)
            image_model.eval()

            t0 = time.time()
            all_feats = []
            with torch.no_grad():
                for i in tqdm(range(0, len(image_paths), 16), desc="Encoding images (SigLIP)"):
                    batch_imgs = []
                    for p in image_paths[i:i+16]:
                        try:
                            img = Image.open(p).convert("RGB")
                        except:
                            img = Image.new("RGB", (384, 384))
                        batch_imgs.append(img)

                    inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
                    feats = image_model.get_image_features(**inputs)
                    all_feats.append(feats.cpu().numpy())

            image_embs = np.vstack(all_feats)
            t_img = (time.time() - t0) / len(df)


        elif label == "ALIGN_like":
            from transformers import AutoProcessor, AutoModel
            processor = AutoProcessor.from_pretrained(model_name)
            image_model = AutoModel.from_pretrained(model_name).to(device)
            image_model.eval()

            t0 = time.time()
            all_feats = []
            with torch.no_grad():
                for i in tqdm(range(0, len(image_paths), 16), desc="Encoding images (ALIGN)"):
                    batch_imgs = []
                    for p in image_paths[i:i+16]:
                        try:
                            img = Image.open(p).convert("RGB")
                        except:
                            img = Image.new("RGB", (224, 224))
                        batch_imgs.append(img)

                    inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
                    feats = image_model.get_image_features(**inputs)
                    all_feats.append(feats.cpu().numpy())

            image_embs = np.vstack(all_feats)
            t_img = (time.time() - t0) / len(df)


        else:
            raise ValueError(f"Unknown model label: {label}")


        # =========================================================
        # 🔹 2. Encode TEXTS per model (STANDARDIZED)
        # =========================================================

        if label == "CLIP_OpenAI":
            t0 = time.time()
            text_embs = encode_texts_clip(image_model, captions, device)
            t_text = (time.time() - t0) / len(df)

        else:
            encode_fn = load_text_encoder(label, model_name, device)
            t0 = time.time()
            text_embs = encode_fn(captions)
            t_text = (time.time() - t0) / len(df)


        # -------------------------------------------------
        # 🔹 3. Evaluate similarity
        # -------------------------------------------------
        metrics = evaluate_pair(text_embs, image_embs, t_img + t_text)
        results.append({
            "Model": label,
            **{k: round(v, 4) for k, v in metrics.items()}
        })
        print("  ->", results[-1])

    # -------------------------------------------------
    # 🔹 Final summary table
    # -------------------------------------------------
    print("\n=== 📊 FINAL TABLE ===")
    df_res = pd.DataFrame(results)
    print(tabulate(df_res, headers="keys", tablefmt="grid", floatfmt=".4f"))

    # -------------------------------------------------
    # 🔹 SAVE FINAL TABLE TO CSV
    # -------------------------------------------------

    os.makedirs("results", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_csv_path = os.path.join(
        "results",
        f"final_evaluation_{timestamp}.csv"
    )

    df_res.to_csv(final_csv_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ Final evaluation table saved to: {final_csv_path}")



# =====================================================
# 🏁 CLI ENTRY POINT
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="caption_results_3models.csv", help="CSV with columns: id,image,caption_detail,caption_blip")
    parser.add_argument("--image_dir", type=str, default="dataset_animals/subset/train_subset_50/images", help="directory containing images (paths in CSV relative to this dir)")
    parser.add_argument("--clip_variant", type=str, default="ViT-B/32", help="CLIP variant (openai name)")
    args = parser.parse_args()
    main(args)
