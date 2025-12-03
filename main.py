"""
🐾 **Animal Dataset Manager** - FULL Multimodal Retrieval & Dataset Mgmt Prototype
✨ **BLIP Caption** | **CLIP Multimodal** | **KG Reasoning** | **Doc2Vec Context** | **FAISS** | **Clustering** | **Data Gaps**
✅ **Auto-download 100+ HD Animal Images** | **1-click Run** | **Ready for Thesis Demo**
**Chạy ngay:** 
1. `pip install streamlit torch torchvision torchaudio transformers sentence-transformers faiss-cpu scikit-learn gensim rdflib pillow numpy pandas`
2. `streamlit run app.py`
"""
import os
import re
import io
import shutil
import zipfile
import clip
import requests
import numpy as np
import faiss
import torch
from PIL import Image
from rdflib import Graph, Namespace, RDF, Literal
from sentence_transformers import SentenceTransformer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import glob
from zipfile import Path

from eval_engine import evaluate_system

import time



# -----------------------------
# 📁 Config & Paths (Auto-create)
# -----------------------------
# @st.cache_data
# def get_paths():
#     os.makedirs("data/images", exist_ok=True)
#     os.makedirs("data", exist_ok=True)
#     IMAGE_FOLDER = "data/images"
#     KG_PATH = "data/kg.ttl"
#     INDEX_PATH = "data/faiss.index"
#     EMBEDS_PATH = "data/image_embeds.npy"
#     CAPTIONS_PATH = "data/captions.csv"
#     return IMAGE_FOLDER, KG_PATH, INDEX_PATH, EMBEDS_PATH, CAPTIONS_PATH

@st.cache_data
def get_paths():
    base_data = "data"
    IMAGE_FOLDER = os.path.join(base_data, "images")
    KG_PATH = os.path.join(base_data, "kg.ttl")
    INDEX_PATH = os.path.join(base_data, "faiss.index")
    EMBEDS_PATH = os.path.join(base_data, "image_embeds.npy")
    CAPTIONS_PATH = os.path.join(base_data, "captions.csv")

    # Dọn dẹp và tạo mới
    if os.path.exists(IMAGE_FOLDER):
        shutil.rmtree(IMAGE_FOLDER)
    os.makedirs(IMAGE_FOLDER, exist_ok=True)

    os.makedirs(base_data, exist_ok=True)


    return IMAGE_FOLDER, KG_PATH, INDEX_PATH, EMBEDS_PATH, CAPTIONS_PATH


IMAGE_FOLDER, KG_PATH, INDEX_PATH, EMBEDS_PATH, CAPTIONS_PATH = get_paths()
TOP_K = 9
ALPHA_DOC = 0.3
CONTEXT_RATIO = 0.2
CLUSTER_K = 10

# -----------------------------
# 🤖 1. Load Models (Cached)
# -----------------------------
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"🚀 Using device: {device}")
    
    # CLIP (Multimodal)
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)    

    # BLIP (Captioning)
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model.to(device)
    
    # SentenceTransformer (Text-only fallback/enhance)
    text_model = SentenceTransformer('clip-ViT-B-32')
    
    return clip_model, preprocess, blip_model, blip_proc, text_model, device

    
# -----------------------------
# 📸 2. BLIP Caption Image
# -----------------------------
# @st.cache_data
def caption_image(image, blip_model, blip_proc, device):
    inputs = blip_proc(image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs, max_length=50, num_beams=5)
    caption = blip_proc.decode(out[0], skip_special_tokens=True)
    return caption.lower()

# -----------------------------
# 🔗 3. CLIP Embed (Unified Text/Image)
# -----------------------------
def clip_embed_text(text, clip_model, device):
    text_tok = clip.tokenize([text]).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(text_tok).cpu().numpy()
    return emb[0] / np.linalg.norm(emb[0])

def clip_embed_image(img, preprocess, clip_model, device):
    img_t = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(img_t).cpu().numpy()
    return emb[0] / np.linalg.norm(emb[0])

# -----------------------------
# 🧠 4. Build Dataset: Captions + Embeds + FAISS
# -----------------------------
@st.cache_data
def build_dataset(_models):
    clip_model, preprocess, blip_model, blip_proc, text_model, device = _models

    image_paths = sorted(glob.glob(f"{IMAGE_FOLDER}/*.jpg") + glob.glob(f"{IMAGE_FOLDER}/*.png"))
    if len(image_paths) == 0:
        st.error("❌ No images! Check folder.")
        st.stop()

    st.info(f"🔨 Building dataset for {len(image_paths)} images...")

    captions = []
    embeds = []

    for i, path in enumerate(image_paths):
        img = Image.open(path).convert("RGB")

        # 1️⃣ Generate caption
        cap = caption_image(img, blip_model, blip_proc, device)
        captions.append(cap)

        # 2️⃣ Get CLIP embedding
        emb = clip_embed_image(img, preprocess, clip_model, device)
        if isinstance(emb, np.ndarray):
            emb = emb.squeeze()
        else:
            emb = emb.detach().cpu().numpy().squeeze()
        embeds.append(emb.astype('float32'))

        if (i + 1) % 10 == 0:
            st.progress((i + 1) / len(image_paths))

    embeds = np.array(embeds, dtype='float32')
    st.write(f"✅ Embedding shape: {embeds.shape}, dtype: {embeds.dtype}")

    # Build FAISS index
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    dim = embeds.shape[1]
    st.write(f"🧮 Building FAISS index with dim={dim}")
    index = faiss.IndexFlatIP(dim)
    index.add(embeds)

    # FAISS ID map: chính là chỉ số trong array embeddings
    faiss_id_map = list(range(len(image_paths)))
    np.save("data/faiss_id_map.npy", faiss_id_map)
    faiss.write_index(index, INDEX_PATH)
    st.success("✅ FAISS index + ID mapping saved!")

    # Save CSV + ID tự động
    df = pd.DataFrame({
        'id': [i + 1 for i in range(len(image_paths))],  # ID duy nhất
        'path': image_paths,
        'caption': captions
    })
    df.to_csv(CAPTIONS_PATH, index=False)
    np.save(EMBEDS_PATH, embeds)
    st.success("✅ Dataset Built! Captions + FAISS Ready.")

    return image_paths, df


# -----------------------------
# 🌐 5. Enhanced KG (Auto-build from Captions)
# -----------------------------
def build_kg_from_captions(df_captions):
    g = Graph()
    EX = Namespace("http://animal.org/")
    g.bind("ex", "http://animal.org/")
    
    # Common entities
    animals = ['dog', 'cat', 'cow', 'horse', 'sheep', 'deer']
    behaviors = ['lying', 'running', 'standing', 'sitting', 'eating', 'sleeping']
    envs = ['grass', 'barn', 'field', 'road', 'forest', 'indoor', 'outdoor']
    
    for a in animals: g.add((EX[a], RDF.type, EX.animal))
    for b in behaviors: g.add((EX[b], RDF.type, EX.behavior))
    for e in envs: g.add((EX[e], RDF.type, EX.environment))
    
    # From captions
    for _, row in df_captions.iterrows():
        cap = str(row['caption']).strip()
        if not cap:
            continue
        words = re.findall(r'\b\w+\b', cap.lower())
        ent = next((w for w in words if w in animals), None)
        if ent:
            g.add((EX[ent], EX.in_scene, Literal(cap)))
    
    g.serialize(KG_PATH, format="turtle")
    return g

def kg_reasoning(query, kg_graph):
    """
    Mở rộng truy vấn một cách có kiểm soát:
    - Chỉ mở rộng thực thể chính (ví dụ: dog -> puppy, hound)
    - Không thêm môi trường hoặc hành động nếu đã có trong query
    - Giữ trọng tâm ý nghĩa gốc để tránh lệch kết quả
    """
    expansions = {
        'dog': ['puppy', 'hound'],
        'cat': ['kitten'],
        'cow': ['bull', 'calf'],
        'horse': ['stallion', 'pony'],
        'sheep': ['lamb'],
        'deer': ['fawn'],
        # chỉ mở rộng hành động
        'lying': ['resting'],
        'running': ['jogging'],
        'standing': ['posing'],
        'sitting': ['resting'],
        'eating': ['feeding'],
        'sleeping': ['resting'],
    }

    words = re.findall(r'\b\w+\b', query.lower())
    expanded_words = []

    for w in words:
        expanded_words.append(w)
        if w in expansions:
            for syn in expansions[w]:
                # chỉ thêm nếu synonym chưa có trong query
                if syn not in words:
                    expanded_words.append(syn)

    expanded_query = " ".join(expanded_words)
    print(f"KG Expanded (controlled): {expanded_query}")
    return expanded_query


# -----------------------------
# 📚 6. Doc2Vec Context (safe version)
# -----------------------------
def get_doc2vec(history):
    """
    Trả về vector ngữ cảnh (numpy array) cho lịch sử truy vấn gần nhất.
    Nếu không đủ dữ liệu (ít hơn 2 câu hoặc trống), trả về None.
    """
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument

    # kiểm tra dữ liệu đủ dùng chưa
    if not history or len(history) < 2:
        return None

    # lọc bỏ dòng trống / không phải chuỗi
    docs = []
    for i, h in enumerate(history[-10:]):  # chỉ lấy 10 câu gần nhất
        if not isinstance(h, str):
            continue
        tokens = [t for t in h.split() if t.strip()]
        if len(tokens) == 0:
            continue
        docs.append(TaggedDocument(words=tokens, tags=[str(i)]))

    # nếu sau khi lọc còn < 2 doc → không đủ train
    if len(docs) < 2:
        return None

    # tạo model trống, build vocab và train rõ ràng
    model = Doc2Vec(vector_size=512, epochs=20, min_count=1)
    model.build_vocab(docs)
    model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)

    # infer vector cho câu cuối cùng
    last_tokens = docs[-1].words
    vec = model.infer_vector(last_tokens)
    return vec


# -----------------------------
# 🔍 7. FAISS Search
# -----------------------------
@st.cache_data
def load_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(EMBEDS_PATH):
        st.warning("⚠️ No FAISS index found. Please build the dataset first.")
        return None, None
    return faiss.read_index(INDEX_PATH), np.load(EMBEDS_PATH, allow_pickle=True).astype('float32')

def search(query_emb, top_k=TOP_K):
    # Load FAISS index
    index = faiss.read_index(INDEX_PATH)
    if index is None:
        st.error("❌ Index not found. Build dataset first.")
        return [], []

    # Load FAISS ID map + CSV
    faiss_id_map = np.load("data/faiss_id_map.npy", allow_pickle=True)
    df = pd.read_csv(CAPTIONS_PATH)

    # Search
    D, I = index.search(np.array([query_emb]), top_k)

    # Map FAISS index sang ID trong CSV
    retrieved_indices = I[0].tolist() if isinstance(I, np.ndarray) else [I]
    retrieved_ids = [int(df.iloc[faiss_id_map[i]]['id']) for i in retrieved_indices]

    distances = D[0].tolist() if isinstance(D, np.ndarray) else [D]

    return distances, retrieved_ids


# -----------------------------
# 🎮 9. Streamlit UI - TABS
# -----------------------------
# st.set_page_config(page_title="🐾 Animal Dataset Manager", layout="wide")
def main():
    st.title("🧠 **Animal Research / Dataset Management**")  #Text-to-Image Retrieval for Multimodal Recommendation
    st.markdown("**Multimodal Retrieval + Smart Dataset Insights** | Thesis Demo Ready! 🚀")

    # 1️⃣ Upload dataset
    st.subheader("📤 Upload your dataset (.zip images)")
    uploaded_zip = st.file_uploader("Upload .zip file of images", type=["zip"])

    if uploaded_zip is not None:
        tmp_extract = "data/tmp_zip"
        os.makedirs(tmp_extract, exist_ok=True)
        os.makedirs(IMAGE_FOLDER, exist_ok=True)

        # ✅ Giải nén toàn bộ file ZIP
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            zip_ref.extractall(tmp_extract)

        # ✅ Duyệt đệ quy và di chuyển toàn bộ ảnh sang IMAGE_FOLDER
        count = 0
        for root, dirs, files in os.walk(tmp_extract):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    src = os.path.join(root, file)
                    dst = os.path.join(IMAGE_FOLDER, file)

                    # Chuẩn hoá đường dẫn và hỗ trợ Windows
                    src = os.path.normpath(os.path.abspath(src))
                    dst = os.path.normpath(os.path.abspath(dst))
                    if os.name == "nt":
                        src = "\\\\?\\" + src
                        dst = "\\\\?\\" + dst

                    try:
                        shutil.move(src, dst)
                        count += 1
                    except Exception as e:
                        st.warning(f"⚠️ Skipping file {file}: {e}")

        # ✅ Xóa thư mục tạm
        shutil.rmtree(tmp_extract, ignore_errors=True)

        st.success(f"✅ Extracted dataset to {IMAGE_FOLDER}")
        st.info(f"📸 Found {count} images")

        # ✅ Chỉ xóa dữ liệu nếu đây là file ZIP mới (khác lần trước)
        if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_zip.name:
            st.session_state.last_uploaded = uploaded_zip.name
            for f in [CAPTIONS_PATH, "data/faiss.index", KG_PATH]:
                if os.path.exists(f):
                    os.remove(f)
            st.info("🧹 Old dataset cleared (new upload detected).")


        # ✅ Preview vài ảnh đầu tiên
        image_files = [
            os.path.join(IMAGE_FOLDER, f)
            for f in os.listdir(IMAGE_FOLDER)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        st.write(f"📸 Found {len(image_files)} images total:")
        st.write(image_files[:5])

        cols = st.columns(4)
        for i, img_path in enumerate(image_files[:8]):
            with cols[i % 4]:
                st.image(img_path, use_column_width=True)

        # ⚙️ Tự động build model + dataset + index
        with st.spinner("🔨 Building dataset + FAISS index... (please wait)"):
            t0 = time.perf_counter()
            # 1. Load models
            models = load_models()

            # 2. Build dataset + caption
            image_paths, df_captions = build_dataset(models)

            # 3. Build KG
            build_kg_from_captions(df_captions)

            # 4. Build FAISS index
            # build_dataset(df_captions, save_path="data/faiss.index")
            build_time = time.perf_counter() - t0

        st.success(f"⏱️ Build completed in **{build_time:.2f} seconds**")

        st.success("✅ Dataset built and FAISS index ready!")

        # ✅ Lưu captions
        df_captions.to_csv(CAPTIONS_PATH, index=False)
        st.success("✅ Captions saved successfully!")

        st.info("🎯 You can now perform text or image search below.")

    # -------------------
    # Tabs: Text / Image / Eval
    # -------------------
    tab_text, tab_image, tab_eval = st.tabs(["🔍 Search = Text", "🖼️ Search = Image", "📈 Eval"])

    # -------------------
    # 🧠 Tab 1: Text Search
    # -------------------
    with tab_text:
        st.subheader("🔍 Text-based Search")
        text_query = st.text_input("Enter your query (e.g., 'a dog running on grass')", "")

        if st.button("🚀 Run Text Search"):
            if not text_query:
                st.warning("Please enter a text query.")
            else:
                query_emb = clip_embed_text(text_query, models[0], models[5])
                kg_query = kg_reasoning(text_query, Graph().parse(KG_PATH))
                st.info(f"**KG Expanded:** {kg_query}")

                if "history" not in st.session_state:
                    st.session_state.history = []

                st.session_state.history.append(text_query)
                context_vec = get_doc2vec(st.session_state.history)
                if context_vec is not None:
                    query_emb = (1 - ALPHA_DOC) * query_emb + ALPHA_DOC * context_vec[:512]
                    query_emb /= np.linalg.norm(query_emb)

                
                distances, retrieved_ids = search(query_emb)

                print("================================")
                print("QUERY:", text_query)
                print("FAISS Returned Image IDs:", retrieved_ids)
                print("Distances:", distances)

                st.markdown(f"### 🔍 Query: {text_query}")
                st.write("📌 FAISS Returned Image IDs:", retrieved_ids)
                st.write("📏 Distances:", distances)

                for idx in retrieved_ids:
                    img_path = image_paths[idx]
                    st.image(img_path, caption=f"Image ID: {idx}", width=150)

                print("================================")

                df = pd.read_csv(CAPTIONS_PATH)
                df['path'] = df['path'].apply(lambda p: p.replace('\\', '/'))
                image_paths = df['path'].tolist()

                st.subheader("🔎 Results:")
                cols = st.columns(3)
                for i, idx in enumerate(retrieved_ids):
                    if idx < len(image_paths):
                        img = Image.open(image_paths[int(idx)])
                        cap = df.iloc[int(idx)]['caption']
                        score = 1 - distances[i]

                        with cols[i % 3]:
                            st.image(
                                img,
                                caption=f"**{cap}**\nScore: {score:.3f}",
                                use_column_width=True
            )
                

    # -------------------
    # 🖼️ Tab 2: Image Search
    # -------------------
    with tab_image:
        st.subheader("🖼️ Image-based Search")

        uploaded_img = st.file_uploader("Upload image query", type=["jpg", "png"])
        if uploaded_img is not None:
            img_query = Image.open(uploaded_img).convert("RGB")
            st.image(img_query, caption="Query Image")

            caption = caption_image(img_query, models[2], models[3], models[5])
            st.write(f"**BLIP Caption:** {caption}")
            query_emb = clip_embed_image(img_query, models[1], models[0], models[5])

            # D, I = search(query_emb)
            distances, retrieved_ids = search(query_emb)

            df = pd.read_csv(CAPTIONS_PATH)
            df['path'] = df['path'].apply(lambda p: p.replace('\\', '/'))
            image_paths = df['path'].tolist()

            st.subheader("🔎 Similar Images:")
            # cols = st.columns(3)
            # for i, (d, idx) in enumerate(zip(D[:TOP_K], I[:TOP_K])):
            #     if idx < len(image_paths):
            #         img = Image.open(image_paths[int(idx)])
            #         cap = df.iloc[int(idx)]['caption']
            #         with cols[i % 3]:
            #             st.image(img, caption=f"**{cap}**\nScore: {1-d:.3f}", use_column_width=True)


            cols = st.columns(3)
            for i, idx in enumerate(retrieved_ids):
                if idx < len(image_paths):
                    img = Image.open(image_paths[int(idx)])
                    cap = df.iloc[int(idx)]['caption']
                    score = 1 - distances[i]

                    with cols[i % 3]:
                        st.image(
                                img,
                                caption=f"**{cap}**\nScore: {score:.3f}",
                                use_column_width=True
            )
    # -------------------
    # 📈 Tab 3: Eval
    # -------------------
    with tab_eval:
        st.subheader("📈 System Evaluation & Metrics")

        if st.button("🚀 Run Full Evaluation", type="primary"):
            if not os.path.exists(INDEX_PATH):
                st.error("❌ Chưa có FAISS index! Vui lòng upload dataset và build trước.")
                st.stop()

            if not os.path.exists(CAPTIONS_PATH):
                st.error("❌ Chưa có captions! Vui lòng build dataset trước.")
                st.stop()

            # Load models để tạo embedding query
            with st.spinner("Đang load CLIP model để đánh giá..."):
                clip_model, preprocess, blip_model, blip_proc, text_model, device = load_models()

            # =======================
            # Test Queries + Ground Truth
            # =======================
            test_queries = [
                "a black dog laying on the grass",
                "a dog swimming in a pool",
                "a grey cat sleeping on a laptop"
            ]

            ground_truth = [
                [33, 35, 44],
                [42, 41, 40],
                [21, 22, 19]
            ]

            all_retrieved = []
            progress = st.progress(0)
            status_text = st.empty()

            for i, query in enumerate(test_queries):
                status_text.text(f"Đang xử lý query {i+1}/{len(test_queries)}: {query}")

                # 1. Tạo embedding từ text query
                query_emb = clip_embed_text(query, clip_model, device)

                # 2. (Tùy chọn) KG Expansion để tăng recall
                try:
                    kg = Graph().parse(KG_PATH, format="turtle")
                    expanded = kg_reasoning(query, kg)
                    emb2 = clip_embed_text(expanded, clip_model, device)
                    query_emb = 0.75 * query_emb + 0.25 * emb2
                    query_emb = query_emb / np.linalg.norm(query_emb)
                except:
                    pass  # nếu lỗi KG thì bỏ qua

                # 3. Search FAISS
                distances, retrieved_ids = search(query_emb, top_k=10)
                all_retrieved.append(retrieved_ids[:10])  # lấy top 10 để tính MRR chính xác

                progress.progress((i + 1) / len(test_queries))

            status_text.text("Đang tính toán metrics...")

            # ==============================
            # ĐÁNH GIÁ BẰNG eval_engine.py CỦA BẠN
            # ==============================
            from eval_engine import evaluate_system

            df_results = evaluate_system(
                query_results=all_retrieved,
                ground_truth=ground_truth,
                k=5,
                save_path="eval_results.csv"
            )

            # Hiển thị kết quả đẹp
            st.success("🎉 Đánh giá hoàn tất!")

            # Làm đẹp bảng kết quả
            df_display = df_results.copy()
            if "AVG" in df_display.index:
                df_display = df_display.drop("AVG")
            df_display = df_display.round(4)
            # st.dataframe(df_display.style.highlight_max(axis=0), use_container_width=True)

            # Hiển thị metrics trung bình
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Precision@5", f"{df_results.iloc[-1]['Precision@K']:.4f}")
            with col2:
                st.metric("Recall@5", f"{df_results.iloc[-1]['Recall@K']:.4f}")
            with col3:
                st.metric("MRR", f"{df_results.iloc[-1]['MRR']:.4f}")

            st.dataframe(df_display.style.highlight_max(axis=0), use_container_width=True)

            # Nút tải kết quả
            csv = df_results.to_csv(index=False).encode()
            st.download_button(
                label="📥 Tải kết quả đánh giá (CSV)",
                data=csv,
                file_name=f"eval_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

                
if __name__ == "__main__":
    main()