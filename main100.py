"""
Animal Text-to-Image Retrieval (Optimized Version)
- BLIP Captioning
- CLIP Multimodal Embedding
- KG Query Expansion
- Doc2Vec Conversational Context
- FAISS Search
- Semantic History for "similar" queries
- Evaluation Tab

Chạy nhanh: dữ liệu load trực tiếp từ thư mục, cache nặng
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import numpy as np
from PIL import Image
import torch
import clip
import faiss
from transformers.models.blip import BlipProcessor, BlipForConditionalGeneration
from rdflib import Graph, Namespace, RDF, Literal
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import streamlit as st
import pandas as pd
import glob
import time

# -------------------------------
# Cấu hình cố định (hard-code)
# -------------------------------
DATA_FOLDER = r"D:\Sem1_25-26\Thesis\source\dataset_animals\subset\subset_100"
CACHE_DIR = "cache100"
os.makedirs(CACHE_DIR, exist_ok=True)

CAPTIONS_PATH = os.path.join(CACHE_DIR, "captions.csv")
EMBEDS_PATH  = os.path.join(CACHE_DIR, "image_embeds.npy")
INDEX_PATH   = os.path.join(CACHE_DIR, "faiss.index")
KG_PATH      = os.path.join(CACHE_DIR, "kg.ttl")

TOP_K = 9
ALPHA_DOC = 0.3

ANIMALS   = ['dog', 'cat', 'cow', 'horse', 'sheep', 'deer']
BEHAVIORS = ['lying', 'running', 'standing', 'sitting', 'eating', 'sleeping', 'swimming']
ENVS      = ['grass', 'barn', 'field', 'road', 'forest', 'indoor', 'outdoor', 'laptop']

# -------------------------------
# Load Models (cached)
# -------------------------------
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"Using device: {device}")

    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    return clip_model, preprocess, blip_processor, blip_model, device

clip_model, preprocess, blip_processor, blip_model, device = load_models()

# -------------------------------
# BLIP Caption single image
# -------------------------------
def caption_image(img):
    inputs = blip_processor(img.convert("RGB"), return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_length=50)
    return blip_processor.decode(out[0], skip_special_tokens=True).lower()

# -------------------------------
# CLIP embed functions
# -------------------------------
def clip_embed_text(text):
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]

def clip_embed_image(img):
    img_t = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(img_t)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]

# -------------------------------
# Build / Load Dataset (cached)
# -------------------------------
@st.cache_resource
def get_or_build_dataset():
    # Kiểm tra cache tồn tại
    if all(os.path.exists(p) for p in [CAPTIONS_PATH, EMBEDS_PATH, INDEX_PATH]):
        # st.info("Loading cached dataset...")
        df = pd.read_csv(CAPTIONS_PATH)
        embeds = np.load(EMBEDS_PATH)
        index = faiss.read_index(INDEX_PATH)
        image_paths = df['path'].tolist()
        return image_paths, df, embeds, index

    # st.info("Building dataset (first run)...")
    t0 = time.time()

    image_paths = sorted(glob.glob(os.path.join(DATA_FOLDER, "*.jpg")) +
                         glob.glob(os.path.join(DATA_FOLDER, "*.png")))
    if not image_paths:
        st.error("Not file image!")
        st.stop()

    captions = []
    embeds_list = []

    progress = st.progress(0)
    for i, path in enumerate(image_paths):
        img = Image.open(path)
        cap = caption_image(img)
        captions.append(cap)

        emb = clip_embed_image(img)
        embeds_list.append(emb)

        progress.progress((i+1)/len(image_paths))

    embeds = np.array(embeds_list, dtype='float32')

    # FAISS
    dim = embeds.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeds)

    # Save
    df = pd.DataFrame({
        'path': image_paths,
        'caption': captions
    })
    df.to_csv(CAPTIONS_PATH, index=False)
    np.save(EMBEDS_PATH, embeds)
    faiss.write_index(index, INDEX_PATH)

    build_time = time.time() - t0
    # st.success(f"Build hoàn tất trong {build_time:.1f} giây. Cache đã được lưu.")

    return image_paths, df, embeds, index

image_paths, df_captions, all_embeds, faiss_index = get_or_build_dataset()

# -------------------------------
# Knowledge Graph
# -------------------------------
def build_kg():
    if os.path.exists(KG_PATH):
        return Graph().parse(KG_PATH, format="turtle")

    g = Graph()
    EX = Namespace("http://animal.org/")
    g.bind("ex", EX)

    for a in ANIMALS:   g.add((EX[a],   RDF.type, EX.Animal))
    for b in BEHAVIORS: g.add((EX[b],   RDF.type, EX.Behavior))
    for e in ENVS:      g.add((EX[e],   RDF.type, EX.Environment))

    for idx, (_, row) in enumerate(df_captions.iterrows()):
        cap = row['caption']
        words = set(re.findall(r'\b\w+\b', cap.lower()))
        animal = next((w for w in words if w in ANIMALS), None)
        if animal:
            cap_node = EX[f"cap_{idx}"]
            g.add((cap_node, EX.hasText, Literal(cap)))
            g.add((EX[animal], EX.appearsIn, cap_node))

            for b in BEHAVIORS:
                if b in words:
                    g.add((EX[animal], EX.hasBehavior, EX[b]))
            for env in ENVS:
                if env in words:
                    g.add((EX[animal], EX.inEnvironment, EX[env]))

    g.serialize(KG_PATH, format="turtle")
    return g

kg_graph = build_kg()

def kg_reasoning(query):
    words = re.findall(r'\b\w+\b', query.lower())
    animal = next((w for w in words if w in ANIMALS), None)
    if not animal:
        return query

    sparql = f"""
    PREFIX ex: <http://animal.org/>
    SELECT DISTINCT ?b ?e
    WHERE {{
        ex:{animal} ex:hasBehavior ?beh .
        BIND(STRAFTER(STR(?beh), "#") AS ?b)
        OPTIONAL {{
            ex:{animal} ex:inEnvironment ?env .
            BIND(STRAFTER(STR(?env), "#") AS ?e)
        }}
    }}
    """

    results = kg_graph.query(sparql)
    expansions = set()
    for row in results:
        if row.b: expansions.add(row.b)
        if row.e: expansions.add(row.e)

    if expansions:
        return query + " " + " ".join(expansions)
    return query

# -------------------------------
# Doc2Vec Context (bộ nhớ lịch sử)
# -------------------------------
def get_doc2vec_context(history):
    if len(history) < 2:
        return None

    docs = [TaggedDocument(words=h.split(), tags=[str(i)]) for i, h in enumerate(history[-10:])]
    if len(docs) < 2:
        return None

    model = Doc2Vec(vector_size=512, min_count=1, epochs=15)
    model.build_vocab(docs)
    model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)

    return model.infer_vector(docs[-1].words)

# -------------------------------
# History & Resolve "similar"
# -------------------------------
def parse_semantic(query):
    q = query.lower()
    return {
        "obj": next((a for a in ANIMALS if a in q), None),
        "act": next((b for b in BEHAVIORS if b in q), None),
        "env": next((e for e in ENVS if e in q), None)
    }

def resolve_query(query, semantic_history):
    q = query.lower()
    if "similar" not in q:
        return query, True

    if not semantic_history:
        return query, False

    last = semantic_history[-1]
    new_obj = next((a for a in ANIMALS if a in q), None)
    obj = new_obj or last["obj"]
    act = last["act"]
    env = last["env"]

    parts = []
    if obj: parts.append(f"a {obj}")
    if act: parts.append(f"is {act}")
    if env: parts.append(f"in the {env}")

    return " ".join(parts), False

# -------------------------------
# Search function
# -------------------------------
def search_faiss(query_emb, top_k=TOP_K):
    q = np.array([query_emb], dtype='float32')
    D, I = faiss_index.search(q, top_k)
    return D[0].tolist(), I[0].tolist()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Text-to-Image Retrieval for Multimodal Recommendation")

tab_text, tab_image = st.tabs(["Search by Text", "Search by Image"])

# Init session state
if "semantic_history" not in st.session_state:
    st.session_state.semantic_history = []
if "history" not in st.session_state:
    st.session_state.history = []

with tab_text:
    query = st.text_input("Enter photo description (English):")

    if query:
        resolved, update_hist = resolve_query(query, st.session_state.semantic_history)
        # st.write("Resolved query:", resolved)

        expanded = kg_reasoning(resolved)
        # st.write("KG expanded:", expanded)

        emb = clip_embed_text(expanded)

        if update_hist:
            st.session_state.semantic_history.append(parse_semantic(resolved))
            st.session_state.history.append(resolved)

        context = get_doc2vec_context(st.session_state.history)
        if context is not None:
            emb = (1 - ALPHA_DOC) * emb + ALPHA_DOC * context[:512]
            emb /= np.linalg.norm(emb)

        distances, indices = search_faiss(emb)

        st.subheader("Similar Images")
        cols = st.columns(3)
        for i, idx in enumerate(indices):
            if idx < len(image_paths):
                path = image_paths[idx]
                cap = df_captions.iloc[idx]['caption']
                score = distances[i]
                with cols[i % 3]:
                    st.image(path, caption=f"{cap}\n- Score: {score:.3f}")

with tab_image:
    uploaded = st.file_uploader("Upload query image", type=["jpg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Query Image", width=400)

        cap = caption_image(img)
        st.write("**BLIP Caption:**", cap)

        emb = clip_embed_image(img)
        distances, indices = search_faiss(emb)

        st.subheader("Similar Images")
        cols = st.columns(3)
        for i, idx in enumerate(indices):
            if idx < len(image_paths):
                path = image_paths[idx]
                cap = df_captions.iloc[idx]['caption']
                score = distances[i]
                with cols[i % 3]:
                    st.image(path, caption=f"{cap}\n- Score: {score:.3f}")

# st.markdown("---")
# conda deactivate