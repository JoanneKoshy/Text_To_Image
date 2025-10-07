#task1
import streamlit as st
import os
import hashlib
from typing import List, Tuple

import numpy as np
from PIL import Image
import streamlit as stx
import torch
from transformers import CLIPProcessor, CLIPModel

# ---------------------------
# Config
# ---------------------------
IMAGE_DIR = "images"  # Put your images here
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
MODEL_NAME = "openai/clip-vit-base-patch32"
TOPK_DEFAULT = 10

st.set_page_config(
    page_title="Text → Image Search (CLIP)",
    page_icon="🔎",
    layout="wide",
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Helpers
# ---------------------------

def list_image_paths(img_dir: str = IMAGE_DIR) -> List[str]:
    os.makedirs(img_dir, exist_ok=True)
    files = [
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ]
    files.sort()
    return files


def dataset_signature(paths: List[str]) -> str:
    h = hashlib.sha256()
    for p in paths:
        try:
            st_ = os.stat(p)
            h.update(p.encode("utf-8"))
            h.update(str(int(st_.st_mtime)).encode("utf-8"))
            h.update(str(st_.st_size).encode("utf-8"))
        except FileNotFoundError:
            continue
    return h.hexdigest()


@st.cache_resource(show_spinner=False)
def load_model(name: str = MODEL_NAME):
    model = CLIPModel.from_pretrained(name).to(device)
    processor = CLIPProcessor.from_pretrained(name)
    return model, processor


@st.cache_data(show_spinner=False)
def build_image_index(paths: List[str], ds_sig: str) -> Tuple[np.ndarray, List[str]]:
    """Encode images → (embeddings[N, D], valid_paths)."""
    model, processor = load_model()
    embs = []
    valid_paths = []

    progress = st.progress(0, text="Embedding images…")
    total = len(paths)
    for i, p in enumerate(paths, start=1):
        try:
            image = Image.open(p).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = model.get_image_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            vec = emb.cpu().numpy()[0]
            embs.append(vec)
            valid_paths.append(p)
        except Exception as e:
            st.sidebar.warning(f"Skipped {os.path.basename(p)}: {e}")
        finally:
            progress.progress(i / max(total, 1))
    progress.empty()

    if len(embs) == 0:
        return np.zeros((0, 512), dtype=np.float32), []

    emb_mat = np.vstack(embs)
    return emb_mat, valid_paths


def cosine_topk(query_vec: np.ndarray, emb_mat: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    if emb_mat.size == 0:
        return np.array([]), np.array([], dtype=int)
    scores = emb_mat @ query_vec.astype(np.float32)
    k = int(min(k, scores.shape[0]))
    topk_idx = np.argpartition(-scores, kth=k - 1)[:k]
    order = np.argsort(-scores[topk_idx])
    topk_idx = topk_idx[order]
    topk_scores = scores[topk_idx]
    return topk_scores, topk_idx


# ---------------------------
# UI
# ---------------------------
st.title("🔎 Text → Image Search (CLIP)")
with st.expander("How to use", expanded=True):
    st.markdown(
        f"""
        1. Add images inside **`{IMAGE_DIR}`** folder.
        2. Click **Rebuild Index** (sidebar) to embed them.
        3. Type a keyword (e.g., "cat", "dog").
        4. Only exact matches will be shown with similarity scores.
        """
    )

# Sidebar controls
st.sidebar.header("⚙️ Controls")
rebuild = st.sidebar.button("🔁 Rebuild Index", use_container_width=True)
K = st.sidebar.slider("Results to show", min_value=5, max_value=50, value=TOPK_DEFAULT, step=1)

# List images & (re)build index if needed
all_paths = list_image_paths(IMAGE_DIR)
if len(all_paths) == 0:
    st.info(f"No images found in **{IMAGE_DIR}/**. Add some images, then click **Rebuild Index**.")

sig = dataset_signature(all_paths)

if rebuild:
    build_image_index.clear()

emb_mat, paths = build_image_index(all_paths, sig)

# Search box
st.subheader("Search")
query = st.text_input("Describe what you're looking for:", placeholder="e.g., 'dog', 'cat'", label_visibility="visible")
search_clicked = st.button("Search", use_container_width=True, type="primary")

if search_clicked and (query.strip() == ""):
    st.warning("Please enter a search query.")

if (search_clicked and query.strip()) or (query.strip() and "autosearch" in st.session_state):
    if emb_mat.size == 0:
        st.error("Index is empty. Add images and click Rebuild Index.")
    else:
        st.session_state["autosearch"] = True

        model, processor = load_model()
        inputs = processor(text=[query.strip()], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_emb = model.get_text_features(**inputs)
        text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
        q_vec = text_emb.cpu().numpy()[0]

        scores, idxs = cosine_topk(q_vec, emb_mat, K)

        # strict filtering: only keep files with query in filename
        filtered = [(s, i) for s, i in zip(scores, idxs) if query.lower() in os.path.basename(paths[i]).lower()]

        if not filtered:
            st.info("No exact matches found. Try adding more images.")
        else:
            cols = st.columns(5, gap="small")
            for i, (score, idx) in enumerate(filtered):
                col = cols[i % len(cols)]
                p = paths[idx]
                with col:
                    st.image(p, caption=f"{os.path.basename(p)}\nscore={score:.3f}", use_container_width=True)

# Footer
st.markdown("---")
with st.expander("About this app"):
    st.markdown(
        """
        **Model**: `openai/clip-vit-base-patch32` (Hugging Face CLIP).  
        **Similarity**: Cosine similarity on normalized embeddings.  
        **Filtering**: Strict filename matching to keep only exact search category.  
        """
    )
