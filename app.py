"""
Mineral Classifier — Streamlit app
-----------------------------------
Loads the weights produced by the "YOLOv8_CLS_Mineral_Classification" notebook
and lets a user classify an uploaded mineral photo with the model of their choice.

Drop this file (and requirements.txt / .streamlit/config.toml) into the SAME
folder that contains your trained weight files:

    yolov8x_cls_best.pt        (YOLOv8x-CLS)
    resnet50_minerals.pth      (ResNet50 baseline)
    resnet101_minerals.pth     (ResNet101 baseline)
    rd_minnet120.pth           (RD-MinNet, the novel architecture)

The app also looks inside a "minerals_results/weights" subfolder (the notebook's
default output location), so you can point it at either layout.

Run with:  streamlit run app.py
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision import models as tv_models

# --------------------------------------------------------------------------
# Page setup / light theme
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Mineral Classifier",
    page_icon="🪨",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #FFFFFF; }
    section[data-testid="stSidebar"] { background-color: #F4F1EC; }
    .mineral-card {
        background-color: #F9F7F3;
        border: 1px solid #E7E1D6;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        margin-top: 0.6rem;
    }
    .pred-name {
        font-size: 1.6rem;
        font-weight: 700;
        color: #6B4A2C;
        text-transform: capitalize;
    }
    .pred-conf {
        font-size: 1rem;
        color: #7A7266;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Constants — must match the notebook exactly
# --------------------------------------------------------------------------
# Order used for the raw CLASSES list / RD-MinNet (NOT alphabetical)
CLASSES = ["bornite", "quartz", "malachite", "pyrite", "muscovite", "biotite", "chrysocolla"]

# torchvision.datasets.ImageFolder sorts folder names alphabetically, and the
# ResNet baselines were trained against that ordering.
RESNET_CLASSES = sorted(CLASSES)

MINERAL_PROPERTIES = {
    "bornite":     {"luster": "metallic", "hardness": 3.00},
    "pyrite":      {"luster": "metallic", "hardness": 6.25},
    "quartz":      {"luster": "vitreous", "hardness": 7.00},
    "chrysocolla": {"luster": "vitreous", "hardness": 3.50},
    "malachite":   {"luster": "silky",    "hardness": 3.75},
    "muscovite":   {"luster": "pearly",   "hardness": 2.25},
    "biotite":     {"luster": "pearly",   "hardness": 2.75},
}
LUSTER_CATEGORIES = sorted(set(v["luster"] for v in MINERAL_PROPERTIES.values()))

YOLO_WEIGHTS_NAME = "yolov8x_cls_best.pt"
RESNET50_WEIGHTS_NAME = "resnet50_minerals.pth"
RESNET101_WEIGHTS_NAME = "resnet101_minerals.pth"
RDMINNET_WEIGHTS_NAME = "rd_minnet120.pth"

APP_DIR = Path(__file__).resolve().parent
SEARCH_DIRS = [APP_DIR, APP_DIR / "minerals_results" / "weights", APP_DIR / "weights"]


def find_weights(filename: str) -> Path | None:
    for d in SEARCH_DIRS:
        candidate = d / filename
        if candidate.exists():
            return candidate
    return None


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------
# Transforms (identical to the notebook's eval-time transforms)
# --------------------------------------------------------------------------
resnet_eval_tfms = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

rdminnet_eval_tfms = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
    ]
)

# --------------------------------------------------------------------------
# RD-MinNet architecture (copied verbatim from the notebook so state_dict
# keys line up when loading rd_minnet120.pth)
# --------------------------------------------------------------------------
class SpecularDiffuseDecomposer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, img):
        mask_logits = self.net(img)
        mask = torch.sigmoid(mask_logits)
        diffuse = img * (1 - mask)
        specular = img * mask

        maxc, _ = torch.max(img, dim=1, keepdim=True)
        minc, _ = torch.min(img, dim=1, keepdim=True)
        sat = (maxc - minc) / (maxc + 1e-6)
        prior = maxc * (1 - sat)
        prior_loss = F.mse_loss(mask, prior.detach())

        return diffuse, specular, mask, prior_loss


class SpecularBranch(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()

        def block(cin, cout, stride=2):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, out_dim))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return x


class RDMinNet(nn.Module):
    def __init__(self, n_classes=7, n_luster=4,
                 use_specular_branch=True, use_luster_aux=True, use_hardness_aux=True):
        super().__init__()
        self.use_specular_branch = use_specular_branch
        self.use_luster_aux = use_luster_aux
        self.use_hardness_aux = use_hardness_aux

        self.decomposer = SpecularDiffuseDecomposer()

        effnet = tv_models.efficientnet_b0(weights=None)
        self.diffuse_backbone = effnet.features
        self.diffuse_pool = nn.AdaptiveAvgPool2d(1)
        diffuse_dim = 1280

        if use_specular_branch:
            self.specular_branch = SpecularBranch(out_dim=128)
            fusion_in = diffuse_dim + 128
        else:
            self.specular_branch = None
            fusion_in = diffuse_dim

        self.fusion = nn.Sequential(nn.Linear(fusion_in, 256), nn.ReLU(inplace=True), nn.Dropout(0.3))
        self.class_head = nn.Linear(256, n_classes)
        self.luster_head = nn.Linear(256, n_luster) if use_luster_aux else None
        self.hardness_head = nn.Linear(256, 1) if use_hardness_aux else None

    def forward(self, img):
        diffuse, specular, mask, prior_loss = self.decomposer(img)
        d_feat = self.diffuse_backbone(diffuse)
        d_feat = self.diffuse_pool(d_feat).flatten(1)

        if self.use_specular_branch:
            s_feat = self.specular_branch(specular)
            fused_in = torch.cat([d_feat, s_feat], dim=1)
        else:
            fused_in = d_feat

        fused = self.fusion(fused_in)
        class_logits = self.class_head(fused)
        luster_logits = self.luster_head(fused) if self.use_luster_aux else None
        hardness_pred = self.hardness_head(fused).squeeze(-1) if self.use_hardness_aux else None

        return {
            "class_logits": class_logits,
            "luster_logits": luster_logits,
            "hardness_pred": hardness_pred,
            "specular_mask": mask,
        }


def build_resnet(arch: str, n_classes: int) -> nn.Module:
    if arch == "resnet50":
        m = tv_models.resnet50(weights=None)
    elif arch == "resnet101":
        m = tv_models.resnet101(weights=None)
    else:
        raise ValueError(arch)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


# --------------------------------------------------------------------------
# Cached model loaders — each is only loaded once per session
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLOv8x-CLS…")
def load_yolo(weights_path: str):
    from ultralytics import YOLO
    return YOLO(weights_path)


@st.cache_resource(show_spinner="Loading ResNet…")
def load_resnet(arch: str, weights_path: str):
    m = build_resnet(arch, n_classes=len(RESNET_CLASSES))
    state = torch.load(weights_path, map_location=DEVICE)
    m.load_state_dict(state)
    m.to(DEVICE)
    m.eval()
    return m


@st.cache_resource(show_spinner="Loading RD-MinNet…")
def load_rdminnet(weights_path: str):
    m = RDMinNet(n_classes=len(CLASSES), n_luster=len(LUSTER_CATEGORIES))
    state = torch.load(weights_path, map_location=DEVICE)
    m.load_state_dict(state)
    m.to(DEVICE)
    m.eval()
    return m


# --------------------------------------------------------------------------
# Inference helpers — each returns a pandas Series of class -> probability
# --------------------------------------------------------------------------
def predict_yolo(model, image: Image.Image) -> pd.Series:
    results = model.predict(image, imgsz=224, verbose=False)
    r = results[0]
    probs = r.probs.data.cpu().numpy()
    names = [r.names[i] for i in range(len(probs))]
    return pd.Series(probs, index=names).sort_values(ascending=False)


def predict_resnet(model, image: Image.Image) -> pd.Series:
    x = resnet_eval_tfms(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return pd.Series(probs, index=RESNET_CLASSES).sort_values(ascending=False)


def predict_rdminnet(model, image: Image.Image):
    x = rdminnet_eval_tfms(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        cls_probs = F.softmax(out["class_logits"], dim=1).squeeze(0).cpu().numpy()
        luster_probs = F.softmax(out["luster_logits"], dim=1).squeeze(0).cpu().numpy()
        hardness = float(out["hardness_pred"].item())
    cls_series = pd.Series(cls_probs, index=CLASSES).sort_values(ascending=False)
    luster_series = pd.Series(luster_probs, index=LUSTER_CATEGORIES).sort_values(ascending=False)
    return cls_series, luster_series, hardness


MODEL_OPTIONS = {
    "YOLOv8x-CLS (best overall)": ("yolo", YOLO_WEIGHTS_NAME),
    "RD-MinNet (novel, adds luster & hardness)": ("rdminnet", RDMINNET_WEIGHTS_NAME),
    "ResNet50 (baseline)": ("resnet50", RESNET50_WEIGHTS_NAME),
    "ResNet101 (baseline)": ("resnet101", RESNET101_WEIGHTS_NAME),
}

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("🪨 Settings")
model_label = st.sidebar.radio("Model", list(MODEL_OPTIONS.keys()), index=0)
model_kind, weights_name = MODEL_OPTIONS[model_label]

st.sidebar.markdown("---")
st.sidebar.caption(
    "Weights are looked up next to `app.py`, or in a `minerals_results/weights` "
    "subfolder — matching the notebook's output layout."
)
with st.sidebar.expander("Expected weight files"):
    for name in [YOLO_WEIGHTS_NAME, RDMINNET_WEIGHTS_NAME, RESNET50_WEIGHTS_NAME, RESNET101_WEIGHTS_NAME]:
        found = find_weights(name) is not None
        st.write(("✅ " if found else "❌ ") + name)

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
st.title("Mineral Classifier")
st.write(
    "Upload a photo of a mineral specimen and classify it using any of the "
    "models trained in the research notebook (YOLOv8x-CLS, ResNet baselines, "
    "or the custom RD-MinNet)."
)

uploaded = st.file_uploader("Upload a mineral image", type=["jpg", "jpeg", "png", "bmp", "webp"])

col_img, col_result = st.columns([1, 1])

if uploaded is not None:
    image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    with col_img:
        st.image(image, caption="Uploaded image", width="stretch")

    weights_path = find_weights(weights_name)

    if weights_path is None:
        st.error(
            f"Couldn't find `{weights_name}`. Place it next to `app.py` "
            f"(or under `minerals_results/weights/`) and reload the page."
        )
        st.stop()

    with st.spinner("Classifying…"):
        if model_kind == "yolo":
            model = load_yolo(str(weights_path))
            probs = predict_yolo(model, image)
            luster_series, hardness = None, None
        elif model_kind == "resnet50":
            model = load_resnet("resnet50", str(weights_path))
            probs = predict_resnet(model, image)
            luster_series, hardness = None, None
        elif model_kind == "resnet101":
            model = load_resnet("resnet101", str(weights_path))
            probs = predict_resnet(model, image)
            luster_series, hardness = None, None
        else:  # rdminnet
            model = load_rdminnet(str(weights_path))
            probs, luster_series, hardness = predict_rdminnet(model, image)

    top_class = probs.index[0]
    top_conf = float(probs.iloc[0])

    with col_result:
        st.markdown(
            f"""
            <div class="mineral-card">
                <div class="pred-name">{top_class}</div>
                <div class="pred-conf">Confidence: {top_conf*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        props = MINERAL_PROPERTIES.get(top_class)
        if props:
            st.markdown(
                f"""
                <div class="mineral-card">
                    <b>Reference properties</b><br>
                    Luster: {props['luster'].capitalize()}<br>
                    Mohs hardness: {props['hardness']}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if model_kind == "rdminnet":
            st.markdown(
                f"""
                <div class="mineral-card">
                    <b>RD-MinNet auxiliary predictions</b><br>
                    Predicted luster: {luster_series.index[0].capitalize()}
                    ({luster_series.iloc[0]*100:.1f}%)<br>
                    Predicted hardness: {hardness:.2f}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("Class probabilities")
    st.bar_chart(probs)
    st.dataframe(
        probs.rename("probability").to_frame().style.format({"probability": "{:.2%}"}),
        width="stretch",
    )
else:
    st.info("Upload an image to get a prediction.")