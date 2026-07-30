import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np

st.set_page_config(page_title="Dermatoscan", layout="wide", page_icon="🔬")

# ================= GLOBAL CSS =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;0,700;1,600&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

.block-container { padding-top: 0rem; padding-bottom: 0rem; max-width: 100%; }
header[data-testid="stHeader"] { display: none; }
footer { visibility: hidden; }

.section { padding: 4rem 8% 4rem 8%; }

/* ---- HERO ---- */
.hero { background-color: #0e1e18; color: #f2f2f2; }
.tag { color: #4fd1a5; letter-spacing: 3px; font-size: 0.8rem; font-weight: 600; }
.hero-title { font-family: 'Playfair Display', serif; font-size: 3.2rem; font-weight: 700; margin: 0.3rem 0 1rem 0; color: white; }
.hero-title .accent { color: #4fd1a5; }
.hero-desc { font-size: 1.05rem; color: #cfd8d4; max-width: 500px; line-height: 1.6; }

.disclaimer { background-color: #14231c; border-left: 3px solid #f0a500; padding: 0.9rem 1.2rem; border-radius: 6px; color: #ddd; font-size: 0.9rem; margin-top: 2rem; }

/* ---- ABCDE SECTION ---- */
.abcde-section { background-color: #ece7f5; }
.section-tag { text-align:center; color: #5b4b8a; letter-spacing:3px; font-size:0.8rem; font-weight:600; }
.section-title { font-family: 'Playfair Display', serif; text-align:center; font-size: 2.4rem; font-weight:700; color:#1a1a1a; margin: 0.3rem 0 0.8rem 0; }
.section-sub { text-align:center; color:#4a4a4a; max-width:700px; margin:0 auto 2.5rem auto; }

.abcde-card { background:white; border-radius: 14px; padding: 1.8rem 1.2rem; text-align:center; box-shadow: 0 2px 10px rgba(0,0,0,0.05); height: 100%; }
.abcde-letter { font-family:'Playfair Display', serif; font-size: 2.6rem; color:#2e7d5b; font-weight:700; }
.abcde-name { font-weight:700; margin: 0.3rem 0 0.6rem 0; color:#1a1a1a; }
.abcde-desc { font-size:0.88rem; color:#555; line-height:1.4; }

/* ---- DISEASE SECTION ---- */
.disease-section { background-color: #f5efe6; }
.disease-card { background:white; border-radius: 14px; padding: 1.6rem; box-shadow: 0 2px 10px rgba(0,0,0,0.05); height: 100%; }
.disease-code { float:right; color:#999; font-size:0.75rem; font-family: monospace; }
.disease-name { font-family:'Playfair Display', serif; font-size:1.3rem; font-weight:700; color:#1a1a1a; margin-top:0.6rem;}
.disease-sub { color:#888; font-size:0.85rem; margin-bottom:0.7rem; }
.disease-desc { color:#444; font-size:0.9rem; line-height:1.5; margin-bottom:1rem; }

.badge { display:inline-block; padding: 0.25rem 0.7rem; border-radius: 6px; font-size:0.7rem; font-weight:700; letter-spacing:1px; }
.low { background:#e3f2e9; color:#2e7d5b; }
.moderate { background:#fbe9d0; color:#b8720a; }
.high { background:#f8d9d3; color:#c0392b; }
</style>
""", unsafe_allow_html=True)

device = torch.device('cpu')
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
DISEASE_FULL_NAMES = {
    'akiec': 'Actinic Keratosis', 'bcc': 'Basal Cell Carcinoma', 'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma', 'mel': 'Melanoma', 'nv': 'Melanocytic Nevus', 'vasc': 'Vascular Lesion'
}

@st.cache_resource
def load_model():
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=7)
    model.load_state_dict(torch.load('best_derma_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    return model

model = load_model()
target_layers = [model.conv_head]
cam = GradCAM(model=model, target_layers=target_layers)

transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(img):
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]
    top3_probs, top3_idx = torch.topk(probs, 3)
    results = []
    for prob, idx in zip(top3_probs, top3_idx):
        cls = CLASS_NAMES[idx.item()]
        results.append({'code': cls, 'name': DISEASE_FULL_NAMES[cls], 'confidence': round(prob.item()*100, 2)})
    pred_idx = top3_idx[0].item()
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    img_np = img_tensor[0].cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(std * img_np + mean, 0, 1)
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    return results, visualization

# ================= HERO SECTION =================
st.markdown('<div class="section hero">', unsafe_allow_html=True)
col1, col2 = st.columns([1.3, 1])

with col1:
    st.markdown("""
        <p class="tag">PRELIMINARY SCREENING TOOL</p>
        <p class="hero-title">Dermato<span class="accent">scan</span></p>
        <p class="hero-desc">Upload a photo of a skin lesion and get an instant, explainable read on what it might be — trained on 10,015 clinically-labeled dermatoscopic images.</p>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, use_container_width=True)
        with st.spinner("Analyzing..."):
            results, gradcam_img = predict(img)
        st.image(gradcam_img, caption="AI Focus Area (Grad-CAM)", use_container_width=True)
        st.markdown("#### Top 3 Possible Conditions")
        for r in results:
            st.write(f"**{r['name']}** ({r['code']}) — {r['confidence']}%")
            st.progress(r['confidence'] / 100)
        if results[0]['code'] in ['mel', 'bcc', 'akiec']:
            st.error("⚠️ Potentially serious classification — consult a dermatologist promptly.")

    st.markdown("""
        <div class="disclaimer">This tool offers a preliminary read only — it is not a diagnosis. Always have any lesion of concern examined by a dermatologist.</div>
    """, unsafe_allow_html=True)

with col2:
    st.image("doctor.jpg", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ================= ABCDE SECTION =================
st.markdown('<div class="section abcde-section">', unsafe_allow_html=True)
st.markdown("""
    <p class="section-tag">SELF-CHECK REFERENCE</p>
    <p class="section-title">The ABCDE Rule</p>
    <p class="section-sub">A widely-used dermatology mnemonic for spotting a mole worth having examined.</p>
""", unsafe_allow_html=True)

abcde = [
    ("A", "Asymmetry", "One half of the lesion does not match the other."),
    ("B", "Border", "Edges are irregular, notched, or poorly defined."),
    ("C", "Color", "Uneven shading — mixes of brown, black, red, or white."),
    ("D", "Diameter", "Larger than 6mm — roughly the size of a pencil eraser."),
    ("E", "Evolving", "Any change in size, shape, color, or sensation over time.")
]
cols = st.columns(5)
for c, (letter, name, desc) in zip(cols, abcde):
    with c:
        st.markdown(f"""
        <div class="abcde-card">
            <div class="abcde-letter">{letter}</div>
            <div class="abcde-name">{name}</div>
            <div class="abcde-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ================= DISEASE GUIDE SECTION =================
st.markdown('<div class="section disease-section">', unsafe_allow_html=True)
st.markdown("""
    <p class="section-tag">REFERENCE PANEL</p>
    <p class="section-title">Conditions This Model Screens For</p>
    <p class="section-sub">Seven categories, drawn from the HAM10000 dermatoscopic dataset — spanning benign, pre-cancerous, and malignant lesions.</p>
""", unsafe_allow_html=True)

diseases = [
    ("NV-01", "Melanocytic Nevi", "Common Mole", "Clusters of pigment-producing cells that form ordinary moles. Usually stable for years — the main thing to track is whether one starts changing.", "low"),
    ("MEL-02", "Melanoma", "Malignant Melanoma", "A cancer of pigment-producing cells and the most dangerous common skin cancer. Early detection changes outcomes dramatically.", "high"),
    ("BKL-03", "Benign Keratosis", "Seborrheic Keratosis & related", "Rough, waxy, 'stuck-on' looking patches. Harmless, but can visually mimic more serious lesions.", "low"),
    ("BCC-04", "Basal Cell Carcinoma", "Most Common Skin Cancer", "Grows slowly and rarely spreads, but left untreated can damage surrounding skin and tissue.", "high"),
    ("AKI-05", "Actinic Keratosis", "Sun-Damage Patch", "Rough, scaly patches from cumulative sun exposure. Considered pre-cancerous.", "moderate"),
    ("VAS-06", "Vascular Lesions", "Angioma & related", "Growths involving blood vessels near the skin surface, appearing as red or purple marks. Almost always benign.", "low"),
]

badge_labels = {"low": "LOW RISK", "moderate": "MODERATE RISK", "high": "HIGH RISK"}

for i in range(0, len(diseases), 3):
    row = st.columns(3)
    for c, (code, name, sub, desc, risk) in zip(row, diseases[i:i+3]):
        with c:
            st.markdown(f"""
            <div class="disease-card">
                <span class="disease-code">{code}</span>
                <div class="disease-name">{name}</div>
                <div class="disease-sub">{sub}</div>
                <div class="disease-desc">{desc}</div>
                <span class="badge {risk}">{badge_labels[risk]}</span>
            </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
