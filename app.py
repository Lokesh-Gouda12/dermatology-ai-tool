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
.stApp { background-color: #0e1a17; }

.section { padding: 4rem 8% 4rem 8%; }

/* ---- HERO ---- */
.tag { color: #4fd1a5; letter-spacing: 4px; font-size: 1rem; font-weight: 700; }
.hero-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 7rem !important;
    font-weight: 700 !important;
    margin: 0.3rem 0 1rem 0 !important;
    color: white !important;
    line-height: 1.0 !important;
    white-space: nowrap !important;
    display: block !important;
}
.hero-title .accent { color: #4fd1a5; }
.hero-desc { font-size: 1.22rem; color: #cfd8d4; max-width: 620px; line-height: 1.8; }
.disclaimer { background-color: #14231c; border-left: 3px solid #f0a500; padding: 0.9rem 1.2rem; border-radius: 6px; color: #ddd; font-size: 0.9rem; margin-top: 2rem; }

/* ---- ABCDE SECTION (dark blue bg, light yellow cards) ---- */
.abcde-wrap { background-color: #101d3a; padding: 4rem 6%; }
.section-tag-dark { text-align:center; color: #7fb2e8; letter-spacing:3px; font-size:0.8rem; font-weight:600; }
.section-title-dark { font-family: 'Playfair Display', serif; text-align:center; font-size: 2.4rem; font-weight:700; color:#fff; margin: 0.3rem 0 0.8rem 0; }
.section-sub-dark { text-align:center !important; color:#c3cbe0; width:100%; margin:0 auto 2.5rem auto; white-space: nowrap; display:block; }

.abcde-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 1.2rem; }
.abcde-card { background:#fdf3d3; border-radius: 14px; padding: 1.8rem 1.2rem; text-align:center; }
.abcde-letter { font-family:'Playfair Display', serif; font-size: 2.6rem; color:#2e7d5b; font-weight:700; }
.abcde-name { font-weight:700; margin: 0.3rem 0 0.6rem 0; color:#1a1a1a; }
.abcde-desc { font-size:0.88rem; color:#555; line-height:1.4; }

/* ---- DISEASE SECTION ---- */
.disease-wrap { background-color: #0e1a17; padding: 4rem 6%; }
.section-tag { text-align:center; color: #4fd1a5; letter-spacing:3px; font-size:0.8rem; font-weight:600; }
.section-title { font-family: 'Playfair Display', serif; text-align:center; font-size: 2.4rem; font-weight:700; color:#fff; margin: 0.3rem 0 0.8rem 0; }
.section-sub { text-align:center !important; color:#c9cfcb; width:100%; margin:0 auto 2.5rem auto; white-space: nowrap; display:block; }

.disease-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; }
.disease-card { background:white; border-radius: 14px; padding: 1.6rem; }
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

def crop_to_ratio(img, target_ratio=0.85):
    """Crop image (center) to make it shorter/less tall. target_ratio = width/height desired."""
    w, h = img.size
    new_h = int(w / target_ratio)
    if new_h < h:
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    return img

# ================= HERO SECTION =================
# ================= SESSION STATE SETUP =================
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'uploaded_img' not in st.session_state:
    st.session_state.uploaded_img = None

# ================= HOME PAGE (HERO) =================
if st.session_state.page == 'home':
    st.markdown('<div class="section">', unsafe_allow_html=True)
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("""
            <p class="tag">PRELIMINARY SCREENING TOOL</p>
            <p class="hero-title">Dermato<span class="accent">scan</span></p>
            <p class="hero-desc">Upload a photo of a skin lesion and get an instant, explainable read on what it might be — trained on 10,015 clinically-labeled dermatoscopic images.</p>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

        if uploaded_file is not None:
            st.session_state.uploaded_img = Image.open(uploaded_file).convert('RGB')
            st.session_state.page = 'results'
            st.rerun()

        st.markdown("""
            <div class="disclaimer">This tool offers a preliminary read only — it is not a diagnosis. Always have any lesion of concern examined by a dermatologist.</div>
        """, unsafe_allow_html=True)

    with col2:
        doctor_img = Image.open("doctor.jpg").convert("RGB")
        doctor_img = crop_to_ratio(doctor_img, target_ratio=0.9)
        st.image(doctor_img, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ================= RESULTS PAGE =================
elif st.session_state.page == 'results':
    st.markdown('<div class="section">', unsafe_allow_html=True)

    if st.button("← Back to Home"):
        st.session_state.page = 'home'
        st.session_state.uploaded_img = None
        st.rerun()

    st.markdown('<p class="hero-title" style="font-size: 3rem;">Analysis <span class="accent">Results</span></p>', unsafe_allow_html=True)

    img = st.session_state.uploaded_img
    with st.spinner("Analyzing..."):
        results, gradcam_img = predict(img)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Uploaded Image", width=350)
    with col2:
        st.image(gradcam_img, caption="AI Focus Area (Grad-CAM)", width=350)

    st.markdown("#### Top 3 Possible Conditions")
    for r in results:
        st.write(f"**{r['name']}** ({r['code']}) — {r['confidence']}%")
        st.progress(r['confidence'] / 100)

    if results[0]['code'] in ['mel', 'bcc', 'akiec']:
        st.error("⚠️ This result includes a potentially serious classification. Please consult a dermatologist promptly.")

    st.markdown("""
        <div class="disclaimer">This tool offers a preliminary read only — it is not a diagnosis. Always have any lesion of concern examined by a dermatologist.</div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.page == 'home':

# ================= ABCDE SECTION (single HTML block — dark blue bg + yellow cards) =================
abcde = [
    ("A", "Asymmetry", "One half of the lesion does not match the other."),
    ("B", "Border", "Edges are irregular, notched, or poorly defined."),
    ("C", "Color", "Uneven shading — mixes of brown, black, red, or white."),
    ("D", "Diameter", "Larger than 6mm — roughly the size of a pencil eraser."),
    ("E", "Evolving", "Any change in size, shape, color, or sensation over time.")
]
abcde_cards_html = "".join([
    f"""<div class="abcde-card">
        <div class="abcde-letter">{letter}</div>
        <div class="abcde-name">{name}</div>
        <div class="abcde-desc">{desc}</div>
    </div>""" for letter, name, desc in abcde
])

st.markdown(f"""
<div class="abcde-wrap">
    <p class="section-tag-dark">SELF-CHECK REFERENCE</p>
    <p class="section-title-dark">The ABCDE Rule</p>
    <p class="section-sub-dark">A widely-used dermatology mnemonic for spotting a mole worth having examined.</p>
    <div class="abcde-grid">{abcde_cards_html}</div>
</div>
""", unsafe_allow_html=True)

# ================= DISEASE GUIDE SECTION (single HTML block, with proper gaps) =================
diseases = [
    ("NV-01", "Melanocytic Nevi", "Common Mole", "Clusters of pigment-producing cells that form ordinary moles. Usually stable for years — the main thing to track is whether one starts changing.", "low"),
    ("MEL-02", "Melanoma", "Malignant Melanoma", "A cancer of pigment-producing cells and the most dangerous common skin cancer. Early detection changes outcomes dramatically.", "high"),
    ("BKL-03", "Benign Keratosis", "Seborrheic Keratosis & related", "Rough, waxy, 'stuck-on' looking patches. Harmless, but can visually mimic more serious lesions.", "low"),
    ("BCC-04", "Basal Cell Carcinoma", "Most Common Skin Cancer", "Grows slowly and rarely spreads, but left untreated can damage surrounding skin and tissue.", "high"),
    ("AKI-05", "Actinic Keratosis", "Sun-Damage Patch", "Rough, scaly patches from cumulative sun exposure. Considered pre-cancerous.", "moderate"),
    ("VAS-06", "Vascular Lesions", "Angioma & related", "Growths involving blood vessels near the skin surface, appearing as red or purple marks. Almost always benign.", "low"),
]
badge_labels = {"low": "LOW RISK", "moderate": "MODERATE RISK", "high": "HIGH RISK"}

disease_cards_html = "".join([
    f"""<div class="disease-card">
        <span class="disease-code">{code}</span>
        <div class="disease-name">{name}</div>
        <div class="disease-sub">{sub}</div>
        <div class="disease-desc">{desc}</div>
        <span class="badge {risk}">{badge_labels[risk]}</span>
    </div>""" for code, name, sub, desc, risk in diseases
])

st.markdown(f"""
<div class="disease-wrap">
    <p class="section-tag">REFERENCE PANEL</p>
    <p class="section-title">Conditions This Model Screens For</p>
    <p class="section-sub">Seven categories, drawn from the HAM10000 dermatoscopic dataset — spanning benign, pre-cancerous, and malignant lesions.</p>
    <div class="disease-grid">{disease_cards_html}</div>
</div>
""", unsafe_allow_html=True)
