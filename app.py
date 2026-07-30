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

# ---------- Custom CSS for dark theme ----------
st.markdown("""
<style>
    .stApp { background-color: #0e1a17; color: #e0e0e0; }
    .main-title { font-size: 3rem; font-weight: 800; color: white; }
    .accent { color: #4fd1a5; }
    .subtitle-tag { color: #4fd1a5; letter-spacing: 3px; font-size: 0.85rem; font-weight: 600; }
    .card { background-color: #16261f; border-radius: 12px; padding: 1.5rem; border: 1px solid #223530; }
    .disclaimer-box { background-color: #1a1a1a; border-left: 4px solid #f0a500; padding: 1rem; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

device = torch.device('cpu')

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
DISEASE_FULL_NAMES = {
    'akiec': 'Actinic Keratosis / Intraepithelial Carcinoma',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis-like Lesion',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevus (Mole)',
    'vasc': 'Vascular Lesion'
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
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
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
        results.append({'code': cls, 'name': DISEASE_FULL_NAMES[cls], 'confidence': round(prob.item() * 100, 2)})

    pred_idx = top3_idx[0].item()
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_tensor[0].cpu().numpy().transpose(1, 2, 0)
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    return results, visualization

# ---------- TABS ----------
tab1, tab2, tab3 = st.tabs(["🏠  Home / Scan", "📋  ABCDE Rule", "🩺  Skin Disease Guide"])

# ================= TAB 1: HOME / SCAN =================
with tab1:
    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        st.markdown('<p class="subtitle-tag">PRELIMINARY SCREENING TOOL</p>', unsafe_allow_html=True)
        st.markdown('<p class="main-title">Dermato<span class="accent">scan</span></p>', unsafe_allow_html=True)
        st.write("Upload a photo of a skin lesion and get an instant, explainable read on what it might be — trained on 10,015 clinically-labeled dermatoscopic images.")

        uploaded_file = st.file_uploader("Upload a skin lesion image", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption="Uploaded Image", use_container_width=True)

            with st.spinner("Analyzing..."):
                results, gradcam_img = predict(img)

            st.image(gradcam_img, caption="AI Focus Area (Grad-CAM)", use_container_width=True)

            st.subheader("Top 3 Possible Conditions")
            for r in results:
                st.write(f"**{r['name']}** ({r['code']}) — {r['confidence']}%")
                st.progress(r['confidence'] / 100)

            if results[0]['code'] in ['mel', 'bcc', 'akiec']:
                st.error("⚠️ This result includes a potentially serious classification. Please consult a dermatologist promptly.")

    with col_right:
        st.image("https://images.unsplash.com/photo-1584982751601-97dcc096659c?w=600", use_container_width=True)

    st.markdown("""
    <div class="disclaimer-box">
    This tool offers a preliminary read only — it is not a diagnosis. Always have any lesion of concern examined by a dermatologist.
    </div>
    """, unsafe_allow_html=True)

# ================= TAB 2: ABCDE RULE =================
with tab2:
    st.markdown('<p class="main-title">The <span class="accent">ABCDE</span> Rule</p>', unsafe_allow_html=True)
    st.write("A simple guide dermatologists use to spot warning signs of melanoma. If a mole shows several of these features, get it checked.")

    abcde = [
        ("A — Asymmetry", "One half of the mole doesn't match the other half in shape."),
        ("B — Border", "Edges are irregular, ragged, notched, or blurred — not smooth."),
        ("C — Color", "Color is not uniform; may include shades of brown, black, tan, red, white, or blue."),
        ("D — Diameter", "Larger than 6mm (about the size of a pencil eraser), though melanomas can be smaller."),
        ("E — Evolving", "The mole is changing in size, shape, color, or elevation over time, or develops new symptoms like bleeding or itching.")
    ]

    for title, desc in abcde:
        st.markdown(f"""
        <div class="card" style="margin-bottom: 1rem;">
            <h4 class="accent">{title}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.info("💡 This rule is a screening guide, not a diagnostic tool. Any new, unusual, or changing mole should be evaluated by a dermatologist regardless of how many ABCDE criteria it meets.")

# ================= TAB 3: SKIN DISEASE GUIDE =================
with tab3:
    st.markdown('<p class="main-title">Skin Disease <span class="accent">Guide</span></p>', unsafe_allow_html=True)
    st.write("Brief overviews of the 7 conditions this tool screens for.")

    disease_info = {
        "Melanoma (mel)": "The most dangerous form of skin cancer, arising from pigment-producing cells. Can spread rapidly if untreated — early detection is critical.",
        "Basal Cell Carcinoma (bcc)": "The most common form of skin cancer. Grows slowly and rarely spreads, but can cause local tissue damage if untreated.",
        "Actinic Keratosis (akiec)": "A rough, scaly patch caused by years of sun exposure. Considered precancerous — can develop into squamous cell carcinoma.",
        "Melanocytic Nevus (nv)": "A common mole. Almost always benign, though changes over time should still be monitored.",
        "Benign Keratosis-like Lesion (bkl)": "A group of non-cancerous growths including seborrheic keratoses and solar lentigines — very common, especially with age.",
        "Dermatofibroma (df)": "A firm, harmless skin nodule, often on the legs, caused by minor injury such as an insect bite.",
        "Vascular Lesion (vasc)": "A group of benign blood-vessel-related marks, including angiomas and blood spots."
    }

    for name, desc in disease_info.items():
        st.markdown(f"""
        <div class="card" style="margin-bottom: 1rem;">
            <h4 class="accent">{name}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)
