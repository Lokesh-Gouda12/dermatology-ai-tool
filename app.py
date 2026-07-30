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

st.set_page_config(page_title="Dermatology AI Screening Tool", layout="centered")

device = torch.device('cpu')  # Streamlit Cloud free tier is CPU only

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
        results.append({
            'code': cls,
            'name': DISEASE_FULL_NAMES[cls],
            'confidence': round(prob.item() * 100, 2)
        })

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

st.title("🔬 Dermatology AI Screening Tool")
st.warning("⚠️ This is a preliminary screening tool only, NOT a medical diagnosis. Always consult a dermatologist for any skin concerns.")

uploaded_file = st.file_uploader("Upload a skin lesion image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Analyzing..."):
        results, gradcam_img = predict(img)
    
    with col2:
        st.image(gradcam_img, caption="AI Focus Area (Grad-CAM)", use_container_width=True)
    
    st.subheader("Top 3 Possible Conditions")
    for r in results:
        st.write(f"**{r['name']}** ({r['code']}) — {r['confidence']}%")
        st.progress(r['confidence'] / 100)
    
    if results[0]['code'] in ['mel', 'bcc', 'akiec']:
        st.error("⚠️ This result includes a potentially serious classification. Please consult a dermatologist promptly.")
    
    st.caption("This tool uses an EfficientNet-B0 model trained on the HAM10000 dataset for preliminary screening purposes only.")