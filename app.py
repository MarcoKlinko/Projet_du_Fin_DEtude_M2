# app_gradio.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import gradio as gr
import tempfile
import shutil

# ========================================
# CONFIGURATION
# ========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CHEMIN VERS TON MODÈLE YOLO ENTRAÎNÉ
YOLO_MODEL_PATH = "/home/marco/Projet_du_Fin_DEtude_M2/Models/Model_YOLO/runs/detect/train35/weights/best.pt"

# Vérifie que le modèle existe
if not os.path.exists(YOLO_MODEL_PATH):
    raise FileNotFoundError(f"Modèle YOLO non trouvé : {YOLO_MODEL_PATH}")

yolo_model = YOLO(YOLO_MODEL_PATH)
print("Classes YOLO :", yolo_model.names)  # Doit afficher {0: 'signature'}

# ========================================
# MODÈLE SIAMOIS
# ========================================
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Identity()
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = backbone
        self.embedding = nn.Linear(num_ftrs, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        feat = self.backbone(x)
        emb = self.embedding(feat)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, x1, x2):
        emb1 = self.forward_once(x1)
        emb2 = self.forward_once(x2)
        combined = torch.cat((emb1, emb2), dim=1)
        return self.classifier(combined)

# Charger le modèle Siamese
siamese_model = SiameseNetwork().to(device)
try:
    state_dict = torch.load("/home/marco/Projet_du_Fin_DEtude_M2/Models/Model_siemese/siamese_best.pth", map_location=device)
    siamese_model.load_state_dict(state_dict, strict=False)
    print("Modèle Siamese chargé avec succès.")
except Exception as e:
    print(f"Erreur chargement Siamese : {e}")
siamese_model.eval()

# Transform pour Siamese
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ========================================
# FONCTIONS
# ========================================
def extract_signature(img_np, temp_dir):
    """Extrait la signature avec debug visuel"""
    try:
        temp_path = os.path.join(temp_dir, "cin_temp.jpg")
        cv2.imwrite(temp_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

        # DÉTECTION YOLO AVEC DEBUG
        results = yolo_model.predict(
            temp_path,
            conf=0.25,           # Seuil bas pour éviter les faux négatifs
            iou=0.45,
            verbose=True,
            save=True,           # Sauvegarde l'image annotée
            project="debug_yolo",
            name="inference"
        )

        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    class_name = yolo_model.names[cls]

                    print(f"Détection : {class_name}, confiance={conf:.3f}")

                    if class_name == "signature" and conf > 0.25:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cropped = img_np[y1:y2, x1:x2]
                        return cropped, None

        return None, "Aucune signature détectée (conf > 0.25)"
    except Exception as e:
        return None, f"Erreur YOLO : {str(e)}"

def compare_signatures(sig1, sig2):
    if sig1 is None or sig2 is None:
        return "Erreur : une signature n'a pas été détectée."

    try:
        img1 = transform(Image.fromarray(sig1).convert("L")).unsqueeze(0).to(device)
        img2 = transform(Image.fromarray(sig2).convert("L")).unsqueeze(0).to(device)

        with torch.no_grad():
            prob = siamese_model(img1, img2).item()

        result = "Signatures identiques" if prob > 0.5 else "Signatures différentes"
        return f"**{result}**<br><small>Confiance : {prob:.3f}</small>"
    except Exception as e:
        return f"Erreur comparaison : {e}"

# ========================================
# INTERFACE GRADIO
# ========================================
def process_cin(cin1, cin2):
    with tempfile.TemporaryDirectory() as temp_dir:
        sig1_crop, err1 = extract_signature(cin1, temp_dir)
        sig2_crop, err2 = extract_signature(cin2, temp_dir)

        if err1 or err2:
            error_msg = err1 or err2
            print(f"ERREUR : {error_msg}")
            return None, None, error_msg

        result_text = compare_signatures(sig1_crop, sig2_crop)

        sig1_pil = Image.fromarray(sig1_crop)
        sig2_pil = Image.fromarray(sig2_crop)

        return sig1_pil, sig2_pil, result_text

# Interface
with gr.Blocks(title="Vérification de Signature CIN") as demo:
    gr.Markdown("# Vérification de Signature sur CIN")
    gr.Markdown("Upload deux CIN → Extraction automatique → Comparaison IA")

    with gr.Row():
        cin1_input = gr.Image(label="CIN 1", type="numpy", height=300)
        cin2_input = gr.Image(label="CIN 2", type="numpy", height=300)

    compare_btn = gr.Button("Comparer les Signatures", variant="primary", size="lg")

    with gr.Row():
        sig1_output = gr.Image(label="Signature extraite (CIN 1)", height=200)
        sig2_output = gr.Image(label="Signature extraite (CIN 2)", height=200)

    result_output = gr.HTML()  # Pour HTML (gras, petite police)

    # Action
    compare_btn.click(
        fn=process_cin,
        inputs=[cin1_input, cin2_input],
        outputs=[sig1_output, sig2_output, result_output]
    )

    gr.Markdown("**Debug** : Images annotées dans `debug_yolo/inference/`")

# ========================================
# LANCER
# ========================================
if __name__ == "__main__":
    print("Démarrage de l'application Gradio...")
    print(f"Accès local : http://127.0.0.1:7860")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True  # Lien public (ngrok)
    )