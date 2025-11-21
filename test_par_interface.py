# app.py
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
from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import tempfile
import shutil
from datetime import datetime

# ========================================
# CONFIG FLASK
# ========================================
app = Flask(__name__)
app.secret_key = "supersecretkey2025"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DEBUG_YOLO_FOLDER'] = 'debug_yolo'
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB max

# Créer les dossiers
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DEBUG_YOLO_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ========================================
# CHARGEMENT DES MODÈLES
# ========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TON MODÈLE YOLO ENTRAÎNÉ (à adapter si tu as un autre run)
YOLO_PATH = "/home/marco/Projet_du_Fin_DEtude_M2/Models/Model_YOLO/runs/detect/train35/weights/best.pt"
if not os.path.exists(YOLO_PATH):
    raise FileNotFoundError(f"YOLO non trouvé : {YOLO_PATH}")

yolo_model = YOLO(YOLO_PATH)
print(f"Modèle YOLO chargé → classes : {yolo_model.names}")

# Siamese Network
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

# Charger Siamese
siamese_model = SiameseNetwork().to(device)
siamese_path = "/home/marco/Projet_du_Fin_DEtude_M2/Models/Model_siemese/siamese_jupyter.pth"
try:
    state_dict = torch.load(siamese_path, map_location=device)
    siamese_model.load_state_dict(state_dict, strict=False)
    print("Modèle Siamese chargé.")
except Exception as e:
    print(f"Erreur Siamese : {e}")
siamese_model.eval()

# Transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ========================================
# FONCTIONS
# ========================================
def extract_and_save_signature(img_path, debug_folder):
    results = yolo_model.predict(
        img_path,
        conf=0.25,           # seuil bas pour ne rien rater
        iou=0.45,
        save=True,
        project=debug_folder,
        name="latest",
        exist_ok=True
    )

    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                if yolo_model.names[int(box.cls[0])] == "signature" and box.conf[0] > 0.25:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    img = cv2.imread(img_path)
                    cropped = img[y1:y2, x1:x2]
                    return cropped, True
    return None, False

def compare_signatures(sig1, sig2):
    if sig1 is None or sig2 is None:
        return "Impossible de comparer : une signature n'a pas été détectée."

    img1 = transform(Image.fromarray(sig1).convert("L")).unsqueeze(0).to(device)
    img2 = transform(Image.fromarray(sig2).convert("L")).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = siamese_model(img1, img2).item()

    if prob > 0.6:
        result = f"<strong style='color:green;'>Signatures IDENTIQUES</strong>"
    else:
        result = f"<strong style='color:red;'>Signatures DIFFÉRENTES</strong>"

    return f"{result}<br><small>Confiance : {prob:.3f}</small>"

# ========================================
# ROUTES
# ========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    if 'cin1' not in request.files or 'cin2' not in request.files:
        flash("Deux images sont requises.", "danger")
        return redirect(url_for('index'))

    file1 = request.files['cin1']
    file2 = request.files['cin2']

    if file1.filename == '' or file2.filename == '':
        flash("Veuillez sélectionner deux fichiers.", "danger")
        return redirect(url_for('index'))

    if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
        flash("Format non supporté (png, jpg, jpeg, bmp, tiff, webp).", "danger")
        return redirect(url_for('index'))

    # Dossier temporaire
    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = os.path.join(tmpdir, secure_filename(file1.filename))
        path2 = os.path.join(tmpdir, secure_filename(file2.filename))
        file1.save(path1)
        file2.save(path2)

        # Extraction
        sig1, ok1 = extract_and_save_signature(path1, app.config['DEBUG_YOLO_FOLDER'])
        sig2, ok2 = extract_and_save_signature(path2, app.config['DEBUG_YOLO_FOLDER'])

        if not ok1 or not ok2:
            flash("Aucune signature détectée sur une ou deux images.", "warning")
            return redirect(url_for('index'))

        # Comparaison
        result = compare_signatures(sig1, sig2)

        # Sauvegarde temporaire des crops pour affichage
        crop1_path = os.path.join(app.config['UPLOAD_FOLDER'], f"crop1_{datetime.now().strftime('%H%M%S')}.jpg")
        crop2_path = os.path.join(app.config['UPLOAD_FOLDER'], f"crop2_{datetime.now().strftime('%H%M%S')}.jpg")
        cv2.imwrite(crop1_path, cv2.cvtColor(sig1, cv2.COLOR_RGB2BGR))
        cv2.imwrite(crop2_path, cv2.cvtColor(sig2, cv2.COLOR_RGB2BGR))

        flash(result, "result")
        return render_template('index.html',
                               crop1=os.path.basename(crop1_path),
                               crop2=os.path.basename(crop2_path))

# Servir les crops
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ========================================
# LANCER
# ========================================
if __name__ == '__main__':
    print("Démarrage Flask → http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)