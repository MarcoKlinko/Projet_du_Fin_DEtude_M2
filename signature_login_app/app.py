# app.py
from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import os
from pathlib import Path
import base64
from io import BytesIO

app = Flask(__name__)
app.secret_key = "change_me_in_production"

# ==================== CHARGEMENT DU MODÈLE ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LightweightSiamese(torch.nn.Module):
    # ... (colle ici exactement ta classe LightweightSiamese du notebook)
    # Je te remets la version compacte
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 256, pool=False),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = torch.nn.Linear(256, embedding_dim)

    def forward_once(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        emb = self.fc(x)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

# Nécessaire pour ConvBlock (si pas déjà défini)
class ConvBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [torch.nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                  torch.nn.BatchNorm2d(out_ch),
                  torch.nn.ReLU(inplace=True)]
        if pool: layers.append(torch.nn.MaxPool2d(2))
        self.block = torch.nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

# Chargement du modèle
model = LightweightSiamese().to(device)
model.load_state_dict(torch.load("model/siamese.pth", map_location=device))
model.eval()

# Transform identique à l'entraînement
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Dossier des signatures de référence
REF_FOLDER = "registered_signatures"
Path(REF_FOLDER).mkdir(exist_ok=True)

# Seuil optimisé (tu peux l'ajuster après tests)
THRESHOLD = 0.68

# ==================== FONCTIONS ====================
def embedding_from_base64(data_url):
    # data_url = "data:image/png;base64,iVBORw0KGgo..."
    header, encoded = data_url.split(",", 1)
    data = base64.b64decode(encoded)
    image = Image.open(BytesIO(data)).convert("L")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.forward_once(tensor)
    return emb

def compare_with_reference(drawn_signature_b64, reference_path):
    if not os.path.exists(reference_path):
        return False, 999.0
    
    ref_img = transform(Image.open(reference_path)).unsqueeze(0).to(device)
    drawn_emb = embedding_from_base64(drawn_signature_b64)
    
    with torch.no_grad():
        ref_emb = model.forward_once(ref_img)
        distance = F.pairwise_distance(drawn_emb, ref_emb).item()
    
    return distance < THRESHOLD, distance

# ==================== ROUTES ====================
@app.route('/')
def index():
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    signature_b64 = data.get('signature')
    username = data.get('username', 'default')  # tu peux ajouter un champ pseudo si tu veux

    ref_path = os.path.join(REF_FOLDER, f"{username}.png")
    
    success, dist = compare_with_reference(signature64, ref_path)
    
    if success:
        session['logged_in'] = True
        session['user'] = username
        return jsonify({"success": True, "message": "Connexion réussie !"})
    else:
        return jsonify({"success": False, "message": f"Signature refusée (distance={dist:.3f})"})

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('index'))
    return f"""
    <h1>Bienvenue {session['user']} !</h1>
    <p>Vous êtes connecté avec votre signature manuscrite.</p>
    <a href="/logout">Déconnexion</a>
    """

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# ==================== RUN ====================
if __name__ == '__main__':
    print("Application de login par signature démarrée !")
    print("http://127.0.0.1:5000")
    app.run(debug=True)