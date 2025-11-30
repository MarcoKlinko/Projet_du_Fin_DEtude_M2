# app.py
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import base64
from io import BytesIO
from pymongo import MongoClient
import gridfs
from datetime import datetime

app = Flask(__name__)
app.secret_key = "signature_login_2025"

# ================== MongoDB ==================
client = MongoClient("mongodb://127.0.0.1:27017")
db = client.signature_auth
users = db.users
fs = gridfs.GridFS(db)

# ================== Modèle Siamese ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                  nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        if pool: layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

class LightweightSiamese(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(1, 32), ConvBlock(32, 64),
            ConvBlock(64, 128), ConvBlock(128, 256),
            ConvBlock(256, 256, pool=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(256, embedding_dim)
    def forward_once(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        emb = self.fc(x)
        return F.normalize(emb, p=2, dim=1)
    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

model = LightweightSiamese().to(device)
model.load_state_dict(torch.load("/home/marco/Projet_du_Fin_DEtude_M2/Models/Model_siemese/siamese_jupyter.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  
])

THRESHOLD = 0.68

# ================== Fonctions ==================
def embedding_from_b64(b64_str):
    header, encoded = b64_str.split(",", 1)
    data = base64.b64decode(encoded)
    img = Image.open(BytesIO(data)).convert("L")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.forward_once(tensor)

def get_user_embeddings(username):
    user = users.find_one({"username": username.lower()})
    if not user or not user.get("signatures"): return []
    embs = []
    for fid in user["signatures"]:
        try:
            file = fs.get(fid)
            img = Image.open(file).convert("L")
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                embs.append(model.forward_once(tensor))
        except: pass
    return embs

def compare_signature(drawn_b64, username):
    refs = get_user_embeddings(username)
    if not refs:
        return False, 999.0
    drawn = embedding_from_b64(drawn_b64)
    distances = [F.pairwise_distance(drawn, ref).item() for ref in refs]
    return min(distances) < THRESHOLD, min(distances)

# ================== Routes ==================
@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    sig = data.get('signature')
    user = data.get('username', '').strip().lower()
    
    print(f"\n[DEBUG] Tentative de connexion pour : {user}")
    
    if not sig or not user:
        return jsonify({"success": False, "message": "Données manquantes"}), 400

    ok, dist = compare_signature(sig, user)
    
    # DEBUG ULTRA-IMPORTANT
    print(f"[RÉSULTAT] Distance calculée = {dist:.4f} | Seuil = {THRESHOLD}")
    print(f"[RÉSULTAT] → {'ACCEPTÉE' if ok else 'REFUSÉE'}")

    if ok:
        session['user'] = user
        return jsonify({"success": True, "distance": round(dist, 4)})
    else:
        return jsonify({
            "success": False,
            "distance": round(dist, 4),
            "message": "Signature refusée"
        }), 401

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    sig = data.get('signature')
    user = data.get('username', '').strip().lower()
    if not sig or not user:
        return jsonify({"success": False, "message": "Erreur"}), 400
    header, encoded = sig.split(",", 1)
    img_data = base64.b64decode(encoded)
    fid = fs.put(img_data, filename=f"{user}_sig.png", username=user)
    users.update_one(
        {"username": user},
        {"$push": {"signatures": fid}},
        upsert=True
    )
    return jsonify({"success": True, "message": f"{user} enregistré !"})

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('index'))
    return f"""
    <h1>Bienvenue {session['user']} !</h1>
    <p>Connexion par signature réussie !</p>
    <a href="/">← Retour</a>
    """

if __name__ == '__main__':
    print("Application prête ! → http://127.0.0.1:5000")
    app.run(debug=True, port=5000)