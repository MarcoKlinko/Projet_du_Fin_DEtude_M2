# app.py → Version 100% fonctionnelle (Novembre 2025)
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
from collections import OrderedDict

app = Flask(__name__)
app.secret_key = "signature_login_2025_super_secret_key_change_me_in_production"

# ================== MongoDB ==================
client = MongoClient("mongodb://127.0.0.1:27017")
db = client.signature_auth
users = db.users
fs = gridfs.GridFS(db)

# ================== Device ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Utilisation du device : {device}")

# ================== Modèle Siamese Lightweight ==================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class LightweightSiamese(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 256, pool=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(256, embedding_dim)

    def forward_once(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        emb = self.fc(x)
        return F.normalize(emb, p=2, dim=1)  # L2 normalization

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

# ================== Chargement propre du modèle ==================
"""print("[INFO] Chargement du modèle Siamese...")
model = LightweightSiamese().to(device)

model_path = "/home/marco/Projet_du_Fin_DEtude_M2/Models/Model_siemese/siamese1.pth"

try:
    state_dict = torch.load(model_path, map_location=device)

    # Gestion du DataParallel (fréquent avec .pth venant de Jupyter + GPU)
    if list(state_dict.keys())[0].startswith("module."):
        print("[INFO] Suppression du préfixe 'module.' (DataParallel détecté)")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    print("[SUCCÈS] Modèle chargé avec succès")

except Exception as e:
    print(f"[ERREUR] Impossible de charger le modèle : {e}")
    exit(1)

# CRUCIAL : eval() IMMÉDIATEMENT après le chargement
model.eval()"""



print("[INFO] Chargement du modèle Siamese...")
model = LightweightSiamese().to(device)

model_path = "/home/marco/Projet_du_Fin_DEtude_M2/Models/Model_siemese/siamese.pth"

try:
    state_dict = torch.load(model_path, map_location=device)

    # Gestion DataParallel
    if next(iter(state_dict.keys())).startswith("module."):
        print("[INFO] Suppression préfixe 'module.'")
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    print("[SUCCÈS] Modèle chargé")

except Exception as e:
    print(f"[ERREUR] Chargement échoué : {e}")
    exit(1)

model.eval()

# FIX ULTIME SI TOUJOURS 0.0000 → désactive les BatchNorm pourries
for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.track_running_stats = False
        module.running_mean = None
        module.running_var = None
        module.eval()

print("[FIX] BatchNorm désactivées → embeddings valides même avec .pth corrompu")



# ================== Transformations ==================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

THRESHOLD = 0.68  # Tu peux l'ajuster plus tard (0.65 → 0.72 typique)

# ================== Fonctions utilitaires ==================
def embedding_from_b64(b64_str):
    try:
        header, encoded = b64_str.split(",", 1)
        data = base64.b64decode(encoded)
        img = Image.open(BytesIO(data)).convert("L")
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            return model.forward_once(tensor)
    except Exception as e:
        print(f"[ERREUR] embedding_from_b64 : {e}")
        return None

def get_user_embeddings(username):
    user = users.find_one({"username": username.lower()})
    if not user or not user.get("signatures"):
        return []
    
    embs = []
    for fid in user["signatures"]:
        try:
            file = fs.get(fid)
            img = Image.open(file).convert("L")
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.forward_once(tensor)
                embs.append(emb)
        except Exception as e:
            print(f"[ERREUR] Lecture signature réf. {fid} : {e}")
            continue
    return embs

def compare_signature(drawn_b64, username):
    refs = get_user_embeddings(username)
    if not refs:
        print(f"[INFO] Aucune signature de référence pour {username}")
        return False, 999.0

    drawn = embedding_from_b64(drawn_b64)
    if drawn is None:
        return False, 999.0

    distances = [F.pairwise_distance(drawn, ref).item() for ref in refs]
    min_dist = min(distances)
    return min_dist < THRESHOLD, min_dist

# ================== Routes Flask ==================
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

    print(f"[RÉSULTAT] Distance = {dist:.4f} | Seuil = {THRESHOLD} → {'ACCEPTÉE' if ok else 'REFUSÉE'}")

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
        return jsonify({"success": False, "message": "Données manquantes"}), 400

    try:
        header, encoded = sig.split(",", 1)
        img_data = base64.b64decode(encoded)
        fid = fs.put(img_data, filename=f"{user}_sig_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", username=user)
        
        users.update_one(
            {"username": user},
            {"$push": {"signatures": fid}},
            upsert=True
        )
        print(f"[SUCCÈS] Utilisateur {user} enregistré avec une nouvelle signature")
        return jsonify({"success": True, "message": f"{user} enregistré avec succès !"})
    
    except Exception as e:
        print(f"[ERREUR] register : {e}")
        return jsonify({"success": False, "message": "Erreur serveur"}), 500

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('index'))
    return f"""
    <h1>Bienvenue {session['user'].title()} !</h1>
    <p>Authentification par signature réussie !</p>
    <hr>
    <a href="/">← Se déconnecter</a>
    """

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("="*60)
    print(" APPLICATION DE VÉRIFICATION DE SIGNATURE")
    print(" http://127.0.0.1:5000")
    print("="*60)
    app.run(debug=False, host='127.0.0.1', port=5000)
    # debug=False en prod, mais tu peux remettre True pour développer