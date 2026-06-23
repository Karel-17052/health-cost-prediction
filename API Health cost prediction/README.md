# Health Cost Prediction API

## 📋 Description
API de prédiction des coûts d'assurance santé basée sur un modèle **XGBoost** entraîné sur le dataset *Medical Cost Personal Datasets*.

**Performances du modèle :**
| Métrique     | Valeur   |
|--------------|----------|
| R² Test      | 0.9052   |
| RMSE Test    | ~3 884 $ |
| MAE Test     | ~2 353 $ |
| Best CV RMSE | -4 590   |

**Variables les plus importantes :**
1. `smoker` — 81.8% d'importance 🚬
2. `bmi` — 8.7%
3. `age` — 4.7%

---

## 📁 Structure du projet

```
Health cost prediction/
│
├── export_model.py        ← Script d'entraînement & export du modèle
├── app.py                 ← API FastAPI
├── streamlit_app.py       ← Dashboard Streamlit
├── requirements.txt       ← Dépendances Python
├── Procfile               ← Config déploiement Render
├── README.md              ← Ce fichier
│
└── models/                ← Créé automatiquement par export_model.py
    ├── health_cost_model.joblib
    └── model_metadata.json
```

---

## 🚀 Mise en route (local)

### Étape 1 — Installer les dépendances
```bash
pip install -r requirements.txt
```

### Étape 2 — Entraîner et exporter le modèle
```bash
python export_model.py
```
> ⚠️ Placez votre fichier `insurance.csv` dans ce dossier avant de lancer.
> Le script génère automatiquement `models/health_cost_model.joblib` et `models/model_metadata.json`.

### Étape 3 — Lancer l'API
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
Accédez à la documentation interactive : http://localhost:8000/docs

### Étape 4 — Lancer le Dashboard Streamlit
```bash
streamlit run streamlit_app.py
```

---

## 🌐 Déploiement sur Render.com

### Prérequis
- Compte Render.com
- Le dossier `models/` avec le fichier `.joblib` doit être **commit dans le repo Git**
  (ou utiliser un volume persistant sur Render)

### Étapes
1. Créez un nouveau **Web Service** sur [render.com](https://render.com)
2. Connectez votre repo GitHub
3. Configurez :
   - **Build Command** : `pip install -r requirements.txt`
   - **Start Command** : `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Cliquez sur **Deploy**

### Variables d'environnement Render (optionnelles)
| Variable    | Description                |
|-------------|----------------------------|
| `PORT`      | Port (géré automatiquement)|

---

## 📡 Endpoints de l'API

| Méthode | Endpoint              | Description                          |
|---------|-----------------------|--------------------------------------|
| GET     | `/`                   | Status de l'API                      |
| GET     | `/health`             | Santé + métriques du modèle          |
| GET     | `/model-info`         | Infos complètes sur le modèle        |
| GET     | `/feature-importance` | Importance des variables             |
| POST    | `/predict`            | Prédiction pour un patient           |
| POST    | `/predict/batch`      | Prédictions pour plusieurs patients  |
| GET     | `/docs`               | Documentation Swagger UI             |
| GET     | `/redoc`              | Documentation ReDoc                  |

---

## 📝 Exemple d'utilisation

### Prédiction simple (curl)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 31,
    "sex": "male",
    "bmi": 25.0,
    "children": 1,
    "smoker": "no",
    "region": "southwest"
  }'
```

**Réponse :**
```json
{
  "predicted_charge": 18782.26,
  "predicted_charge_str": "$18,782.26",
  "model_used": "XGBoost Regressor",
  "input_data": { ... }
}
```

### Prédiction batch (Python)
```python
import requests

payload = {
  "data": [
    {"age": 31, "sex": "male", "bmi": 25.0, "children": 1, "smoker": "no", "region": "southwest"},
    {"age": 60, "sex": "male", "bmi": 20.9, "children": 3, "smoker": "no", "region": "northeast"},
  ]
}
resp = requests.post("http://localhost:8000/predict/batch", json=payload)
print(resp.json())
# {'predictions': [18782.26, 16168.05], 'count': 2}
```

---

## ⚙️ Meilleurs paramètres XGBoost

```python
XGBRegressor(
    colsample_bytree = 1.0,
    learning_rate    = 0.05,
    max_depth        = 3,
    n_estimators     = 100,
    subsample        = 0.8,
)
```

---

## 👨‍💻 Équipe
Projet ISEP2 — Machine Learning  
Année académique 2025-2026
