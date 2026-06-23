# 🏥 Prédiction des Coûts de Soins de Santé

> Projet de Machine Learning — ISEP2 | Année académique 2025-2026

Prédiction du coût annuel d'assurance santé d'un patient à partir de ses caractéristiques démographiques et médicales, à l'aide d'un modèle **XGBoost** déployé via une **API FastAPI** et un **dashboard Streamlit**.

---

## 📊 Performances du modèle

| Métrique      | Train  | Test   |
|---------------|--------|--------|
| R²            | 0.8766 | 0.9052 |
| RMSE          | —      | ~3 903 $ |
| MAE           | —      | ~2 392 $ |

**Variables les plus déterminantes :**

| Rang | Variable     | Importance |
|------|--------------|------------|
| 1    | `smoker`     | 82.1 %     |
| 2    | `bmi`        | 8.4 %      |
| 3    | `age`        | 4.7 %      |
| 4    | `children`   | 1.4 %      |
| 5    | `region`     | ~3.1 %     |

---

## 📁 Architecture du projet

```
Prédiction des coûts de soins de santé/
│
├── README.md                          ← Ce fichier
├── Prédiction des coûts médicaux.pdf  ← Rapport complet du projet
│
├── Notebooks/                         ← Phase d'exploration et de modélisation
│   ├── EDA_health_cost.ipynb          ← Analyse exploratoire des données (EDA)
│   └── Modelling final.ipynb          ← Comparaison des modèles & sélection finale
│
└── API Health cost prediction/        ← Application déployable (API + Dashboard)
    │
    ├── app.py                         ← API FastAPI (point d'entrée serveur)
    ├── export_model.py                ← Script d'entraînement & sérialisation du modèle
    ├── streamlit_app.py               ← Dashboard interactif Streamlit
    ├── test_model.py                  ← Tests de validation rapide du modèle
    ├── requirements.txt               ← Dépendances Python
    ├── Procfile                       ← Configuration de déploiement Render.com
    ├── run_local.bat                  ← Script de lancement local (Windows)
    ├── .gitignore                     ← Fichiers exclus du versionnement Git
    │
    ├── models/                        ← Artefacts du modèle entraîné
    │   ├── health_cost_model.joblib   ← Pipeline sérialisé (preprocessing + XGBoost)
    │   └── model_metadata.json        ← Métriques, paramètres et importances des features
    │
    └── Health cost prediction.ipynb   ← Notebook de référence (version API)
```

---

## 🔬 Description détaillée des fichiers

### `Notebooks/`

#### `EDA_health_cost.ipynb`
Analyse exploratoire complète du dataset *Medical Cost Personal Datasets* (`insurance.csv`, 1 338 lignes). Contient :
- Distribution des variables (`age`, `bmi`, `charges`, `smoker`, `region`, `sex`, `children`)
- Analyse des corrélations et de l'impact du tabac sur les charges
- Visualisations (histogrammes, boxplots, scatter plots, heatmaps)
- Détection des valeurs aberrantes et premières hypothèses métier

#### `Modelling final.ipynb`
Notebook de modélisation comparant plusieurs algorithmes de régression :
- **Baseline** : Régression linéaire, Ridge, Lasso
- **Ensembles** : Random Forest, Gradient Boosting
- **Finaliste** : XGBoost (meilleur RMSE CV = 4 590 $)
- GridSearchCV pour l'optimisation des hyperparamètres
- Feature engineering (voir section dédiée ci-dessous)
- Sélection et validation du modèle final

---

### `API Health cost prediction/`

#### `export_model.py` — Entraînement & sérialisation
Script autonome qui reproduit l'entraînement du modèle de bout en bout :

1. **Chargement des données** — cherche `insurance.csv` localement, tente un téléchargement depuis GitHub en fallback, et génère des données synthétiques en dernier recours.
2. **Feature engineering** — crée 4 variables supplémentaires :
   - `age_category` : catégorisation de l'âge (`Young` < 30, `Middle_aged` < 50, `Senior` ≥ 50)
   - `bmi_category` : catégorisation selon les seuils OMS (`Underweight`, `Normal`, `Overweight`, `Obese`)
   - `age_imc` : interaction numérique `age × bmi`
   - `parent_fumeur` : indicateur binaire — fumeur avec au moins un enfant (0/1)
3. **Pipeline sklearn** :
   - `OneHotEncoder` sur les variables catégorielles (`sex`, `smoker`, `region`, `age_category`, `bmi_category`)
   - `StandardScaler` sur les variables numériques (`age`, `bmi`, `children`, `age_imc`, `parent_fumeur`)
   - `XGBRegressor` avec les meilleurs hyperparamètres
4. **Export** :
   - `models/health_cost_model.joblib` — pipeline complet sérialisé via `joblib`
   - `models/model_metadata.json` — métriques, paramètres, importances des features

#### `app.py` — API FastAPI
Serveur REST exposant les prédictions du modèle. Détails :

- **Chargement au démarrage** : le pipeline `joblib` et les métadonnées JSON sont chargés une seule fois en mémoire.
- **Middleware CORS** activé (origines `*` — à restreindre en production).
- **Schémas Pydantic** avec validation stricte des entrées :
  - `age` : entier entre 18 et 100
  - `sex` : `"male"` ou `"female"` (Literal)
  - `bmi` : flottant entre 10.0 et 60.0
  - `children` : entier entre 0 et 10
  - `smoker` : `"yes"` ou `"no"` (Literal)
  - `region` : `"northeast"`, `"northwest"`, `"southeast"` ou `"southwest"` (Literal)

**Endpoints disponibles :**

| Méthode | Endpoint              | Description                                      |
|---------|-----------------------|--------------------------------------------------|
| GET     | `/`                   | Vérification que l'API est opérationnelle        |
| GET     | `/health`             | État de santé + métriques R² et RMSE du modèle  |
| GET     | `/model-info`         | Toutes les métadonnées du modèle (JSON complet)  |
| GET     | `/feature-importance` | Importance de chaque variable (classée)          |
| POST    | `/predict`            | Prédiction pour un seul patient                  |
| POST    | `/predict/batch`      | Prédictions pour jusqu'à 100 patients            |
| GET     | `/docs`               | Documentation interactive Swagger UI             |
| GET     | `/redoc`              | Documentation ReDoc                              |

**Format de réponse `/predict` :**
```json
{
  "predicted_charge": 18782.26,
  "predicted_charge_str": "$18,782.26",
  "model_used": "XGBoost Regressor",
  "input_data": { "age": 31, "sex": "male", "bmi": 25.0, ... }
}
```

#### `streamlit_app.py` — Dashboard interactif
Interface web connectée à l'API déployée sur Render (`https://heath-cost-prediction.onrender.com`). Fonctionnalités :
- Formulaire de saisie patient (sliders et listes déroulantes)
- Affichage de la prédiction en temps réel
- Visualisations Plotly des importances de variables et distributions
- Design personnalisé (CSS injected, typographies Google Fonts Inter + Space Grotesk, palette sombre)

#### `test_model.py` — Validation rapide
Script minimaliste (< 20 lignes) qui charge le pipeline `joblib` et effectue 3 prédictions de test pour s'assurer que le modèle est opérationnel avant déploiement :

| Patient | Age | Fumeur | IMC  | Enfants | Prédiction attendue |
|---------|-----|--------|------|---------|---------------------|
| 1       | 31  | Non    | 25.0 | 1       | ~18 782 $           |
| 2       | 60  | Non    | 20.9 | 3       | ~16 168 $           |
| 3       | 45  | **Oui**| 30.5 | 2       | Fortement majorée   |

#### `requirements.txt` — Dépendances
```
fastapi
uvicorn[standard]
xgboost
scikit-learn==1.6.1
pandas
numpy
joblib
pydantic
python-multipart
```
> ⚠️ `scikit-learn` est épinglé en `1.6.1` pour garantir la compatibilité de désérialisation du pipeline `.joblib`.

#### `Procfile` — Déploiement Render.com
```
web: uvicorn app:app --host 0.0.0.0 --port $PORT
```
Render injecte automatiquement la variable d'environnement `$PORT`.

#### `run_local.bat` — Lancement local Windows
Script batch qui :
1. Tue les processus éventuellement déjà actifs sur les ports `8000` et `8501`
2. Lance l'API FastAPI dans une fenêtre de terminal séparée (`port 8000`)
3. Lance le dashboard Streamlit dans une autre fenêtre (`port 8501`)

#### `models/health_cost_model.joblib`
Pipeline sklearn sérialisé contenant le preprocesseur (`ColumnTransformer`) et le modèle (`XGBRegressor`). Généré par `export_model.py`. **Doit être commité dans Git** pour que Render.com puisse le charger au démarrage (voir `.gitignore`).

#### `models/model_metadata.json`
Fichier JSON généré automatiquement par `export_model.py`, contenant :
- Nom du modèle, tâche, variable cible
- Liste complète des features (numériques + catégorielles)
- Description du feature engineering appliqué
- Meilleurs hyperparamètres XGBoost
- Métriques d'évaluation (R² train/test, RMSE, MAE)
- Tableau des importances de toutes les features encodées

#### `.gitignore`
Exclut les fichiers courants Python/Jupyter (`__pycache__`, `*.pyc`, `.ipynb_checkpoints/`), les environnements virtuels, et les fichiers IDE. **Le dossier `models/` n'est intentionnellement PAS ignoré** afin que le fichier `.joblib` soit commité et disponible sur Render.

---

## ⚙️ Hyperparamètres XGBoost (optimaux)

```python
XGBRegressor(
    colsample_bytree = 1.0,   # fraction des features utilisées par arbre
    learning_rate    = 0.05,  # taux d'apprentissage (shrinkage)
    max_depth        = 3,     # profondeur max des arbres (évite l'overfitting)
    n_estimators     = 100,   # nombre d'arbres
    subsample        = 0.8,   # fraction des échantillons par arbre (bagging)
    random_state     = 42,
)
```

---

## 🚀 Mise en route locale

### Prérequis
- Python 3.10+
- `insurance.csv` placé dans le dossier `API Health cost prediction/`

### Installation
```bash
cd "API Health cost prediction"
pip install -r requirements.txt
```

### Entraîner et exporter le modèle
```bash
python export_model.py
# Génère models/health_cost_model.joblib et models/model_metadata.json
```

### Lancer l'API
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
# Documentation : http://localhost:8000/docs
```

### Lancer le dashboard
```bash
streamlit run streamlit_app.py
# Interface : http://localhost:8501
```

### Lancement rapide (Windows uniquement)
```bash
run_local.bat
# Lance l'API et Streamlit dans deux terminaux séparés
```

### Tester le modèle
```bash
python test_model.py
# Vérifie que le modèle est chargé et produit des prédictions cohérentes
```

---

## 🌐 Déploiement sur Render.com

1. Créer un **Web Service** sur [render.com](https://render.com)
2. Connecter le repo GitHub
3. Configurer :
   - **Build Command** : `pip install -r requirements.txt`
   - **Start Command** : `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. S'assurer que `models/health_cost_model.joblib` est bien commité dans le repo
5. Cliquer sur **Deploy**

---

## 📝 Exemple d'appel API

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

### Prédiction batch (Python)
```python
import requests

payload = {
  "data": [
    {"age": 31, "sex": "male",   "bmi": 25.0, "children": 1, "smoker": "no",  "region": "southwest"},
    {"age": 60, "sex": "male",   "bmi": 20.9, "children": 3, "smoker": "no",  "region": "northeast"},
    {"age": 45, "sex": "female", "bmi": 30.5, "children": 2, "smoker": "yes", "region": "southeast"},
  ]
}
resp = requests.post("http://localhost:8000/predict/batch", json=payload)
print(resp.json())
# {'predictions': [18782.26, 16168.05, 42XXX.XX], 'count': 3}
```

---

## 👨‍💻 Équipe
Mame Diarra NDIAYE, Karel SODJINOUTI, Durel TEUMO

Projet ISEP2 — Machine Learning | Année académique 2025-2026
