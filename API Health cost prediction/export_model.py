# -*- coding: utf-8 -*-
"""
export_model.py
---------------
Script d'entraînement et d'export du meilleur modèle pour l'API Health Cost Prediction.
- Modèle : XGBoost (meilleur CV RMSE = 4590.74)
- Meilleurs paramètres identifiés dans le notebook
- Feature engineering : age_category, bmi_category, age_imc, parent_fumeur
- Export via joblib
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ─────────────────────────────────────────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Chargement des donnees...")

data_file = None
search_dirs = ["."] + [os.path.join("..", *[".."] * i) for i in range(3)]
csv_candidates = ["insurance.csv", "Insurance.csv", "health_cost.csv"]

for search_dir in search_dirs:
    for candidate in csv_candidates:
        path = os.path.join(search_dir, candidate)
        if os.path.exists(path):
            data_file = path
            break
    if data_file is None and os.path.isdir(search_dir):
        for fname in os.listdir(search_dir):
            if fname.endswith(".csv") and any(k in fname.lower() for k in ["insurance", "health"]):
                data_file = os.path.join(search_dir, fname)
                break
    if data_file:
        break

if data_file is None:
    print("[WARN] Fichier CSV non trouve localement. Tentative de telechargement...")
    URLS = [
        "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv",
        "https://raw.githubusercontent.com/aniketmaurya/machine-learning/master/datasets/insurance.csv",
    ]
    df = None
    for url in URLS:
        try:
            df = pd.read_csv(url)
            print(f"[OK] Donnees telechargees depuis : {url}")
            break
        except Exception:
            continue

    if df is None:
        print("[WARN] Telechargement impossible. Generation de donnees synthetiques...")
        np.random.seed(42)
        n = 1338
        age      = np.random.randint(18, 65, n)
        sex      = np.random.choice(["male", "female"], n)
        bmi      = np.round(np.random.normal(30.7, 6.1, n).clip(15, 55), 1)
        children = np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.43, 0.24, 0.18, 0.10, 0.03, 0.02])
        smoker   = np.random.choice(["yes", "no"], n, p=[0.205, 0.795])
        region   = np.random.choice(["northeast", "northwest", "southeast", "southwest"], n)
        charges  = (age * 256 + bmi * 332 + children * 475
                    + (smoker == "yes") * 23850
                    + np.random.normal(0, 3000, n)).clip(1122, 63770)
        df = pd.DataFrame({"age": age, "sex": sex, "bmi": bmi, "children": children,
                           "smoker": smoker, "region": region, "charges": charges})
        print(f"[OK] Donnees synthetiques generees : {df.shape}")
else:
    df = pd.read_csv(data_file)
    print(f"[OK] Donnees chargees depuis : {data_file}")

print(f"   Shape : {df.shape}")
print(f"   Colonnes : {list(df.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[INFO] Feature engineering...")

def categorize_age(age):
    """Regroupe l'âge en classes : Young / Middle_aged / Senior."""
    if age < 30:
        return 'Young'
    elif age < 50:
        return 'Middle_aged'
    else:
        return 'Senior'

def categorize_bmi(bmi):
    """Regroupe le BMI selon les seuils conventionnels de l'OMS."""
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

# Variables catégorielles issues du feature engineering
df['age_category']  = df['age'].apply(categorize_age)
df['bmi_category']  = df['bmi'].apply(categorize_bmi)

# Variables numériques issues du feature engineering
df['age_imc']       = df['age'] * df['bmi']                                   # interaction âge × IMC
df['parent_fumeur'] = ((df['smoker'] == 'yes') & (df['children'] > 0)).astype(int)

print(f"   age_category   : {df['age_category'].value_counts().to_dict()}")
print(f"   bmi_category   : {df['bmi_category'].value_counts().to_dict()}")
print(f"   parent_fumeur  : {df['parent_fumeur'].value_counts().to_dict()}")
print(f"   age_imc (mean) : {df['age_imc'].mean():.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PRÉPARATION DES DONNÉES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[INFO] Preparation des donnees...")

TARGET = "charges"

CATEGORICAL_FEATURES = ["sex", "smoker", "region", "age_category", "bmi_category"]
NUMERICAL_FEATURES   = ["age", "bmi", "children", "age_imc", "parent_fumeur"]
ALL_FEATURES         = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

X = df[ALL_FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
print(f"   Train: {X_train.shape} | Test: {X_test.shape}")
print(f"   Features numeriques   : {NUMERICAL_FEATURES}")
print(f"   Features categorielle : {CATEGORICAL_FEATURES}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PIPELINE — MEILLEUR MODÈLE (XGBoost)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[INFO] Construction du pipeline XGBoost (meilleurs parametres)...")

preprocessor = ColumnTransformer(transformers=[
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ("numeric", StandardScaler(),                                            NUMERICAL_FEATURES),
])

best_pipeline = Pipeline(steps=[
    ("encoding", preprocessor),
    ("xgboost", XGBRegressor(
        colsample_bytree = 1.0,
        learning_rate    = 0.05,
        max_depth        = 3,
        n_estimators     = 100,
        subsample        = 0.8,
        random_state     = 42,
        verbosity        = 0,
    )),
])

# ─────────────────────────────────────────────────────────────────────────────
# 5. ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Entrainement en cours...")
best_pipeline.fit(X_train, y_train)
print("[OK] Entrainement termine.")

# ─────────────────────────────────────────────────────────────────────────────
# 6. ÉVALUATION
# ─────────────────────────────────────────────────────────────────────────────
y_pred_train = best_pipeline.predict(X_train)
y_pred_test  = best_pipeline.predict(X_test)

print("\n[INFO] Metriques d'evaluation :")
print(f"   R²  Train : {r2_score(y_train, y_pred_train):.4f}")
print(f"   R²  Test  : {r2_score(y_test, y_pred_test):.4f}")
print(f"   RMSE Train: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.2f}")
print(f"   RMSE Test : {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f}")
print(f"   MAE  Test : {mean_absolute_error(y_test, y_pred_test):.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. FEATURE IMPORTANCES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[INFO] Feature Importances :")
encoder       = best_pipeline.named_steps["encoding"]
feature_names = encoder.get_feature_names_out()
importances   = best_pipeline.named_steps["xgboost"].feature_importances_

fi_df = (
    pd.DataFrame({"Feature": feature_names, "Importance": importances})
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)
print(fi_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 8. EXPORT DU MODÈLE
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "health_cost_model.joblib")
joblib.dump(best_pipeline, model_path)
print(f"\n[OK] Modele exporte : {model_path}")

# Sauvegarder les métadonnées du modèle
metadata = {
    "model_name"          : "XGBoost Regressor",
    "task"                : "regression",
    "target"              : TARGET,
    "features"            : ALL_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "numerical_features"  : NUMERICAL_FEATURES,
    "feature_engineering" : {
        "age_category" : "Regroupement de l'âge en classes (Young / Middle_aged / Senior)",
        "bmi_category" : "Regroupement du BMI selon seuils OMS (Underweight / Normal / Overweight / Obese)",
        "age_imc"      : "Interaction numérique age × bmi",
        "parent_fumeur": "Indicateur binaire : fumeur avec au moins un enfant",
    },
    "best_params"         : {
        "colsample_bytree": 1.0,
        "learning_rate"   : 0.05,
        "max_depth"       : 3,
        "n_estimators"    : 100,
        "subsample"       : 0.8,
    },
    "metrics": {
        "r2_train" : round(r2_score(y_train, y_pred_train), 4),
        "r2_test"  : round(r2_score(y_test, y_pred_test), 4),
        "rmse_test": round(float(np.sqrt(mean_squared_error(y_test, y_pred_test))), 2),
        "mae_test" : round(float(mean_absolute_error(y_test, y_pred_test)), 2),
    },
    "feature_importances": fi_df.to_dict(orient="records"),
    "model_path"         : model_path,
}

metadata_path = os.path.join("models", "model_metadata.json")
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)

print(f"[OK] Metadonnees exportees : {metadata_path}")
print("\n[OK] Export termine avec succes !")