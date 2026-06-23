"""
app.py
------
API FastAPI pour la prédiction des coûts d'assurance santé.
Modèle : XGBoost (R² test = 0.905, RMSE = 3884$)

Déploiement : Render.com
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np
import pandas as pd
import json
import os
from typing import Literal

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH    = os.path.join("models", "health_cost_model.joblib")
METADATA_PATH = os.path.join("models", "model_metadata.json")

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Modèle chargé depuis {MODEL_PATH}")
except FileNotFoundError:
    raise RuntimeError(
        f"Modèle introuvable : {MODEL_PATH}. "
        "Veuillez exécuter export_model.py en premier."
    )

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION FASTAPI
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "🏥 Health Cost Prediction API",
    description = (
        "API de prédiction des coûts d'assurance santé basée sur un modèle XGBoost. "
        "Fournissez les caractéristiques du patient pour obtenir une estimation du coût annuel."
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# CORS — autoriser toutes les origines (à restreindre en production)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# SCHÉMAS PYDANTIC
# ─────────────────────────────────────────────────────────────────────────────
class PredictionInput(BaseModel):
    age     : int   = Field(..., ge=18, le=100,  example=31,         description="Âge de la personne (18-100 ans)")
    sex     : Literal["male", "female"]          = Field(..., example="male",   description="Sexe")
    bmi     : float = Field(..., ge=10.0, le=60.0, example=25.0,     description="Indice de Masse Corporelle (IMC)")
    children: int   = Field(..., ge=0, le=10,    example=1,          description="Nombre d'enfants couverts")
    smoker  : Literal["yes", "no"]               = Field(..., example="no",     description="Fumeur (yes/no)")
    region  : Literal["northeast", "northwest", "southeast", "southwest"] = Field(
        ..., example="southwest", description="Région d'habitation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "age"     : 31,
                "sex"     : "male",
                "bmi"     : 25.0,
                "children": 1,
                "smoker"  : "no",
                "region"  : "southwest",
            }
        }


class PredictionOutput(BaseModel):
    predicted_charge   : float
    predicted_charge_str: str
    model_used         : str
    input_data         : dict


class BatchInput(BaseModel):
    data: list[PredictionInput]


class BatchOutput(BaseModel):
    predictions: list[float]
    count      : int

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Point d'entrée — vérifie que l'API est opérationnelle."""
    return {
        "message": "🏥 Health Cost Prediction API is running!",
        "version": "1.0.0",
        "docs"   : "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Vérification de l'état de l'API et du modèle."""
    return {
        "status"    : "healthy",
        "model"     : metadata.get("model_name"),
        "r2_test"   : metadata.get("metrics", {}).get("r2_test"),
        "rmse_test" : metadata.get("metrics", {}).get("rmse_test"),
    }


@app.get("/model-info", tags=["Model"])
def model_info():
    """Informations complètes sur le modèle entraîné."""
    return metadata


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(input_data: PredictionInput):
    """
    Prédit le coût annuel d'assurance santé pour un patient.
    
    **Paramètres d'entrée :**
    - **age** : Âge du patient (18-100)
    - **sex** : Sexe (male / female)
    - **bmi** : Indice de Masse Corporelle
    - **children** : Nombre d'enfants couverts par l'assurance
    - **smoker** : Fumeur (yes / no) — variable la plus influente !
    - **region** : Région d'habitation US (northeast / northwest / southeast / southwest)
    
    **Retourne :**
    - **predicted_charge** : Montant prédit en dollars US
    """
    try:
        df = pd.DataFrame([input_data.model_dump()])
        prediction = float(model.predict(df)[0])
        prediction = max(0, prediction)  # Pas de charge négative

        return PredictionOutput(
            predicted_charge    = round(prediction, 2),
            predicted_charge_str= f"${prediction:,.2f}",
            model_used          = metadata.get("model_name", "XGBoost"),
            input_data          = input_data.model_dump(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")


@app.post("/predict/batch", response_model=BatchOutput, tags=["Prediction"])
def predict_batch(batch: BatchInput):
    """
    Prédit les coûts pour plusieurs patients en une seule requête.
    Maximum 100 patients par requête.
    """
    if len(batch.data) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 patients par requête batch."
        )
    try:
        df          = pd.DataFrame([item.model_dump() for item in batch.data])
        predictions = model.predict(df).tolist()
        predictions = [round(max(0, p), 2) for p in predictions]

        return BatchOutput(
            predictions = predictions,
            count       = len(predictions),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur batch : {str(e)}")


@app.get("/feature-importance", tags=["Model"])
def feature_importance():
    """Retourne l'importance des variables dans le modèle XGBoost."""
    return {
        "feature_importances": metadata.get("feature_importances", []),
        "note": "smoker est la variable la plus déterminante (81.8% d'importance)"
    }
