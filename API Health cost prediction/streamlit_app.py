# -*- coding: utf-8 -*-
"""
streamlit_app.py
----------------
Dashboard Streamlit pour visualiser et tester l'API Health Cost Prediction.
Connexion à l'API déployée sur Render.com.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Health Cost Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "https://heath-cost-prediction.onrender.com"

# ─────────────────────────────────────────────────────────────────────────────
# CSS CUSTOM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Hero header ── */
    .hero-wrap {
        background: linear-gradient(135deg, #0f2027 0%, #1a3a4a 50%, #0d3b47 100%);
        border-radius: 16px;
        padding: 2.4rem 2rem 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hero-wrap::before {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse at 70% 30%, rgba(0,200,180,0.12) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin: 0 0 0.4rem;
    }
    .hero-sub {
        font-size: 1rem;
        color: #7ecec4;
        margin: 0;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(0,200,180,0.15);
        border: 1px solid rgba(0,200,180,0.35);
        color: #00c8b4;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        margin-bottom: 0.9rem;
        text-transform: uppercase;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0f1e26 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #d0e8e4 !important;
    }
    .sidebar-logo {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: #00c8b4 !important;
        margin-bottom: 0.3rem;
    }
    .sidebar-section {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #4a7a74 !important;
        margin: 1.2rem 0 0.4rem;
    }
    .status-pill {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .status-ok   { background: rgba(46,204,113,0.15); color: #2ecc71 !important; border: 1px solid #2ecc71; }
    .status-warn { background: rgba(231,76,60,0.15);  color: #e74c3c !important; border: 1px solid #e74c3c; }

    /* ── Cards ── */
    .kpi-card {
        background: #111e27;
        border: 1px solid #1e3440;
        border-radius: 12px;
        padding: 1.1rem 1rem;
        text-align: center;
    }
    .kpi-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #4a7a74;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #00c8b4;
    }

    /* ── Prediction result ── */
    .pred-box {
        background: linear-gradient(135deg, #0d2e28 0%, #0f3a32 100%);
        border: 1px solid rgba(0,200,180,0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .pred-amount {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.2rem;
        font-weight: 700;
        color: #00c8b4;
        line-height: 1;
    }
    .pred-caption {
        font-size: 0.9rem;
        color: #7ecec4;
        margin-top: 0.4rem;
    }

    /* ── Form labels ── */
    .form-section {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #4a7a74;
        margin: 1rem 0 0.2rem;
    }

    /* ── Info chip ── */
    .info-chip {
        display: inline-block;
        background: rgba(0,200,180,0.08);
        border: 1px solid rgba(0,200,180,0.2);
        color: #7ecec4;
        font-size: 0.78rem;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        margin: 0.15rem;
    }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #0c1a22;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px;
        color: #7ecec4;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #1a3a4a !important;
        color: #00c8b4 !important;
    }

    /* ── General Streamlit overrides ── */
    .stButton > button {
        background: linear-gradient(135deg, #00c8b4, #00a89a);
        color: #0f1e26;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-size: 0.95rem;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }
    div[data-testid="stMetricValue"] { color: #00c8b4; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-badge">XGBoost · Feature Engineering</div>
  <div class="hero-title">🏥 Health Cost Prediction</div>
  <p class="hero-sub">Estimation des coûts d'assurance santé — avec catégories d'âge, d'IMC et interactions</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — STATUT API + VARIABLES DÉRIVÉES
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">⚕️ HealthCost AI</div>', unsafe_allow_html=True)
    st.caption("Modèle XGBoost avec feature engineering")

    st.markdown('<div class="sidebar-section">Statut API</div>', unsafe_allow_html=True)
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        if resp.status_code == 200:
            info = resp.json()
            st.markdown('<span class="status-pill status-ok">● Connectée</span>', unsafe_allow_html=True)
            st.markdown("")
            c1, c2 = st.columns(2)
            c1.metric("R² Test",   f"{info.get('r2_test', 0):.4f}")
            c2.metric("RMSE Test", f"${info.get('rmse_test', 0):,.0f}")
        else:
            st.markdown('<span class="status-pill status-warn">● Non disponible</span>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<span class="status-pill status-warn">● Non joignable</span>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Variables dérivées actives</div>', unsafe_allow_html=True)
    st.markdown("""
    <div>
      <span class="info-chip">🎂 age_category</span>
      <span class="info-chip">⚖️ bmi_category</span>
      <span class="info-chip">✖️ age_imc</span>
      <span class="info-chip">🚬 parent_fumeur</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Seuils OMS (BMI)</div>', unsafe_allow_html=True)
    st.markdown("""
    | Catégorie    | IMC       |
    |--------------|-----------|
    | Underweight  | < 18.5    |
    | Normal       | 18.5–24.9 |
    | Overweight   | 25–29.9   |
    | Obese        | ≥ 30      |
    """)

    st.markdown('<div class="sidebar-section">Seuils âge</div>', unsafe_allow_html=True)
    st.markdown("""
    | Catégorie    | Âge     |
    |--------------|---------|
    | Young        | < 30    |
    | Middle-aged  | 30–49   |
    | Senior       | ≥ 50    |
    """)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER : calcul local des variables dérivées (affichage)
# ─────────────────────────────────────────────────────────────────────────────
def get_age_category(age):
    if age < 30:   return "Young", "🟢"
    if age < 50:   return "Middle-aged", "🟡"
    return "Senior", "🔴"

def get_bmi_category(bmi):
    if bmi < 18.5: return "Underweight", "🔵"
    if bmi < 25:   return "Normal", "🟢"
    if bmi < 30:   return "Overweight", "🟡"
    return "Obese", "🔴"

# ─────────────────────────────────────────────────────────────────────────────
# ONGLETS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯  Prédiction individuelle", "📊  Prédiction batch", "🔬  Infos modèle"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRÉDICTION INDIVIDUELLE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("#### Profil du patient")

        st.markdown('<div class="form-section">Données démographiques</div>', unsafe_allow_html=True)
        age      = st.slider("Âge", min_value=18, max_value=100, value=35)
        sex      = st.radio("Sexe", ["male", "female"], horizontal=True,
                             format_func=lambda x: "👨 Homme" if x == "male" else "👩 Femme")

        st.markdown('<div class="form-section">Santé & Morphologie</div>', unsafe_allow_html=True)
        bmi      = st.number_input("IMC (BMI)", min_value=10.0, max_value=60.0, value=25.0, step=0.1,
                                   help="Indice de Masse Corporelle")

        st.markdown('<div class="form-section">Mode de vie & Situation</div>', unsafe_allow_html=True)
        smoker   = st.radio("Fumeur ?", ["no", "yes"], horizontal=True,
                             format_func=lambda x: "🚭 Non-fumeur" if x == "no" else "🚬 Fumeur")
        children = st.slider("Nombre d'enfants à charge", min_value=0, max_value=10, value=0)
        region   = st.selectbox("Région", ["northeast", "northwest", "southeast", "southwest"],
                                 format_func=lambda x: {
                                     "northeast": "🗺️ Nord-Est",
                                     "northwest": "🗺️ Nord-Ouest",
                                     "southeast": "🗺️ Sud-Est",
                                     "southwest": "🗺️ Sud-Ouest",
                                 }[x])

        # Preview des variables dérivées
        age_cat, age_icon = get_age_category(age)
        bmi_cat, bmi_icon = get_bmi_category(bmi)
        age_imc           = round(age * bmi, 1)
        parent_fumeur     = 1 if smoker == "yes" and children > 0 else 0

        st.markdown('<div class="form-section">Variables calculées automatiquement</div>', unsafe_allow_html=True)
        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.markdown(f'<div class="kpi-card"><div class="kpi-label">Âge</div><div class="kpi-value" style="font-size:1rem">{age_icon} {age_cat}</div></div>', unsafe_allow_html=True)
        dc2.markdown(f'<div class="kpi-card"><div class="kpi-label">IMC</div><div class="kpi-value" style="font-size:1rem">{bmi_icon} {bmi_cat}</div></div>', unsafe_allow_html=True)
        dc3.markdown(f'<div class="kpi-card"><div class="kpi-label">Âge × IMC</div><div class="kpi-value" style="font-size:1rem">{age_imc}</div></div>', unsafe_allow_html=True)
        dc4.markdown(f'<div class="kpi-card"><div class="kpi-label">Parent fumeur</div><div class="kpi-value" style="font-size:1rem">{"✅ Oui" if parent_fumeur else "❌ Non"}</div></div>', unsafe_allow_html=True)

        st.markdown("")
        predict_btn = st.button("🔮 Estimer le coût annuel", use_container_width=True, type="primary")

    with col_result:
        st.markdown("#### Résultat de la prédiction")

        if predict_btn:
            payload = {
                "age"     : age,
                "sex"     : sex,
                "bmi"     : bmi,
                "children": children,
                "smoker"  : smoker,
                "region"  : region,
            }
            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                if resp.status_code == 200:
                    result = resp.json()
                    pred   = result["predicted_charge"]

                    # Résultat principal
                    st.markdown(f"""
                    <div class="pred-box">
                        <div class="pred-amount">{result["predicted_charge_str"]}</div>
                        <div class="pred-caption">Coût annuel estimé · assurance santé</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode  = "gauge+number+delta",
                        value = pred,
                        delta = {"reference": 13270, "valueformat": ",.0f", "prefix": "$"},
                        title = {"text": "vs. moyenne nationale ($13 270)", "font": {"size": 13}},
                        number= {"prefix": "$", "valueformat": ",.0f"},
                        gauge = {
                            "axis"  : {"range": [0, 65000], "tickformat": "$,.0f"},
                            "bar"   : {"color": "#00c8b4", "thickness": 0.25},
                            "bgcolor": "#0f1e26",
                            "bordercolor": "#1e3440",
                            "steps": [
                                {"range": [0, 10000],     "color": "#0d2e28"},
                                {"range": [10000, 30000], "color": "#1a3525"},
                                {"range": [30000, 65000], "color": "#2a1e1e"},
                            ],
                            "threshold": {
                                "line"     : {"color": "#e74c3c", "width": 3},
                                "thickness": 0.75,
                                "value"    : 30000,
                            },
                        }
                    ))
                    fig_gauge.update_layout(
                        height=260,
                        paper_bgcolor="#0c1a22",
                        font={"color": "#d0e8e4"},
                        margin={"t": 40, "b": 10, "l": 20, "r": 20},
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    # Facteurs de risque
                    risk_factors = []
                    if smoker == "yes":
                        risk_factors.append(("🚬 Fumeur", "Facteur de coût majeur (+$23 000 en moyenne)", "high"))
                    if bmi >= 30:
                        risk_factors.append(("⚖️ Obésité", f"BMI = {bmi} — catégorie Obese (OMS)", "medium"))
                    if age >= 50:
                        risk_factors.append(("🎂 Senior", f"Âge {age} ans — coûts plus élevés", "medium"))
                    if smoker == "yes" and children > 0:
                        risk_factors.append(("👨‍👧 Parent fumeur", "Combinaison à risque élevé", "high"))
                    if not risk_factors:
                        risk_factors.append(("✅ Profil faible risque", "Aucun facteur aggravant majeur détecté", "low"))

                    st.markdown("**Facteurs identifiés**")
                    for label, desc, level in risk_factors:
                        color = {"high": "#e74c3c", "medium": "#f39c12", "low": "#2ecc71"}[level]
                        st.markdown(
                            f'<div style="border-left:3px solid {color};padding:0.4rem 0.8rem;'
                            f'background:rgba(0,0,0,0.2);border-radius:0 6px 6px 0;margin:0.3rem 0;">'
                            f'<b style="color:{color}">{label}</b><br>'
                            f'<span style="font-size:0.82rem;color:#7ecec4">{desc}</span></div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.error(f"Erreur API : {resp.json()}")
            except Exception as e:
                st.error(f"❌ Connexion échouée : {e}")
        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;color:#4a7a74;">
              <div style="font-size:3rem;margin-bottom:1rem">🔮</div>
              <div style="font-size:1rem;font-weight:500">Remplissez le formulaire<br>et cliquez sur <b style="color:#00c8b4">Estimer</b></div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PRÉDICTION BATCH
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    c_left, c_right = st.columns([1, 1], gap="large")

    with c_left:
        st.markdown("#### Importer un fichier CSV")
        st.markdown("""
        Colonnes attendues (les variables dérivées sont calculées automatiquement) :

        | Colonne    | Type    | Exemple      |
        |------------|---------|--------------|
        | `age`      | entier  | 31           |
        | `sex`      | texte   | male / female|
        | `bmi`      | décimal | 25.0         |
        | `children` | entier  | 1            |
        | `smoker`   | texte   | yes / no     |
        | `region`   | texte   | southwest    |
        """)

        uploaded_file = st.file_uploader("Choisir un fichier CSV", type=["csv"])

    with c_right:
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            st.markdown(f"#### Aperçu — {len(df_upload)} ligne(s) détectée(s)")
            st.dataframe(df_upload.head(8), use_container_width=True)

            if st.button("🚀 Lancer les prédictions", type="primary", use_container_width=True):
                with st.spinner("Calcul en cours..."):
                    records = df_upload.to_dict(orient="records")
                    payload = {"data": records}
                    try:
                        resp = requests.post(f"{API_URL}/predict/batch", json=payload, timeout=30)
                        if resp.status_code == 200:
                            result     = resp.json()
                            df_upload["Coût estimé ($)"] = result["predictions"]

                            st.success(f"✅ {result['count']} prédictions effectuées")

                            # KPIs
                            k1, k2, k3 = st.columns(3)
                            k1.markdown(f'<div class="kpi-card"><div class="kpi-label">Moyenne</div><div class="kpi-value">${df_upload["Coût estimé ($)"].mean():,.0f}</div></div>', unsafe_allow_html=True)
                            k2.markdown(f'<div class="kpi-card"><div class="kpi-label">Maximum</div><div class="kpi-value">${df_upload["Coût estimé ($)"].max():,.0f}</div></div>', unsafe_allow_html=True)
                            k3.markdown(f'<div class="kpi-card"><div class="kpi-label">Médiane</div><div class="kpi-value">${df_upload["Coût estimé ($)"].median():,.0f}</div></div>', unsafe_allow_html=True)

                            st.markdown("")
                            st.dataframe(df_upload, use_container_width=True)

                            # Distribution
                            fig_dist = px.histogram(
                                df_upload, x="Coût estimé ($)",
                                nbins=30,
                                color_discrete_sequence=["#00c8b4"],
                                title="Distribution des coûts prédits",
                            )
                            fig_dist.update_layout(
                                paper_bgcolor="#0c1a22",
                                plot_bgcolor="#0c1a22",
                                font={"color": "#d0e8e4"},
                                bargap=0.05,
                            )
                            st.plotly_chart(fig_dist, use_container_width=True)

                            # Export
                            csv_out = df_upload.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "📥 Télécharger les résultats CSV",
                                data      = csv_out,
                                file_name = "predictions_health_cost.csv",
                                mime      = "text/csv",
                                use_container_width=True,
                            )
                        else:
                            st.error(f"Erreur API : {resp.json()}")
                    except Exception as e:
                        st.error(f"❌ Erreur : {e}")
        else:
            st.markdown("""
            <div style="text-align:center;padding:4rem 1rem;color:#4a7a74;border:1px dashed #1e3440;border-radius:12px;">
              <div style="font-size:2.5rem;margin-bottom:0.8rem">📂</div>
              <div>Importez un CSV pour démarrer</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — INFOS MODÈLE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    try:
        resp = requests.get(f"{API_URL}/model-info", timeout=5)
        if resp.status_code == 200:
            info    = resp.json()
            metrics = info.get("metrics", {})

            st.markdown("#### Performance du modèle")
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f'<div class="kpi-card"><div class="kpi-label">R² Test</div><div class="kpi-value">{metrics.get("r2_test", "N/A"):.4f}</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="kpi-card"><div class="kpi-label">RMSE Test</div><div class="kpi-value">${metrics.get("rmse_test", 0):,.0f}</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="kpi-card"><div class="kpi-label">MAE Test</div><div class="kpi-value">${metrics.get("mae_test", 0):,.0f}</div></div>', unsafe_allow_html=True)
            m4.markdown(f'<div class="kpi-card"><div class="kpi-label">R² Train</div><div class="kpi-value">{metrics.get("r2_train", "N/A"):.4f}</div></div>', unsafe_allow_html=True)

            st.markdown("")
            col_params, col_fe = st.columns(2, gap="large")

            with col_params:
                st.markdown("#### ⚙️ Hyperparamètres XGBoost")
                params = info.get("best_params", {})
                for k, v in params.items():
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;padding:0.4rem 0.8rem;'
                        f'background:#111e27;border-radius:6px;margin:0.2rem 0;">'
                        f'<span style="color:#7ecec4;font-family:monospace">{k}</span>'
                        f'<span style="color:#00c8b4;font-weight:600">{v}</span></div>',
                        unsafe_allow_html=True
                    )

            with col_fe:
                st.markdown("#### 🧬 Feature Engineering")
                fe_info = info.get("feature_engineering", {
                    "age_category" : "Regroupement en classes d'âge (Young / Middle-aged / Senior)",
                    "bmi_category" : "Regroupement OMS (Underweight / Normal / Overweight / Obese)",
                    "age_imc"      : "Interaction numérique : âge × IMC",
                    "parent_fumeur": "Indicateur binaire : fumeur avec enfant(s)",
                })
                icons = {"age_category": "🎂", "bmi_category": "⚖️", "age_imc": "✖️", "parent_fumeur": "🚬"}
                for feat, desc in fe_info.items():
                    icon = icons.get(feat, "📌")
                    st.markdown(
                        f'<div style="padding:0.5rem 0.8rem;background:#111e27;border-radius:6px;margin:0.2rem 0;">'
                        f'<b style="color:#00c8b4">{icon} {feat}</b><br>'
                        f'<span style="font-size:0.82rem;color:#7ecec4">{desc}</span></div>',
                        unsafe_allow_html=True
                    )

            st.markdown("#### 📊 Importance des variables")
            fi = pd.DataFrame(info.get("feature_importances", []))
            if not fi.empty:
                fig_fi = px.bar(
                    fi.head(15), x="Importance", y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale=[[0, "#1a3a4a"], [1, "#00c8b4"]],
                    title="Top features — XGBoost (gain moyen)",
                )
                fig_fi.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    paper_bgcolor="#0c1a22",
                    plot_bgcolor="#0c1a22",
                    font={"color": "#d0e8e4"},
                    coloraxis_showscale=False,
                    margin={"t": 40},
                    height=420,
                )
                fig_fi.update_traces(marker_line_width=0)
                st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.warning("⚠️ Impossible de récupérer les informations du modèle.")
    except Exception as e:
        st.warning(f"⚠️ Connexion à l'API impossible : {e}")