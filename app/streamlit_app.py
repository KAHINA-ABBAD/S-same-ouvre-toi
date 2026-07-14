import json
from pathlib import Path

import joblib
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent
MODEL_PATH = ROOT_DIR / "models" / "xgb_tuned.joblib"
AUDIT_PATH = APP_DIR / "audit_results.json"

FEATURES = ["age", "is_male", "priors_count", "juv_total", "is_felony"]

st.set_page_config(page_title="Sésame, ouvre-toi — Audit COMPAS", page_icon="⚖️", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


@st.cache_data
def load_audit_results():
    return json.loads(AUDIT_PATH.read_text())


model = load_model()
explainer = load_explainer(model)
audit = load_audit_results()

st.title("⚖️ Sésame, ouvre-toi — Audit du biais algorithmique COMPAS")
st.caption(
    "Interface de démonstration : prédiction de récidive à 2 ans (modèle indépendant, sans le score "
    "COMPAS ni la race comme feature), comparaison à COMPAS, et résultats de l'audit d'équité (Parties 1 & 2)."
)

tab_predict, tab_compas, tab_race, tab_sexe = st.tabs(
    ["🔮 Prédiction interactive", "📊 Comparaison à COMPAS", "🧑🏾‍🤝‍🧑🏻 Audit — Race", "🚻 Audit — Sexe"]
)

# ---------------------------------------------------------------------------
# Onglet 1 : prédiction interactive + explication SHAP locale
# ---------------------------------------------------------------------------
with tab_predict:
    st.subheader("Simuler un profil et voir la prédiction du modèle")
    st.info(
        "⚠️ Ce modèle n'utilise **que 5 variables** (`age`, `is_male`, `priors_count`, `juv_total`, "
        "`is_felony`) et **jamais** la race, conformément au choix *fairness through unawareness* de la "
        "Partie 2 (section 2.1.3). C'est un outil pédagogique d'audit, pas un outil de décision réelle.",
        icon="ℹ️",
    )

    col_in, col_out = st.columns([1, 1.4], gap="large")

    with col_in:
        age = st.slider("Âge", min_value=18, max_value=96, value=30)
        sexe = st.radio("Sexe", ["Homme", "Femme"], horizontal=True)
        priors_count = st.slider("Nombre d'antécédents (priors_count)", min_value=0, max_value=38, value=2)
        juv_total = st.slider(
            "Antécédents judiciaires mineurs cumulés (juv_total)", min_value=0, max_value=21, value=0
        )
        is_felony = st.radio("Type de charge", ["Félonie", "Délit"], horizontal=True)

    x_input = pd.DataFrame(
        [{
            "age": age,
            "is_male": 1 if sexe == "Homme" else 0,
            "priors_count": priors_count,
            "juv_total": juv_total,
            "is_felony": 1 if is_felony == "Félonie" else 0,
        }]
    )[FEATURES]

    proba = float(model.predict_proba(x_input)[0, 1])
    risque = "Récidive prédite" if proba >= 0.5 else "Pas de récidive prédite"

    with col_out:
        st.metric("Probabilité de récidive à 2 ans (modèle)", f"{proba:.1%}")
        st.progress(min(max(proba, 0.0), 1.0))
        if proba >= 0.5:
            st.error(f"**{risque}** (seuil de décision = 50 %)")
        else:
            st.success(f"**{risque}** (seuil de décision = 50 %)")

        st.markdown("**Profil saisi :**")
        st.dataframe(x_input, hide_index=True, width='stretch')

    st.markdown("---")
    st.markdown("### Décomposition SHAP de cette prédiction")
    st.caption(
        "Contribution de chaque variable à l'écart entre cette prédiction et la prédiction moyenne du "
        "modèle (base value) — méthode TreeSHAP, exacte, cf. Partie 2 section 2.5."
    )

    shap_values_input = explainer.shap_values(x_input)
    explanation = shap.Explanation(
        values=shap_values_input[0],
        base_values=explainer.expected_value,
        data=x_input.iloc[0].values,
        feature_names=FEATURES,
    )
    fig, ax = plt.subplots(figsize=(8, 3))
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(fig, clear_figure=True)

# ---------------------------------------------------------------------------
# Onglet 2 : comparaison à COMPAS
# ---------------------------------------------------------------------------
with tab_compas:
    st.subheader("Performance : notre modèle vs COMPAS (decile_score ≥ 7)")
    st.caption("Calculé sur le même test set (1 780 individus), jamais vu pendant l'entraînement — Partie 2, section 2.4.")

    comp = audit["comparaison_compas"]
    df_comp = pd.DataFrame(
        {"COMPAS": comp["COMPAS (decile_score >= 7)"], "Notre modèle": comp["Notre modele (XGBoost optimise)"]},
        index=comp["metrics"],
    )
    col_table, col_chart = st.columns([1, 1.2], gap="large")
    with col_table:
        st.dataframe(df_comp.style.format("{:.3f}"), width='stretch')
    with col_chart:
        fig, ax = plt.subplots(figsize=(6, 4))
        df_comp.plot.barh(ax=ax, color=["#AAAAAA", "#2D6A9F"])
        ax.set_xlabel("Score")
        ax.set_title("Notre modèle vs COMPAS", fontweight="bold")
        st.pyplot(fig, clear_figure=True)

    st.warning(
        "Comparaison partielle : COMPAS est un système propriétaire dont la totalité des features n'est pas "
        "connue, et le seuil `decile_score ≥ 7` est une convention qui ne reflète peut-être pas exactement "
        "le seuil opérationnel réellement utilisé par les tribunaux.",
        icon="⚠️",
    )

# ---------------------------------------------------------------------------
# Onglet 3 : audit d'équité — race
# ---------------------------------------------------------------------------
with tab_race:
    st.subheader("Audit d'équité par race — African-American vs Caucasian")
    race = audit["audit_race"]
    df_race = pd.DataFrame({k: v for k, v in race.items() if k != "metrics"}, index=race["metrics"])
    st.dataframe(df_race.style.format("{:.3f}"), width='stretch')

    lo, hi = audit["seuil_equite"]["disparate_impact_min"], audit["seuil_equite"]["disparate_impact_max"]
    di_row = df_race.loc["Disparate Impact"]
    eod_row = df_race.loc["Equalized Odds Diff"]

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(di_row.index, di_row.values, marker="o", color="#2D6A9F", linewidth=2)
        ax.axhspan(lo, hi, color="#2D6A9F", alpha=0.1, label=f"Seuil acceptable ({lo}-{hi})")
        ax.set_title("Disparate Impact — étapes de mitigation", fontweight="bold")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=8)
        st.pyplot(fig, clear_figure=True)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(eod_row.index, eod_row.values, marker="o", color="#E84A5F", linewidth=2)
        ax.axhline(0, color="#333", linestyle="--", linewidth=1)
        ax.set_title("Equalized Odds Difference (0 = équité parfaite)", fontweight="bold")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig, clear_figure=True)

    st.success(
        "Reweighing + ThresholdOptimizer combinés ramènent le Disparate Impact à **1,000** (parité "
        "démographique atteinte) et l'Equalized Odds Diff à **0,080** (contre 0,286 pour COMPAS), pour un "
        "coût d'accuracy inférieur à 6 points sur le groupe le plus touché.",
        icon="✅",
    )
    st.warning(
        "Le `ThresholdOptimizer` choisit un seuil de décision différent selon la race au moment de la "
        "prédiction (*disparate treatment*) — un compromis légal/éthique à documenter, pas une solution "
        "sans contrepartie (cf. Partie 2, section 2.6).",
        icon="⚠️",
    )

# ---------------------------------------------------------------------------
# Onglet 4 : audit d'équité — sexe
# ---------------------------------------------------------------------------
with tab_sexe:
    st.subheader("Audit d'équité par sexe — Femme vs Homme")
    sexe_data = audit["audit_sexe"]
    df_sexe = pd.DataFrame({k: v for k, v in sexe_data.items() if k != "metrics"}, index=sexe_data["metrics"])
    st.dataframe(df_sexe.style.format("{:.3f}"), width='stretch')

    st.error(
        "Contrairement à la race, le sexe n'a **pas été corrigé** : le Disparate Impact de notre modèle "
        "(0,314) est **plus mauvais** que celui de COMPAS (0,654), et l'Equalized Odds Diff **double presque** "
        "(0,193 → 0,350). Optimiser la race sans surveiller le sexe a dégradé cet axe — un exemple concret de "
        "*fairness gerrymandering* (cf. Partie 2, section 2.6bis). Une mitigation dédiée au sexe reste à faire.",
        icon="🚨",
    )

st.markdown("---")
st.caption("Sésame, ouvre-toi — projet d'audit du biais algorithmique COMPAS. Voir `sesame_partie1_eda.ipynb` et `sesame_partie2_modelisation.ipynb` pour l'analyse complète.")
