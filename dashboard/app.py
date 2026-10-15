"""
Dashboard de pilotage du projet "Football Match Outcome Prediction".

Lancement : streamlit run dashboard/app.py   (depuis la racine du projet)
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

from data_prep import charger_plusieurs_saisons, vers_format_long          # noqa: E402
from features import calculer_forme_glissante, construire_dataset_features, FEATURE_COLUMNS  # noqa: E402

OUTPUTS = RACINE / "outputs"
DATA_DIR = RACINE / "data"

st.set_page_config(page_title="Football Match Prediction", page_icon="⚽", layout="wide")
st.title("⚽ Football Match Outcome Prediction")
st.caption("La Liga — Analyse prédictive & Machine Learning (Logistic Regression, Random Forest, XGBoost)")


@st.cache_data
def charger_resultats():
    with open(OUTPUTS / "resultats_evaluation.json", encoding="utf-8") as f:
        resultats = json.load(f)
    with open(OUTPUTS / "importances_features.json", encoding="utf-8") as f:
        importances = json.load(f)
    predictions = pd.read_csv(OUTPUTS / "predictions_test.csv")
    dataset = pd.read_csv(OUTPUTS / "dataset_features.csv")
    return resultats, importances, predictions, dataset


@st.cache_resource
def charger_modeles():
    modeles = {
        "logistic_regression": joblib.load(OUTPUTS / "modele_logistic_regression.joblib"),
        "random_forest": joblib.load(OUTPUTS / "modele_random_forest.joblib"),
        "xgboost": joblib.load(OUTPUTS / "modele_xgboost.joblib"),
    }
    scaler = joblib.load(OUTPUTS / "scaler.joblib")
    return modeles, scaler


resultats, importances, predictions, dataset = charger_resultats()
NOMS_MODELES = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost"}

tab_eda, tab_modeles, tab_predire = st.tabs(["📊 Exploration des données", "🤖 Comparaison des modèles", "🔮 Prédire un match"])

# ---------------------------------------------------------------------------
# Onglet EDA
# ---------------------------------------------------------------------------
with tab_eda:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Matchs analysés", len(dataset))
    col2.metric("Saisons", dataset["saison"].nunique() if "saison" in dataset.columns else "—")
    col3.metric("Équipes", dataset["HomeTeam"].nunique())
    col4.metric("Période", f"{dataset['Date'].min()[:10]} → {dataset['Date'].max()[:10]}")

    st.subheader("Répartition des résultats")
    repartition = dataset["FTR"].value_counts().rename({"H": "Victoire domicile", "D": "Match nul", "A": "Victoire extérieur"})
    st.bar_chart(repartition)
    st.caption(
        f"L'avantage du terrain est net : {repartition.iloc[0] / len(dataset):.0%} des matchs sont gagnés "
        "par l'équipe à domicile — c'est la baseline que tout modèle doit battre."
    )

    st.subheader("Forme récente moyenne par équipe (points sur les 5 derniers matchs)")
    derniere_forme = (
        dataset.sort_values("Date")
        .groupby("HomeTeam")["dom_forme_points"].last()
        .sort_values(ascending=False)
    )
    st.bar_chart(derniere_forme)

# ---------------------------------------------------------------------------
# Onglet Comparaison des modèles
# ---------------------------------------------------------------------------
with tab_modeles:
    baseline = resultats["_baseline_classe_majoritaire"]
    st.subheader("Accuracy sur le jeu de test (20% des matchs les plus récents)")

    lignes = [{"Modèle": "Baseline (toujours \"victoire domicile\")", "Accuracy": baseline["accuracy"]}]
    for cle, nom in NOMS_MODELES.items():
        lignes.append({"Modèle": nom, "Accuracy": resultats[cle]["accuracy"]})
    df_acc = pd.DataFrame(lignes).set_index("Modèle")
    st.bar_chart(df_acc)

    meilleur = max(NOMS_MODELES, key=lambda k: resultats[k]["accuracy"])
    if resultats[meilleur]["accuracy"] <= baseline["accuracy"]:
        st.warning(
            "⚠️ Aucun modèle ne bat encore la baseline naïve sur ce jeu de test. "
            "C'est attendu avec une seule saison (~300 matchs d'entraînement) : le signal "
            "de forme récente est bruité sur un échantillon aussi petit. Pistes d'amélioration "
            "ci-dessous."
        )
    else:
        st.success(f"Le meilleur modèle ({NOMS_MODELES[meilleur]}) bat la baseline naïve.")

    st.subheader("Importance des variables")
    modele_choisi = st.selectbox("Modèle", list(NOMS_MODELES.values()))
    cle_choisie = [k for k, v in NOMS_MODELES.items() if v == modele_choisi][0]
    imp = pd.Series(importances[cle_choisie]).sort_values(ascending=False).head(10)
    st.bar_chart(imp)

    with st.expander("Rapport de classification détaillé"):
        st.json(resultats[cle_choisie]["classification_report"])

    with st.expander("💡 Pistes pour améliorer la performance"):
        st.markdown("""
- **Plus de données** : combiner plusieurs saisons (football-data.co.uk propose les mêmes
  colonnes sur 20+ saisons) — 300 matchs c'est peu pour 19 features.
- **Features complémentaires** : classement actuel, confrontations directes (head-to-head),
  jours de repos entre matchs, absences/blessures.
- **Rééquilibrage** : les matchs nuls sont sous-représentés et mal prédits — tester
  `class_weight="balanced"` ou du rééchantillonnage.
- **Validation croisée temporelle** (rolling-origin) plutôt qu'un seul split, pour une
  estimation plus robuste de la performance réelle.
        """)

# ---------------------------------------------------------------------------
# Onglet Prédiction
# ---------------------------------------------------------------------------
with tab_predire:
    st.subheader("Prédire un match à partir de la forme actuelle des équipes")
    equipes = sorted(dataset["HomeTeam"].unique())
    c1, c2 = st.columns(2)
    equipe_dom = c1.selectbox("Équipe à domicile", equipes, index=0)
    equipe_ext = c2.selectbox("Équipe à l'extérieur", equipes, index=1)

    if st.button("Prédire"):
        if equipe_dom == equipe_ext:
            st.error("Choisis deux équipes différentes.")
        else:
            fichiers = sorted(str(p) for p in DATA_DIR.glob("*.csv"))
            df = charger_plusieurs_saisons(fichiers)
            long_df = vers_format_long(df)
            long_df = calculer_forme_glissante(long_df)

            derniere_dom = long_df[long_df["team"] == equipe_dom].sort_values("Date").iloc[-1]
            derniere_ext = long_df[long_df["team"] == equipe_ext].sort_values("Date").iloc[-1]

            # Reconstruit un vecteur de features "comme si" ces deux equipes
            # jouaient leur prochain match, a partir de leur forme la plus
            # recente deja calculee (forme_* = moyenne glissante incluant
            # leur dernier match joue, donc valable comme etat "actuel").
            ligne = {}
            for col in [c for c in long_df.columns if c.startswith("forme_")] + ["matchs_joues_avant"]:
                ligne[f"dom_{col}"] = derniere_dom[col]
                ligne[f"ext_{col}"] = derniere_ext[col]
            ligne["diff_forme_points"] = ligne["dom_forme_points"] - ligne["ext_forme_points"]
            ligne["diff_forme_buts_pour"] = ligne["dom_forme_buts_pour"] - ligne["ext_forme_buts_pour"]
            ligne["diff_forme_buts_contre"] = ligne["dom_forme_buts_contre"] - ligne["ext_forme_buts_contre"]

            X = pd.DataFrame([ligne])[FEATURE_COLUMNS]

            modeles, scaler = charger_modeles()
            labels = {0: "Victoire domicile", 1: "Match nul", 2: "Victoire extérieur"}

            st.write(f"**{equipe_dom}** (domicile) vs **{equipe_ext}** (extérieur)")
            for cle, nom in NOMS_MODELES.items():
                modele = modeles[cle]
                Xin = scaler.transform(X) if cle == "logistic_regression" else X
                proba = modele.predict_proba(Xin)[0]
                st.write(f"**{nom}**")
                st.bar_chart(pd.Series({labels[i]: p for i, p in enumerate(proba)}))
