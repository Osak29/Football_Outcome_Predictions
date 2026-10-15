"""
Entraine et compare 3 modeles de prediction du resultat d'un match
(Victoire Domicile / Nul / Victoire Exterieur) :
  - Regression Logistique (baseline interpretable)
  - Random Forest
  - XGBoost

Split TEMPOREL (pas aleatoire) : on entraine sur le debut de saison et on
teste sur la fin, pour simuler une vraie situation de prediction (on ne
doit jamais s'entrainer sur le "futur" par rapport au test).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_prep import charger_plusieurs_saisons, vers_format_long
from features import calculer_forme_glissante, construire_dataset_features, FEATURE_COLUMNS

RACINE = Path(__file__).resolve().parent.parent
DATA_DIR = RACINE / "data"
OUTPUTS = RACINE / "outputs"
OUTPUTS.mkdir(exist_ok=True)

RESULT_LABELS = {"H": "Victoire domicile", "D": "Match nul", "A": "Victoire exterieur"}


def preparer_dataset():
    fichiers = sorted(str(p) for p in DATA_DIR.glob("*.csv"))
    print(f"Chargement de {len(fichiers)} fichier(s) saison : {[Path(f).name for f in fichiers]}")
    df = charger_plusieurs_saisons(fichiers)
    long_df = vers_format_long(df)
    long_df = calculer_forme_glissante(long_df)
    dataset = construire_dataset_features(df, long_df)

    # On retire les tout premiers matchs de chaque equipe (0 ou tres peu
    # d'historique -> features de forme peu fiables / NaN)
    dataset = dataset.dropna(subset=FEATURE_COLUMNS)
    return dataset


def split_temporel(dataset: pd.DataFrame, ratio_train: float = 0.8):
    dataset = dataset.sort_values("Date").reset_index(drop=True)
    coupure = int(len(dataset) * ratio_train)
    train = dataset.iloc[:coupure]
    test = dataset.iloc[coupure:]
    return train, test


def entrainer_et_evaluer():
    dataset = preparer_dataset()
    train, test = split_temporel(dataset)

    X_train, y_train = train[FEATURE_COLUMNS], train["FTR"]
    X_test, y_test = test[FEATURE_COLUMNS], test["FTR"]

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    label_map = {"H": 0, "D": 1, "A": 2}
    inv_label_map = {v: k for k, v in label_map.items()}
    y_train_num = y_train.map(label_map)
    y_test_num = y_test.map(label_map)

    modeles = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            eval_metric="mlogloss", random_state=42,
        ),
    }

    # Baseline naive : toujours predire la classe majoritaire du train
    # (ici "victoire domicile"). Un bon modele doit faire mieux que ca.
    classe_majoritaire = y_train.value_counts().idxmax()
    acc_baseline = (y_test == classe_majoritaire).mean()

    resultats = {"_baseline_classe_majoritaire": {
        "classe": classe_majoritaire, "accuracy": acc_baseline,
    }}
    importances = {}
    predictions_test = test[["Date", "HomeTeam", "AwayTeam", "FTR"]].copy()

    for nom, modele in modeles.items():
        if nom == "logistic_regression":
            modele.fit(X_train_s, y_train_num)
            y_pred = modele.predict(X_test_s)
            y_proba = modele.predict_proba(X_test_s)
            importance = np.abs(modele.coef_).mean(axis=0)
        else:
            modele.fit(X_train, y_train_num)
            y_pred = modele.predict(X_test)
            y_proba = modele.predict_proba(X_test)
            importance = modele.feature_importances_

        acc = accuracy_score(y_test_num, y_pred)
        report = classification_report(
            y_test_num, y_pred, target_names=["H", "D", "A"], output_dict=True
        )
        matrice = confusion_matrix(y_test_num, y_pred).tolist()

        resultats[nom] = {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": matrice,
        }
        importances[nom] = dict(zip(FEATURE_COLUMNS, importance.tolist()))

        predictions_test[f"pred_{nom}"] = [inv_label_map[p] for p in y_pred]
        for i, classe in enumerate(["H", "D", "A"]):
            predictions_test[f"proba_{nom}_{classe}"] = y_proba[:, i]

        joblib.dump(modele, OUTPUTS / f"modele_{nom}.joblib")
        print(f"{nom:22s} accuracy = {acc:.3f}")

    joblib.dump(scaler, OUTPUTS / "scaler.joblib")

    with open(OUTPUTS / "resultats_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(resultats, f, ensure_ascii=False, indent=2)

    with open(OUTPUTS / "importances_features.json", "w", encoding="utf-8") as f:
        json.dump(importances, f, ensure_ascii=False, indent=2)

    predictions_test.to_csv(OUTPUTS / "predictions_test.csv", index=False)
    dataset.to_csv(OUTPUTS / "dataset_features.csv", index=False)

    return resultats, importances


if __name__ == "__main__":
    entrainer_et_evaluer()
    print("\nTermine. Resultats dans le dossier outputs/.")
