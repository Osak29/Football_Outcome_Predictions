"""
Feature engineering : calcule la "forme recente" de chaque equipe
(moyenne glissante sur les N derniers matchs) et la rattache a chaque
match du dataset d'origine.

Point crucial (anti data-leakage) : on utilise `.shift(1)` avant le
rolling, donc la forme calculee pour un match ne contient QUE les matchs
precedents de l'equipe, jamais le match courant ni les suivants. C'est
indispensable pour que l'evaluation du modele soit honnete.
"""
import pandas as pd

FENETRE_FORME = 5  # nombre de matchs precedents pris en compte


def calculer_forme_glissante(long_df: pd.DataFrame, fenetre: int = FENETRE_FORME) -> pd.DataFrame:
    colonnes = ["buts_pour", "buts_contre", "tirs_pour", "tirs_contre",
                "tirs_cadres_pour", "corners_pour", "points"]

    long_df = long_df.sort_values(["team", "Date"]).copy()

    for col in colonnes:
        long_df[f"forme_{col}"] = (
            long_df.groupby("team")[col]
            .apply(lambda s: s.shift(1).rolling(fenetre, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

    # Nombre de matchs deja joues par l'equipe avant celui-ci (utile pour
    # filtrer / ponderer les tout premiers matchs de la saison, ou il y a
    # peu d'historique)
    long_df["matchs_joues_avant"] = long_df.groupby("team").cumcount()

    return long_df


def construire_dataset_features(df_matchs: pd.DataFrame, long_df_avec_forme: pd.DataFrame) -> pd.DataFrame:
    """Rattache, pour chaque match, les features de forme recente de
    l'equipe a domicile et de l'equipe a l'exterieur."""
    colonnes_forme = [c for c in long_df_avec_forme.columns if c.startswith("forme_")] + ["matchs_joues_avant"]

    base = long_df_avec_forme[["match_idx", "is_home"] + colonnes_forme]

    dom = base[base["is_home"] == 1].drop(columns=["is_home"]).set_index("match_idx")
    dom = dom.add_prefix("dom_")

    ext = base[base["is_home"] == 0].drop(columns=["is_home"]).set_index("match_idx")
    ext = ext.add_prefix("ext_")

    out = df_matchs.join(dom, how="left").join(ext, how="left")

    # Features "differentielles" (souvent les plus predictives : qui est
    # en meilleure forme relative ?)
    out["diff_forme_points"] = out["dom_forme_points"] - out["ext_forme_points"]
    out["diff_forme_buts_pour"] = out["dom_forme_buts_pour"] - out["ext_forme_buts_pour"]
    out["diff_forme_buts_contre"] = out["dom_forme_buts_contre"] - out["ext_forme_buts_contre"]

    return out


FEATURE_COLUMNS = [
    "dom_forme_buts_pour", "dom_forme_buts_contre", "dom_forme_tirs_pour",
    "dom_forme_tirs_contre", "dom_forme_tirs_cadres_pour", "dom_forme_corners_pour",
    "dom_forme_points", "dom_matchs_joues_avant",
    "ext_forme_buts_pour", "ext_forme_buts_contre", "ext_forme_tirs_pour",
    "ext_forme_tirs_contre", "ext_forme_tirs_cadres_pour", "ext_forme_corners_pour",
    "ext_forme_points", "ext_matchs_joues_avant",
    "diff_forme_points", "diff_forme_buts_pour", "diff_forme_buts_contre",
]
