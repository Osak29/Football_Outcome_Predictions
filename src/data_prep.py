"""
Chargement et nettoyage des donnees brutes (format football-data.co.uk).

On ne garde que les colonnes utiles a la prediction de resultat : scores,
tirs, tirs cadres, corners, fautes, cartons. Toutes les colonnes de cotes
de paris (B365*, PS*, Max*, Avg*, ...) sont ecartees : elles ne sont pas
disponibles AVANT le match pour une vraie prediction, et ce ne sont pas
des statistiques de jeu.
"""
import pandas as pd

COLONNES_UTILES = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",            # score final + resultat (H/D/A)
    "HTHG", "HTAG", "HTR",            # score mi-temps
    "HS", "AS",                       # tirs
    "HST", "AST",                     # tirs cadres
    "HC", "AC",                       # corners
    "HF", "AF",                       # fautes
    "HY", "AY", "HR", "AR",           # cartons jaunes/rouges
]


def charger_donnees(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv, encoding="utf-8-sig")
    df = df[COLONNES_UTILES].copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna(subset=["FTR"])  # securite : lignes sans resultat inexploitables
    return df


def charger_plusieurs_saisons(paths_csv: list[str]) -> pd.DataFrame:
    """Charge et empile plusieurs fichiers saison (meme format
    football-data.co.uk). Chaque fichier telecharge/colle manuellement
    est une saison distincte ; cette fonction les assemble en un seul
    historique continu, trie chronologiquement.

    NB : c'est precisement l'etape qu'une API (ex. football-data.org,
    API-Football) rendrait triviale et fiable — voir le README pour le
    detail de cette limite actuelle du projet (fichiers CSV geres a la
    main, un par saison, sans mise a jour automatique)."""
    morceaux = []
    for path in paths_csv:
        df = charger_donnees(path)
        # Convention foot europeen : la saison est notee "annee de debut -
        # annee de fin", meme si le fichier ne couvre qu'une partie (saison
        # en cours), d'ou le calcul sur l'annee de debut uniquement.
        annee_debut = df["Date"].min().year
        df["saison"] = f"{annee_debut}-{str(annee_debut + 1)[-2:]}"
        morceaux.append(df)

    dataset = pd.concat(morceaux, ignore_index=True)
    dataset = dataset.sort_values("Date").reset_index(drop=True)

    avant = sum(len(m) for m in morceaux)
    apres = dataset.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"]).shape[0]
    if apres != avant:
        print(f"Attention : {avant - apres} doublons detectes entre fichiers et retires.")
        dataset = dataset.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    return dataset


def vers_format_long(df: pd.DataFrame) -> pd.DataFrame:
    """Transforme le DataFrame match-par-ligne en vue 'par equipe' (2 lignes
    par match : une pour l'equipe a domicile, une pour l'exterieur). C'est
    ce format qui permet de calculer la forme recente de chaque equipe,
    quel que soit son statut domicile/exterieur au match suivant."""

    def points(buts_pour, buts_contre):
        if buts_pour > buts_contre:
            return 3
        if buts_pour == buts_contre:
            return 1
        return 0

    domicile = pd.DataFrame({
        "Date": df["Date"],
        "match_idx": df.index,
        "team": df["HomeTeam"],
        "opponent": df["AwayTeam"],
        "is_home": 1,
        "buts_pour": df["FTHG"],
        "buts_contre": df["FTAG"],
        "tirs_pour": df["HS"],
        "tirs_contre": df["AS"],
        "tirs_cadres_pour": df["HST"],
        "tirs_cadres_contre": df["AST"],
        "corners_pour": df["HC"],
        "corners_contre": df["AC"],
    })
    exterieur = pd.DataFrame({
        "Date": df["Date"],
        "match_idx": df.index,
        "team": df["AwayTeam"],
        "opponent": df["HomeTeam"],
        "is_home": 0,
        "buts_pour": df["FTAG"],
        "buts_contre": df["FTHG"],
        "tirs_pour": df["AS"],
        "tirs_contre": df["HS"],
        "tirs_cadres_pour": df["AST"],
        "tirs_cadres_contre": df["HST"],
        "corners_pour": df["AC"],
        "corners_contre": df["HC"],
    })

    long_df = pd.concat([domicile, exterieur], ignore_index=True)
    long_df["points"] = [
        points(bp, bc) for bp, bc in zip(long_df["buts_pour"], long_df["buts_contre"])
    ]
    long_df = long_df.sort_values(["team", "Date"]).reset_index(drop=True)
    return long_df
