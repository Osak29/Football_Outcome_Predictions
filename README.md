# ⚽ Football Match Outcome Prediction — La Liga

Pipeline complet de prédiction de résultat de match (Victoire Domicile /
Nul / Victoire Extérieur), à partir de données football-data.co.uk
(fichier `SP1.csv` = La Liga).

## Pourquoi ce découpage

- **`src/data_prep.py`** : charge le CSV brut, ne garde que les colonnes
  de jeu réelles (scores, tirs, corners, fautes, cartons) — toutes les
  colonnes de cotes de paris sont écartées (elles ne reflètent pas des
  statistiques de jeu et ne seraient pas disponibles aussi facilement en
  conditions réelles). Transforme ensuite le dataset "1 ligne = 1 match"
  en vue "1 ligne = 1 équipe pour 1 match", nécessaire pour calculer une
  forme par équipe.
- **`src/features.py`** : calcule la forme récente de chaque équipe
  (moyenne glissante sur les 5 derniers matchs : buts, tirs, tirs
  cadrés, corners, points). **Point important** : un `.shift(1)` est
  appliqué avant le rolling, donc la feature d'un match ne contient
  jamais d'information sur ce match lui-même ni sur le futur — sans
  cette précaution le modèle "tricherait" et les métriques seraient
  fausses.
- **`src/train_models.py`** : entraîne Logistic Regression, Random
  Forest et XGBoost, avec un **split temporel** (80% premiers matchs de
  la saison pour l'entraînement, 20% derniers pour le test) plutôt
  qu'un split aléatoire — pour simuler une vraie situation de prédiction
  où on ne connaît jamais le futur à l'entraînement.
- **`dashboard/app.py`** : dashboard Streamlit (exploration des données,
  comparaison des modèles, importance des variables, prédiction
  interactive d'un match entre deux équipes choisies).

## Données utilisées

Le dossier `data/` contient 5 fichiers saison (format football-data.co.uk,
La Liga) : `S22.csv` (2022-23), `S23.csv` (2023-24), `S24.csv` (2024-25),
`SP1.csv` (2025-26, complète), `S26.csv` (2026-27, en cours, partielle).
`src/data_prep.py` les charge tous automatiquement (tout fichier `.csv`
posé dans `data/`), les empile et les dédoublonne — soit **1598 matchs**
au total, contre ~300 avec une seule saison.

## Lancement

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Entraîne les modèles sur toutes les saisons presentes dans data/
#    (génère outputs/*.joblib, *.json, *.csv)
cd src && python train_models.py && cd ..

# 2. Lance le dashboard
streamlit run dashboard/app.py
```

## Résultats actuels (honnêtes)

Avec les 5 saisons combinées (1598 matchs, split temporel 80/20) :

| Modèle | Accuracy (test) |
|---|---|
| Baseline naïve ("toujours victoire domicile") | ~49% |
| Logistic Regression | ~49% |
| Random Forest | ~47% |
| XGBoost | ~42% |

**Amélioration par rapport à la version à une seule saison** (où tous
les modèles étaient nettement sous la baseline), mais le gain reste
modeste : Logistic Regression rejoint tout juste la baseline, Random
Forest et XGBoost restent en dessous. Le foot reste un sport difficile à
prédire à partir des seules statistiques de jeu (sans cotes de marché,
qui intègrent beaucoup d'information — effectif, blessures, enjeu du
match — que ce projet n'utilise pas volontairement). Pistes
d'amélioration restantes, par ordre d'impact probable :

1. **Features supplémentaires** : classement/points cumulés dans la
   saison, confrontations directes, repos entre matchs, utiliser le xG
   (disponible dans `S26.csv` mais pas dans les saisons plus anciennes —
   actuellement écarté pour garder un jeu de features homogène)
2. **Rééquilibrage des classes** : les matchs nuls sont sous-représentés
   et mal prédits par tous les modèles
3. **Validation croisée temporelle** (plusieurs splits glissants) plutôt
   qu'un split unique, pour une estimation plus fiable
4. **Forme par saison plutôt que continue** : actuellement, la forme
   d'une équipe en tout début de saison est calculée à partir de ses
   derniers matchs de la saison précédente (plus simple à coder, mais
   imparfait : l'effectif a souvent changé l'été suivant)

## Limite structurelle du projet : gestion des données par CSV manuels

Ce projet repose entièrement sur des **fichiers CSV téléchargés et
déposés à la main**, un par saison, depuis football-data.co.uk. Cette
approche a permis d'avancer vite, mais elle a des limites concrètes
qu'une **API de données football** (ex. football-data.org, API-Football,
ou l'export direct de StatsBomb) aurait résolues :

- **Mise à jour automatique** : avec des CSV manuels, le dataset est figé
  au moment du téléchargement — il a fallu re-déposer `S26.csv` à la
  main pour avoir les derniers résultats. Une API interrogée à la
  demande (ou planifiée via un cron) aurait tenu le dataset à jour sans
  intervention.
- **Cohérence du schéma entre saisons** : chaque fichier CSV a un jeu de
  colonnes légèrement différent selon l'année (bookmakers inclus
  variables, présence ou non du xG dans `S26.csv`...), ce qui a demandé
  un travail de nettoyage pour ne garder que les colonnes communes. Une
  API fournit généralement un schéma de réponse stable et versionné,
  qui évite cette maintenance.
- **Détection des doublons/trous** : il a fallu vérifier manuellement
  l'absence de chevauchement de dates entre fichiers (`SP1.csv` vs
  `S26.csv`, par exemple) ; une API avec pagination et filtrage par date
  rend ce risque d'erreur humaine structurellement impossible.
- **Richesse des données** : les CSV ne contiennent pas certaines
  informations utiles (compositions d'équipe, blessures, cotes en temps
  réel), disponibles via certaines API sportives mais absentes de
  football-data.co.uk.
- **Passage à l'échelle** : ajouter un championnat supplémentaire
  (Ligue 1, Serie A...) signifie aujourd'hui télécharger et valider un
  nouveau fichier par saison et par championnat à la main. Avec une API,
  ce serait un simple paramètre (`league=FR1`) dans un appel
  automatisé.

Intégrer une API de données serait donc la prochaine étape naturelle
pour transformer ce projet d'un exercice ponctuel en un pipeline de
données réellement maintenable dans la durée.

## Stack

Python, pandas, scikit-learn, XGBoost, Streamlit.
