# Peut-on prédire les résultats de foot avec l'IA ?

Un projet de data science s'est penché sur une question simple : en regardant les statistiques récentes de deux équipes, peut-on deviner qui va gagner avant même le coup d'envoi ? Voici ce qu'on en a appris, sans jargon technique.

## Le principe

L'idée est de regarder, pour chaque équipe, sa forme des 5 derniers matchs : combien de buts elle a marqués, encaissés, combien de tirs elle a tentés, combien de points elle a pris. Un programme informatique (on parle de "modèle") analyse ensuite des centaines de matchs passés pour apprendre à repérer des tendances : est-ce qu'une équipe en grande forme offensive a plus de chances de gagner son prochain match ?

Trois façons différentes de faire cet apprentissage ont été testées (trois "modèles"), un peu comme on consulterait trois experts avec des méthodes de raisonnement différentes, pour voir lequel se trompe le moins souvent.

## Sur quelles données ?

L'analyse porte sur 5 saisons complètes du championnat espagnol (La Liga), soit plus de 1 500 matchs. Important : le programme n'a jamais eu accès aux cotes des sites de paris sportifs, volontairement — uniquement des statistiques de jeu brutes (buts, tirs, corners). L'idée était de voir ce que ces statistiques, seules, permettent de deviner.

## Ce qu'on a découvert

**Premier constat, le plus important : l'équipe qui joue à domicile gagne très souvent.** Sur l'ensemble des matchs étudiés, l'équipe locale l'emporte pratiquement une fois sur deux. Rien qu'en pariant systématiquement sur l'équipe à domicile, sans aucune analyse, on devine juste presque une fois sur deux. C'est la barre de référence que les modèles doivent dépasser pour prouver qu'ils apportent quelque chose en plus.

**Deuxième constat : les modèles font difficilement mieux que ce simple réflexe.** Le meilleur des trois arrive tout juste à égaler ce score. Les deux autres font même un peu moins bien. Autrement dit, même avec de l'intelligence artificielle, deviner le vainqueur d'un match de foot à l'avance reste très difficile — et c'est normal.

**Troisième constat, plus subtil : certains modèles "trichent" un peu pour avoir l'air performants.** L'un des modèles a appris une stratégie un peu malhonnê te : il ne prédit presque jamais un match nul. Comme les matchs nuls sont plus rares et plus durs à deviner, éviter d'en prédire fait gagner quelques points au score global — mais rend le modèle moins utile en pratique, puisqu'il ne sait jamais signaler les matchs équilibrés. Un autre modèle, moins performant sur le papier, se montre en réalité plus honnête : il ose prédire des matchs nuls, même si ça lui coûte quelques points de précision globale.

## Pourquoi c'est si difficile de prédire un match de foot

Trois raisons simples expliquent ces résultats :

- **Le foot, c'est peu de buts et beaucoup de hasard.** Un seul but peut décider d'un match, et ce but peut venir d'un coup de chance, d'une erreur d'arbitrage ou d'un rebond improbable — pas forcément du niveau réel des équipes.
- **"Jouer à la maison" est déjà un très bon indice à lui seul.** Les modèles doivent faire mieux qu'un réflexe déjà très efficace, ce qui est une barre haute.
- **Beaucoup d'informations utiles manquent.** Blessures, forme psychologique, enjeu du match (une finale ne se joue pas comme un match sans enjeu), tactique du jour : rien de tout ça n'était disponible pour ce projet, qui s'est volontairement limité aux statistiques de jeu passées.

## Ce qu'il faut retenir

Ce projet montre que l'IA repère bien un signal logique — une équipe en forme a statistiquement plus de chances de bien jouer — mais que ce signal reste trop faible, à lui seul, pour battre nettement l'avantage du terrain. Ce n'est pas un échec : c'est une démonstration honnête que le football garde une bonne part d'imprévisible, même face à l'intelligence artificielle — ce qui, après tout, fait aussi le charme du sport.
