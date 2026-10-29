# Développement

## Installation

Depuis la racine du dépôt, dans un environnement virtuel :

```bash
python -m pip install -r requirements-dev.txt
```

Les dépendances d'exécution restent dans `requirements.txt`. Le fichier de développement ajoute une version fixée de Ruff, utilisée aussi par GitHub Actions.

## Style Python

```bash
ruff check .
ruff format --check .
```

Pour appliquer le formatage :

```bash
ruff format .
```

La configuration est dans [`pyproject.toml`](../pyproject.toml) : Python 3.9, lignes de 100 caractères, imports triés et vérifications des erreurs courantes. Les scripts CLI ajoutent la racine du dépôt au chemin d'import avant de charger `specdec` ; cette exception est déclarée une seule fois dans la configuration.

## Tests

```bash
BAGUETTE_SOURCE=../LLM python -m unittest discover -s tests -v
```

Les tests n'utilisent pas les gros checkpoints. Les tests du cache construisent une petite architecture depuis le `model.py` de Baguette. Sans ce fichier, cette classe de tests est ignorée ; les autres tests restent disponibles.

La [configuration GitHub Actions](../.github/workflows/checks.yml) récupère uniquement `model.py` depuis une révision fixée du dépôt Baguette et exécute les tests sur CPU avec Python 3.11. Elle vérifie aussi le formatage et les imports. Pour changer la version de l'architecture testée, modifier explicitement sa référence dans ce workflow et relancer les tests.

## Mesures de performance

Les tests vérifient la correction, pas la vitesse. Les benchmarks se lancent séparément sur la machine cible : voir les [commandes d'expérimentation](experiments.md) et l'[index des rapports](../bench_reports/README.md).

Les résultats bruts contiennent une empreinte des sources au moment de la mesure. Un formatage du code modifie cette empreinte ; il ne faut pas remplacer les empreintes historiques ni présenter les anciens chiffres comme un nouveau benchmark.

## Fichiers locaux

Les environnements virtuels, caches, poids, corpus et sorties dans `results/` sont exclus de Git. Les rapports publiés dans `bench_reports/` conservent les données nécessaires pour vérifier les conclusions. Aucun checkpoint n'est requis pour lire le code ou lancer les tests.

[Retour au projet](../README.md)
