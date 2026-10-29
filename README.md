# baguette-SpecDec

Décodage spéculatif pour [Baguette](https://github.com/HENK0O/baguette), avec une interface locale pour comparer le débit, la latence et les textes générés.

Le modèle vérifie plusieurs propositions ensemble pour réduire le nombre de passages nécessaires à la génération. Deux méthodes sont disponibles :

| Mode | Propositions | État actuel |
| --- | --- | --- |
| **Automatique** | Suites retrouvées dans le prompt ou le texte généré, sans petit modèle | Gains mesurés sur la copie et les répétitions ; repli classique si aucune suite utile n'est disponible. |
| **Brouillon entraîné** | Petit modèle nano utilisant le même tokenizer que Baguette | Expérimental : le brouillon actuel reste plus lent que Baguette seul sur les essais réalisés. |

La qualité des réponses reste celle de Baguette. Le projet mesure une accélération, sans modifier les poids du modèle.

## Démarrage

Python **3.9 ou plus récent** et un checkpoint Baguette sont nécessaires. Le code de Baguette, les poids et les données restent à l'extérieur de ce dépôt.

Les exemples supposent cette organisation :

```text
Code/
├── LLM/                      # dépôt Baguette
│   ├── model.py
│   ├── baguette-123m-sft.pt   # checkpoint de poids exporté
│   └── tokenizer.json
└── baguette-SpecDec/
```

Depuis `baguette-SpecDec` :

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/web_ui.py --baguette-source ../LLM
```

Ouvrir [localhost:8765](http://127.0.0.1:8765/). Pour un premier essai, garder **Automatique**, un bloc de **8 tokens**, puis choisir **Exemple de copie → Comparer**. Le bouton **Documentation** explique les réglages et les mesures.

Le serveur choisit MPS, CUDA ou CPU selon la machine. Sur MPS, une courte préparation des blocs précède le message « Interface prête ». Le brouillon nano est chargé uniquement lorsqu'il est sélectionné.

Pour utiliser d'autres fichiers :

```bash
python scripts/web_ui.py \
  --baguette-source /chemin/vers/Baguette \
  --target-checkpoint /chemin/vers/modele.pt \
  --tokenizer /chemin/vers/tokenizer.json
```

Arrêter le serveur avec `Ctrl+C`. Si le port est déjà utilisé, fermer l'ancienne instance ou choisir `--port 8766`. Toutes les options sont accessibles avec `--help`.

## Résultats mesurés

Extraits des **60 comparaisons sur Apple MPS**, avec blocs de 8 tokens. Les chiffres sont des médianes de trois rapports de débit ; chaque sortie est identique token par token à celle de Baguette seul dans ces essais.

| Cas | Mode | Tokens au plus | Vitesse / Baguette seul |
| --- | --- | ---: | ---: |
| Copie d'un paragraphe | Greedy | 64 | 4,43× |
| CSV répété | Greedy | 64 | 3,45× |
| Copie d'un paragraphe | Échantillonnage | 64 | 1,56× |
| « Bonjour, je suis » | Échantillonnage | 64 | 0,98× |

**Le gain n'est pas systématique.** Les copies et répétitions sont les cas favorables. Sur du texte libre, le mode automatique peut rester proche de la référence ou être plus lent. Les répétitions du modèle peuvent être accélérées sans devenir de meilleures réponses.

Le [rapport adaptatif](bench_reports/adaptive-lookup-mps-20261028.md) détaille les conditions, les cas sans gain et la comparaison avec l'ancienne version. Les autres essais sont regroupés dans l'[index des rapports](bench_reports/README.md).

## Organisation

```text
specdec/        Moteurs de génération, sampling et cache d'attention
scripts/        Interface locale, benchmarks et entraînement
web/            Interface HTML, CSS et JavaScript
configs/        Exemple de grille d'expériences
tests/         Tests unitaires et intégration avec Baguette
bench_reports/  Rapports et résultats bruts
docs/          Fonctionnement, commandes et développement
```

- [Fonctionnement du décodage](docs/decoding.md)
- [Benchmarks et entraînement du brouillon](docs/experiments.md)
- [Développement et vérifications](docs/development.md)

Le cache nécessite actuellement un modèle Baguette avec `hybrid=False`. Les couches DeltaNet ne sont pas encore prises en charge par le moteur avec cache.

## Développement

```bash
python -m pip install -r requirements-dev.txt
ruff check .
ruff format --check .
BAGUETTE_SOURCE=../LLM python -m unittest discover -s tests -v
```

Les tests couvrent les acceptations, les rejets, les tokens de fin, les tirages et le cache. Aucun checkpoint n'est nécessaire : les tests d'intégration construisent de petits modèles depuis le code Baguette. GitHub Actions lance ces vérifications sur CPU.
