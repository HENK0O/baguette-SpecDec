# Benchmarks et entraînement du brouillon

Les commandes se lancent depuis la racine du dépôt. Elles supposent que le code et les poids Baguette se trouvent dans `../LLM` ; les chemins peuvent être remplacés par les options correspondantes.

## Comparer les méthodes en ligne de commande

Tester d'abord la génération classique :

```bash
python scripts/baseline.py \
  --baguette-source ../LLM \
  --target-checkpoint ../LLM/baguette-123m-sft.pt \
  --tokenizer ../LLM/tokenizer.json \
  --prompt "Bonjour, je suis" \
  --max-new-tokens 32 --cache
```

Comparer avec le mode automatique, sans checkpoint assistant :

```bash
python scripts/benchmark.py \
  --strategy auto \
  --baguette-source ../LLM \
  --target-checkpoint ../LLM/baguette-123m-sft.pt \
  --tokenizer ../LLM/tokenizer.json \
  --prompt "Bonjour, je suis" \
  --max-new-tokens 64 --draft-lengths 8 \
  --temperature 0.5 --top-k 20 --top-p 0.95
```

Le cache est obligatoire et activé automatiquement pour cette stratégie. Pour retrouver l'expérience avec le brouillon entraîné :

```bash
python scripts/benchmark.py \
  --strategy draft \
  --baguette-source ../LLM \
  --target-checkpoint ../LLM/baguette-123m-sft.pt \
  --draft-checkpoint ../LLM/runs/specdec-draft-nano/distill-run1/draft-step3000.pt \
  --tokenizer ../LLM/tokenizer.json \
  --draft-tokenizer ../LLM/runs/specdec-draft-nano/tokenizer.json \
  --prompt "Bonjour, je suis" \
  --max-new-tokens 64 --draft-lengths 2 4 \
  --temperature 0.5 --top-k 20 --top-p 0.95 --cache
```

La sortie indique notamment le débit en tokens par seconde, le temps avant le premier token, le taux d'acceptation et le rapport de vitesse par rapport à la génération classique. Un rapport supérieur à `1` signifie que la version spéculative est plus rapide.

Pour comparer les deux méthodes en génération déterministe, ajouter `--greedy` à la commande de benchmark. Dans ce mode, les réglages de température, top-k et top-p sont ignorés. Le benchmark vérifie aussi que la sortie spéculative est identique à celle de la cible seule. L'option `--greedy` fonctionne également avec `scripts/experiments.py`.

Avec un assistant entraîné, les deux modèles doivent utiliser **exactement le même tokenizer**. Employer le même checkpoint pour la cible et le brouillon peut aider à vérifier le fonctionnement du code, mais ne donne pas une comparaison de vitesse utile.


## Historique du brouillon nano

Un brouillon `nano` de **16,6 millions de paramètres** a été distillé à partir du modèle cible Baguette de **123 millions de paramètres**. Les essais ont été faits sur Apple MPS avec le cache activé, trois prompts, trois graines par prompt et jusqu'à 64 nouveaux tokens.

| Méthode | Débit médian | Vitesse relative | Tokens proposés acceptés |
| --- | ---: | ---: | ---: |
| Cible seule | 63,81 tokens/s | 1,00× | — |
| Spéculatif, blocs de 2 | 40,18 tokens/s | 0,653× | 26,5 % |
| Spéculatif, blocs de 4 | 30,91 tokens/s | 0,536× | 13,3 % |

Le petit modèle n'est donc pas encore assez efficace dans cette configuration pour compenser le coût de ses propositions et de leur vérification. Ces chiffres décrivent seulement cette machine et ce jeu de prompts. Les mesures détaillées sont dans le [mini rapport de benchmark](../bench_reports/nano-distilled-step3000-mps.md), avec les résultats bruts en [CSV](../bench_reports/nano-distilled-step3000-mps.csv) et en [JSON](../bench_reports/nano-distilled-step3000-mps.json).

### Optimisation du moteur avec cache

Le moteur évite maintenant un appel au brouillon par bloc, réutilise les masques d'attention et calcule uniquement les logits et les probabilités utiles. Le cache est aussi limité à la longueur de la requête.

Sur 18 essais MPS du 28 octobre 2026, le débit spéculatif augmente de **24 à 25 % par rapport à la version précédente**, avec les mêmes tokens générés. Baguette seul est également optimisé et reste plus rapide avec le brouillon actuel. Le [rapport avant/après](../bench_reports/cached-optimization-20261028.md) contient le protocole, les mesures et la commande pour les reproduire. Ces améliorations sont utilisées par le benchmark et l'interface lorsque le cache est activé.

## Entraîner un brouillon compatible

Le corpus Baguette déjà présent dans `../LLM/data/train.bin` utilise un autre tokenizer. Il faut donc recréer un corpus avec celui du modèle cible avant d'entraîner le brouillon :

```bash
python scripts/prepare_draft_corpus.py \
  --baguette-source ../LLM \
  --target-checkpoint ../LLM/baguette-123m-sft.pt \
  --tokenizer ../LLM/tokenizer.json \
  --output-dir ../LLM/data/draft-123m-tokenizer
```

Le dossier de sortie doit être vide. Le corpus réencodé a été supprimé après les essais pour économiser de la place ; cette commande permet de le recréer à partir des textes sources locaux.

Pour lancer une distillation :

```bash
python scripts/distill_draft.py \
  --baguette-source ../LLM \
  --target-checkpoint ../LLM/baguette-123m-sft.pt \
  --tokenizer ../LLM/tokenizer.json \
  --data-dir ../LLM/data/draft-123m-tokenizer \
  --out-dir ../LLM/runs/specdec-draft-nano/distill \
  --steps 3000 --batch-size 8 --seq-len 128 --device auto
```

Les poids et les données restent dans le dossier local `LLM` et ne sont pas publiés dans ce dépôt. Le fichier [`configs/example.json`](../configs/example.json) donne aussi un exemple de grille pour lancer plusieurs expériences avec `scripts/experiments.py`.


## Reproduire les mesures adaptatives

```bash
python scripts/benchmark_lookup_revision.py \
  --baguette-source ../LLM \
  --target-checkpoint ../LLM/baguette-123m-sft.pt \
  --tokenizer ../LLM/tokenizer.json \
  --previous-ref 6f8cc74 --device mps \
  --output results/adaptive-recheck.json
```

Ce benchmark compare Baguette seul, l'ancien moteur et le moteur courant. La révision historique doit être présente dans l'historique Git. Les mesures enregistrent une empreinte des sources ; un simple formatage du Python change cette empreinte sans constituer une nouvelle mesure de performance.

Les protocoles et données brutes restent accessibles dans l'[index des rapports](../bench_reports/README.md).

[Retour au projet](../README.md)
