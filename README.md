# baguette-SpecDec

`baguette-SpecDec` implémente le **décodage spéculatif** pour [Baguette](https://github.com/HENK0O/baguette). Le modèle vérifie des propositions par blocs pour réduire le nombre de passages nécessaires à la génération.

Le mode **Automatique**, proposé par défaut dans l'interface, réutilise des suites déjà présentes dans le texte. Il ne charge aucun petit modèle. Sur un prompt court sans suite réutilisable, il conserve les étapes classiques de Baguette ; si les vérifications deviennent peu utiles, il les suspend temporairement.

Le mode **Brouillon entraîné** garde l'expérience avec le modèle nano. Son taux d'acceptation actuel est trop faible pour accélérer Baguette sur les essais réalisés. Les deux méthodes sont comparées à la même référence Baguette seul.

## Comment ça marche ?

La génération classique produit un token à la fois avec le modèle cible. Le mode automatique utilise le [prompt lookup decoding](https://huggingface.co/docs/transformers/assisted_decoding#prompt-lookup-decoding) : un index de petits groupes de tokens retrouve des continuations dans le prompt et le texte déjà généré. Deux tokens sont d'abord contrôlés avec des étapes classiques pour éviter de lancer un gros bloc sur une correspondance fragile. Les suites fiables sont ensuite vérifiées ensemble.

Les tokens sont toujours tirés dans les distributions de Baguette. Tant qu'ils correspondent aux propositions copiées, le bloc peut être conservé ; au premier écart, le token tiré devient la correction et la suite du bloc est abandonnée. Le cache est remis à la bonne position. Ce mécanisme conserve la distribution cible, en greedy comme en échantillonnage.

Avec le brouillon entraîné, l'acceptation suit la règle `min(1, p(x) / q(x))`, où `p` est la probabilité donnée par la cible et `q` celle du brouillon. Une distribution résiduelle corrige les rejets. Les deux moteurs utilisent le cache des clés et valeurs de l'attention pour éviter de recalculer tout le préfixe.

## Ce qui est déjà fait

- Une génération classique pour servir de référence, avec ou sans cache.
- Une implémentation du décodage spéculatif avec échantillonnage et correction exacte.
- Un mode automatique sans assistant, avec détection du texte réutilisable et repli vers la cible seule.
- Un mode déterministe `--greedy` qui vérifie que la sortie spéculative est identique à celle de la cible seule.
- Des scripts pour préparer les données, entraîner un brouillon compatible et comparer les vitesses.
- Des tests sur la correction, l'alignement des logits et le cache.

Le cache fonctionne actuellement avec les modèles Baguette configurés avec `hybrid=False`. La gestion de l'état des couches DeltaNet reste à faire.

## Essayer le projet

Les commandes ci-dessous partent du dossier `baguette-SpecDec` et supposent que le dépôt Baguette se trouve juste à côté, dans `../LLM`. Les poids des modèles ne sont pas inclus dans ce dépôt GitHub.

Installer les dépendances :

```bash
pip install -r requirements.txt
```

### Interface web locale

Une interface locale en noir et gris permet de comparer les méthodes sans saisir une commande à chaque essai. Elle affiche les textes générés, le débit, le temps avant le premier token et le taux d'acceptation des propositions.
Le bouton « Documentation » explique les réglages et les mesures directement dans la page.

```bash
python scripts/web_ui.py
```

Ouvrir ensuite `http://127.0.0.1:8765` dans un navigateur. Seuls Baguette et son tokenizer sont nécessaires au mode automatique. Le brouillon est chargé à la demande lorsqu'il est sélectionné. Pour d'autres fichiers, utiliser `--baguette-source`, `--target-checkpoint`, `--tokenizer` et, si nécessaire, `--draft-checkpoint` et `--draft-tokenizer`. Arrêter le serveur avec `Ctrl+C`.

Garder **Automatique**, une taille de bloc de **8** et le cache activé pour commencer. Le bouton **Exemple de copie** charge un cas de reprise de texte. Les prompts libres courts utilisent les étapes classiques lorsque rien n'est réutilisable ; l'interface le signale sans présenter la variation de mesure comme un gain spéculatif. La vitesse dépend du prompt, et la qualité des réponses reste celle de Baguette.

### Ligne de commande

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

## Résultats du mode automatique

Sur les 42 essais MPS du 28 octobre 2026, l'automatique conserve les mêmes tokens que Baguette seul. Les gains médians des cas de copie et de répétition vont de **1,23 à 1,63× en échantillonnage**, et de **1,20 à 3,07× en greedy**. Sur les trois prompts libres courts, le repli reste proche de la référence avec environ 0 à 4 % de surcoût. Ces résultats portent sur ce petit jeu de textes, pas sur tous les prompts.

Le [rapport du mode automatique](bench_reports/lookup-mps-20261028.md) contient les cas sans gain, les résultats bruts, la vérification de l'interface et les commandes de reproduction.

## Historique du brouillon nano

Un brouillon `nano` de **16,6 millions de paramètres** a été distillé à partir du modèle cible Baguette de **123 millions de paramètres**. Les essais ont été faits sur Apple MPS avec le cache activé, trois prompts, trois graines par prompt et jusqu'à 64 nouveaux tokens.

| Méthode | Débit médian | Vitesse relative | Tokens proposés acceptés |
| --- | ---: | ---: | ---: |
| Cible seule | 63,81 tokens/s | 1,00× | — |
| Spéculatif, blocs de 2 | 40,18 tokens/s | 0,653× | 26,5 % |
| Spéculatif, blocs de 4 | 30,91 tokens/s | 0,536× | 13,3 % |

Le petit modèle n'est donc pas encore assez efficace dans cette configuration pour compenser le coût de ses propositions et de leur vérification. Ces chiffres décrivent seulement cette machine et ce jeu de prompts. Les mesures détaillées sont dans le [mini rapport de benchmark](bench_reports/nano-distilled-step3000-mps.md), avec les résultats bruts en [CSV](bench_reports/nano-distilled-step3000-mps.csv) et en [JSON](bench_reports/nano-distilled-step3000-mps.json).

### Optimisation du moteur avec cache

Le moteur évite maintenant un appel au brouillon par bloc, réutilise les masques d'attention et calcule uniquement les logits et les probabilités utiles. Le cache est aussi limité à la longueur de la requête.

Sur 18 essais MPS du 28 octobre 2026, le débit spéculatif augmente de **24 à 25 % par rapport à la version précédente**, avec les mêmes tokens générés. Baguette seul est également optimisé et reste plus rapide avec le brouillon actuel. Le [rapport avant/après](bench_reports/cached-optimization-20261028.md) contient le protocole, les mesures et la commande pour les reproduire. Ces améliorations sont utilisées par le benchmark et l'interface lorsque le cache est activé.

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

Les poids et les données restent dans le dossier local `LLM` et ne sont pas publiés dans ce dépôt. Le fichier [`configs/example.json`](configs/example.json) donne aussi un exemple de grille pour lancer plusieurs expériences avec `scripts/experiments.py`.

## Tests et prochaines étapes

```bash
BAGUETTE_SOURCE=../LLM python -m unittest discover -s tests -v
```

Les tests couvrent notamment les acceptations et rejets, les tokens de fin, la remise en état du cache et l'égalité avec la cible sur plusieurs graines. Les prochains essais doivent élargir les textes de reprise et mesurer les cas où le mode automatique ne gagne rien. Le brouillon entraîné reste une piste de recherche à comparer séparément.
