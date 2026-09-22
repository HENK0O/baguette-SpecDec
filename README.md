# baguette-SpecDec

`baguette-SpecDec` implémente le **décodage spéculatif** pour [Baguette](https://github.com/HENK0O/baguette). Un petit modèle propose plusieurs tokens, puis le modèle principal les vérifie pour essayer d'accélérer la génération.

Le code fonctionne et les premiers tests sont reproductibles. Pour l'instant, la version spéculative est encore plus lente que la génération classique sur la machine utilisée. Le projet sert donc aussi à comprendre *pourquoi* une technique prometteuse sur le papier ne donne pas automatiquement un gain de vitesse en pratique.

## Comment ça marche ?

La génération classique produit un token à la fois avec le modèle cible. Avec le décodage spéculatif, un petit modèle appelé **brouillon** propose plusieurs tokens. Le modèle cible les vérifie ensuite en un seul passage. Les propositions acceptées sont conservées ; au premier refus, un token est tiré à partir d'une distribution de correction.

L'acceptation suit la règle `min(1, p(x) / q(x))`, où `p` est la probabilité donnée par la cible et `q` celle du brouillon. Cette correction permet de garder la même distribution de sortie que le modèle cible, même quand le brouillon se trompe. Le code utilise aussi un cache des clés et valeurs de l'attention pour éviter de recalculer tout le début du texte à chaque étape.

## Ce qui est déjà fait

- Une génération classique pour servir de référence, avec ou sans cache.
- Une implémentation du décodage spéculatif avec échantillonnage et correction exacte.
- Des scripts pour préparer les données, entraîner un brouillon compatible et comparer les vitesses.
- Des tests sur la correction, l'alignement des logits et le cache.

Le cache fonctionne actuellement avec les modèles Baguette configurés avec `hybrid=False`. La gestion de l'état des couches DeltaNet reste à faire.

## Essayer le projet

Les commandes ci-dessous partent du dossier `baguette-SpecDec` et supposent que le dépôt Baguette se trouve juste à côté, dans `../LLM`. Les poids des modèles ne sont pas inclus dans ce dépôt GitHub.

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Tester d'abord la génération classique :

```bash
python scripts/baseline.py \
  --baguette-source ../LLM \
  --target-checkpoint ../LLM/baguette-123m-sft.pt \
  --tokenizer ../LLM/tokenizer.json \
  --prompt "Bonjour, je suis" \
  --max-new-tokens 32 --cache
```

Puis comparer avec le décodage spéculatif, si le brouillon entraîné est disponible localement :

```bash
python scripts/benchmark.py \
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

Les deux modèles doivent utiliser **exactement le même tokenizer**. Employer le même checkpoint pour la cible et le brouillon peut aider à vérifier le fonctionnement du code, mais ne donne pas une comparaison de vitesse utile.

## Premier résultat

Un brouillon `nano` de **16,6 millions de paramètres** a été distillé à partir du modèle cible Baguette de **123 millions de paramètres**. Les essais ont été faits sur Apple MPS avec le cache activé, trois prompts, trois graines par prompt et jusqu'à 64 nouveaux tokens.

| Méthode | Débit médian | Vitesse relative | Tokens proposés acceptés |
| --- | ---: | ---: | ---: |
| Cible seule | 63,81 tokens/s | 1,00× | — |
| Spéculatif, blocs de 2 | 40,18 tokens/s | 0,653× | 26,5 % |
| Spéculatif, blocs de 4 | 30,91 tokens/s | 0,536× | 13,3 % |

Le petit modèle n'est donc pas encore assez efficace dans cette configuration pour compenser le coût de ses propositions et de leur vérification. Ces chiffres décrivent seulement cette machine et ce jeu de prompts. Les mesures détaillées sont dans le [mini rapport de benchmark](bench_reports/nano-distilled-step3000-mps.md), avec les résultats bruts en [CSV](bench_reports/nano-distilled-step3000-mps.csv) et en [JSON](bench_reports/nano-distilled-step3000-mps.json).

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

La suite consiste surtout à améliorer le brouillon et à refaire les mesures sur davantage de prompts. Un meilleur taux d'acceptation pourrait rendre la méthode intéressante, mais seul un nouveau benchmark permettra de vérifier si elle accélère réellement Baguette.
