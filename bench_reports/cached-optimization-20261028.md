# Optimisation du moteur avec cache

Mesure du 28 octobre 2026 : comparaison de `c683099` et du moteur `d5e5468`, avec les mêmes checkpoints. Les résultats bruts sont disponibles pour l'[échantillonnage](cached-optimization-20261028-sampling.json) et le [mode greedy](cached-optimization-20261028-greedy.json).

## Résultat mesuré

Le débit spéculatif augmente de **24 à 25 % par rapport à l'ancienne version** dans ces essais MPS. La cible seule profite aussi des optimisations et reste plus rapide que le couple cible + brouillon.

| Bloc | Ancien débit médian | Nouveau débit médian | Gain médian par essai | Nouveau débit / cible seule |
| --- | ---: | ---: | ---: | ---: |
| K=2 | 41,40 tokens/s | 50,58 tokens/s | 1,239× | 0,683× |
| K=4 | 31,89 tokens/s | 39,71 tokens/s | 1,250× | 0,541× |

Le gain est la médiane des rapports nouveau/ancien calculés pour chaque essai, ce qui peut différer du rapport des deux débits médians. La référence cible seule est également mise à jour : son débit augmente d'environ 14 à 15 %. Les rapports à la cible utilisent cette nouvelle référence.

## Changements

- Le brouillon n'effectue plus de passage après sa dernière proposition avant la vérification. Après une acceptation complète, cette proposition et le token bonus sont traités ensemble. Après un rejet, seul le token de correction est traité.
- Le masque causal est construit une fois pour toutes les couches. Un token seul peut accéder directement à tout son passé ; le préremplissage utilise le masque causal natif de l'attention.
- Le préremplissage et le rattrapage du brouillon projettent uniquement le dernier état vers le vocabulaire. La vérification conserve tous les logits nécessaires.
- Les distributions de la cible sont calculées à la demande. Les propositions situées après un rejet n'ont plus besoin de leur transformation en probabilités.
- Le cache est dimensionné au prompt et au budget de génération.

Le nombre de passages du brouillon passe de `propositions + blocs` à `propositions`. La baisse médiane est de 33,6 % pour K=2 et de 20,4 % pour K=4. Par exemple, le premier essai K=2 conserve les mêmes 64 tokens en passant de 138 à 92 appels au brouillon.

Pour le prompt « Bonjour, je suis » (4 tokens) et un budget de 64 nouveaux tokens, les deux caches KV FP32 occupent **2,52 Mio au lieu de 19 Mio** : 68 positions sont allouées au lieu de 512. Cette estimation concerne uniquement les clés et valeurs du cache, calculées à partir des dimensions des deux modèles.

## Protocole et correction

- Cible : Baguette SFT, 122 710 784 paramètres.
- Brouillon : nano distillé, 16 619 136 paramètres, même tokenizer.
- Apple MPS, PyTorch 2.8.0 ; poids et précision inchangés entre les versions.
- Trois prompts courts, graines 0, 1 et 2, au plus 64 nouveaux tokens, K=2 et K=4 : 18 comparaisons appariées en échantillonnage.
- Température 0,5, top-k 20, top-p 0,95 ; un échauffement complet par méthode, prompt et taille de bloc. L'ordre des quatre méthodes (ancienne/nouvelle cible et ancien/nouveau spéculatif) alterne entre les essais.
- Mesures de génération après chargement des modèles. L'allocation initiale des caches est exclue du chronométrage dans les deux versions.

Les **18 sorties spéculatives et les 18 sorties de référence** sont identiques avant et après, token par token. Une vérification greedy supplémentaire couvre trois prompts et K=1, 2 et 4 sur 32 tokens : les neuf sorties spéculatives sont identiques à la cible seule et à l'ancienne version. Le meilleur rapport greedy mesuré ici est obtenu avec K=1 : 0,811× par rapport à la cible seule, encore inférieur à 1.

Les 28 tests passent. Les nouveaux cas vérifient les projections du dernier token, la conservation du cache, les rejets avec plusieurs graines, un EOS proposé puis rejeté, les tokens bonus et la fin exacte de la fenêtre de contexte.

Ces mesures restent limitées à ces modèles, à cette machine et à ces prompts. Elles démontrent une amélioration du moteur, mais aucun dépassement de la cible seule avec ce brouillon. L'acceptation faible reste un frein à examiner avec un brouillon mieux entraîné.

Les champs `target_seconds` et `draft_seconds` du moteur avec cache comptent maintenant les passages de modèle uniquement. L'ancien `target_seconds` incluait aussi les transformations de tous les logits en probabilités ; les gains du tableau utilisent donc les durées totales, comparables entre les versions.

## Refaire la comparaison

Depuis le dépôt, avec les poids locaux disponibles :

```bash
python scripts/benchmark_cached_revision.py \
  --reference c683099 \
  --baguette-source ../LLM \
  --target-checkpoint ../LLM/baguette-123m-sft.pt \
  --draft-checkpoint ../LLM/runs/specdec-draft-nano/distill-run1/draft-step3000.pt \
  --tokenizer ../LLM/tokenizer.json \
  --draft-tokenizer ../LLM/runs/specdec-draft-nano/tokenizer.json \
  --device mps --output results/recheck-sampling.json
```

Pour le contrôle greedy, ajouter `--greedy --max-new-tokens 32 --draft-lengths 1 2 4 --seeds 0` et choisir un autre fichier de sortie. Le script recharge uniquement `cache.py` et `cached_generation.py` de la révision Git indiquée. Les fonctions d'échantillonnage et les structures de résultat partagées sont identiques dans les deux versions comparées. Les fichiers JSON enregistrent l'empreinte des noyaux optimisés, les réglages, l'ordre des essais, les tokens et les temps bruts.
