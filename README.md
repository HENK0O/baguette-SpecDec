# baguette-SpecDec

Un moteur de décodage spéculatif pour [Baguette](https://github.com/HENK0O/baguette), développé étape par étape. Les modèles, poids et tokenizer sont fournis par des chemins externes ; ce dépôt ne les copie pas.

## Phase 1 : référence autorégressive

Pour chaque nouveau token, on donne au modèle tous les tokens déjà présents. Le dernier vecteur de logits, de taille `[vocabulaire]`, définit la distribution du prochain token. En mode greedy, on prend son maximum. Sinon, on divise par la température, on filtre éventuellement avec top-k et top-p, on normalise par softmax puis on tire un token. La graine fixe le tirage pour rendre une exécution reproductible.

Cette version recalcule le préfixe à chaque étape : elle est simple à contrôler, mais n'est pas encore optimisée par cache KV. Les mesures couvrent la génération uniquement, après chargement du modèle. Le temps du premier token comprend le premier passage du modèle et le tirage. Le débit est `tokens générés / durée totale` ; il dépend du matériel, de la longueur du prompt et des options de sampling.

```bash
pip install -r requirements.txt
python scripts/baseline.py \
  --baguette-source .. \
  --target-checkpoint ../baguette-123m-sft.pt \
  --tokenizer ../tokenizer.json \
  --prompt "Bonjour, je suis" \
  --max-new-tokens 16 --greedy
```

Le checkpoint doit contenir les clés `model_cfg` et `model`, comme l'export de Baguette. `--baguette-source` désigne le dossier contenant son `model.py`. Seuls des checkpoints de confiance doivent être chargés. Le vocabulaire du tokenizer est vérifié contre la configuration du modèle. Le programme refuse une génération qui dépasse la fenêtre de contexte.

Pour le sampling, utiliser par exemple `--temperature 0.5 --top-k 20 --top-p 0.95 --seed 42`. `--greedy` ignore les réglages de sampling. Le programme affiche du JSON avec le texte produit et les trois mesures de temps/débit.

Lancer les tests de cette phase :

```bash
python -m unittest discover -s tests -v
```

## Suite prévue

1. Charger un modèle brouillon plus petit avec le **même tokenizer et les mêmes IDs**.
2. Implémenter l'acceptation probabiliste `min(1, p(x)/q(x))` et, au rejet, échantillonner la partie positive normalisée de `p-q` ; vérifier statistiquement que la loi du modèle cible est conservée.
3. Vérifier un bloc entier avec un seul appel cible quand l'architecture le permet, puis ajouter les caches KV des deux modèles et tester l'équivalence avec et sans cache.
4. Mesurer le débit, le temps au premier token, le taux d'acceptation, les appels cible et le coût du brouillon pour plusieurs tailles de blocs et réglages. Les résultats seront des mesures réelles exportées en JSON/CSV.

Le modèle Baguette peut aussi utiliser des couches DeltaNet (`hybrid=True`). Leur état récurrent demande une gestion de cache différente de l'attention classique ; cette contrainte sera examinée avant l'implémentation du cache ou du retour arrière après rejet.
