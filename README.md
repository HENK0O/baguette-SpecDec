# baguette-SpecDec

Un moteur de décodage spéculatif pour [Baguette](https://github.com/HENK0O/baguette), développé étape par étape. Les modèles, poids et tokenizer sont fournis par des chemins externes ; ce dépôt ne les copie pas.

## Phase 1 : référence autorégressive

Pour chaque nouveau token, on donne au modèle tous les tokens déjà présents. Le dernier vecteur de logits, de taille `[vocabulaire]`, définit la distribution du prochain token. En mode greedy, on prend son maximum. Sinon, on divise par la température, on filtre éventuellement avec top-k et top-p, on normalise par softmax puis on tire un token. La graine fixe le tirage pour rendre une exécution reproductible.

Sans `--cache`, cette version recalcule le préfixe à chaque étape : elle est simple à contrôler. Avec `--cache`, elle réutilise les clés et valeurs des couches d'attention. Les mesures couvrent la génération uniquement, après chargement du modèle. Le temps du premier token comprend le premier passage du modèle et le tirage. Le débit est `tokens générés / durée totale` ; il dépend du matériel, de la longueur du prompt et des options de sampling.

```bash
pip install -r requirements.txt
python scripts/baseline.py \
  --baguette-source ../LLM \
  --target-checkpoint ../LLM/baguette-123m-sft.pt \
  --tokenizer ../LLM/tokenizer.json \
  --prompt "Bonjour, je suis" \
  --max-new-tokens 16 --greedy
```

Le checkpoint doit contenir les clés `model_cfg` et `model`, comme l'export de Baguette. `--baguette-source` désigne le dossier contenant son `model.py`. Seuls des checkpoints de confiance doivent être chargés. Le vocabulaire du tokenizer est vérifié contre la configuration du modèle. Le programme refuse une génération qui dépasse la fenêtre de contexte.

Pour le sampling, utiliser par exemple `--temperature 0.5 --top-k 20 --top-p 0.95 --seed 42`. `--greedy` ignore les réglages de sampling. Le programme affiche du JSON avec le texte produit et les trois mesures de temps/débit.

## Décodage spéculatif exact

Le brouillon tire une suite de `K` tokens dans ses distributions `q_i`. Un seul passage cible sur le préfixe et cette suite donne les distributions `p_i` correspondant à chaque proposition, puis celle du token supplémentaire. Le décalage est important : si le préfixe contient `T` tokens, `logits[T-1+i]` prédit la proposition numéro `i+1`.

Pour une proposition `x`, on l'accepte avec la probabilité `min(1, p_i(x)/q_i(x))`. Si elle est rejetée, le token de correction vient de `r_i(x) = max(0, p_i(x)-q_i(x)) / Σ_y max(0, p_i(y)-q_i(y))`. La masse conservée par acceptation est `min(p_i,q_i)`. La masse de rejet redistribuée par `r_i` est `(p_i-q_i)_+`. Leur somme vaut donc `p_i` pour chaque token : le tirage final suit exactement la loi cible. Les deux modèles appliquent les mêmes règles de température, top-k et top-p avant ce calcul.

Le cache stocke, pour chaque couche d'attention, K et V de forme `[batch, n_kv_head, max_len, head_dim]`. Après un rejet, les entrées après le préfixe accepté sont simplement écrasées par le token de correction. Pendant la vérification d'un bloc, un masque causal explicite empêche chaque proposition de voir les tokens suivants. Le cache est actuellement réservé aux modèles Baguette `hybrid=False` : les couches DeltaNet ont un état récurrent qui demande un retour arrière distinct.

```bash
python scripts/benchmark.py \
  --baguette-source /chemin/vers/baguette \
  --target-checkpoint /chemin/vers/baguette-123m-sft.pt \
  --draft-checkpoint /chemin/vers/baguette-draft.pt \
  --tokenizer /chemin/vers/baguette/tokenizer.json \
  --draft-tokenizer /chemin/vers/draft/tokenizer.json \
  --prompt "Bonjour, je suis" --max-new-tokens 64 \
  --draft-lengths 2 4 6 8 --cache --output results/example.json
```

Les deux fichiers tokenizer doivent être identiques octet par octet, et les tailles de vocabulaire doivent correspondre aux checkpoints. Les checkpoints Baguette ne contiennent pas d'empreinte du tokenizer : il faut donc fournir celui réellement utilisé pour entraîner chaque modèle. Les petits checkpoints actuellement présents dans le dossier Baguette local ont des tokenizers différents et **ne conviennent pas** comme brouillons du modèle 123M. Un vrai brouillon plus petit, entraîné avec son tokenizer, est nécessaire pour mesurer un éventuel gain de vitesse. Le même checkpoint des deux côtés peut servir à vérifier le code, mais ne constitue pas une mesure de performance pertinente.

Le benchmark rapporte débit, temps au premier token, accélération, taux d'acceptation, moyenne de tokens acceptés par bloc, nombre de passages cible/brouillon et temps passé dans chacun. Les valeurs proviennent d'exécutions réelles ; elles varient d'un lancement à l'autre. `--cache` active le cache pour les deux modèles ; sans lui, on obtient la référence simple qui recalcule le préfixe.

Pour plusieurs prompts, tailles de brouillon, longueurs de génération, températures, top-p et graines, modifier [`configs/example.json`](configs/example.json), puis lancer :

```bash
python scripts/experiments.py \
  --baguette-source /chemin/vers/baguette \
  --target-checkpoint /chemin/vers/baguette-123m-sft.pt \
  --tokenizer /chemin/vers/baguette/tokenizer.json \
  --config configs/example.json --cache \
  --output-csv results/experiments.csv \
  --output-json results/experiments.json
```

Chaque ligne est un essai brut avec ses réglages et ses mesures. Répéter plusieurs graines et examiner la dispersion avant de résumer un gain. Le cache exige des modèles Baguette sans DeltaNet.

### Premier résultat mesuré

Un brouillon `nano` de 16,6 M paramètres, distillé sur 3,07 M tokens, a été testé sur Apple MPS avec trois prompts et trois graines. Pour K=2, l'acceptation médiane est de 26,5 % et l'accélération médiane de **0,653×** : cette version est encore plus lente que la cible seule. Voir le [compte rendu et les essais bruts](bench_reports/nano-distilled-step3000-mps.md). Aucun gain de vitesse n'est revendiqué à ce stade.

Lancer les tests de cette phase :

```bash
BAGUETTE_SOURCE=/chemin/vers/baguette python -m unittest discover -s tests -v
```

## Suite prévue

1. Poursuivre l'entraînement et la distillation d'un brouillon plus petit avec **le même tokenizer et les mêmes IDs** que le modèle cible. Les fichiers `LLM/data/train.bin` présents avant ce projet utilisent un autre tokenizer ; `scripts/prepare_draft_corpus.py` recrée le corpus à partir des textes bruts avec celui du modèle cible, dans un nouveau dossier.
2. Exécuter la grille d'expériences sur de vrais brouillons compatibles et analyser la dispersion et le coût par configuration. Aucun gain de vitesse n'est revendiqué avant ces mesures.
3. Étudier le retour arrière de l'état DeltaNet avant de prendre en charge les modèles hybrides.

Les tests couvrent la normalisation, la reproductibilité, la correction résiduelle, une vérification statistique sur un modèle jouet, l'alignement des logits, et l'équivalence du cache avec un passage complet. Les tests d'intégration du cache sont ignorés si `BAGUETTE_SOURCE` ne pointe pas sur le dépôt Baguette.

## Préparer et entraîner un brouillon compatible

La préparation suivante ne retélécharge rien et n'écrase pas le corpus Baguette existant :

```bash
python scripts/prepare_draft_corpus.py \
  --baguette-source ../LLM \
  --target-checkpoint ../LLM/baguette-123m-sft.pt \
  --tokenizer ../LLM/tokenizer.json \
  --output-dir ../LLM/data/draft-123m-tokenizer
```

`manifest.json` enregistre l'empreinte du tokenizer et les sources. Le dossier de sortie doit être vide avant le lancement. Pour entraîner directement par distillation, sans checkpoint de départ :

```bash
python scripts/distill_draft.py \
  --baguette-source ../LLM \
  --target-checkpoint ../LLM/baguette-123m-sft.pt \
  --tokenizer ../LLM/tokenizer.json \
  --data-dir ../LLM/data/draft-123m-tokenizer \
  --out-dir ../LLM/runs/specdec-draft-nano/distill \
  --steps 3000 --batch-size 8 --seq-len 128 --device auto
```

La perte principale est `KL(cible || brouillon)` sur la distribution du prochain token ; une petite part de cross-entropy utilise le token réel du corpus. Les checkpoints de poids `draft-stepN.pt` peuvent être utilisés directement par `scripts/benchmark.py` avec `--draft-tokenizer ../LLM/data/draft-123m-tokenizer/tokenizer.json`. Les poids et les jeux de données restent hors de ce dépôt GitHub.
