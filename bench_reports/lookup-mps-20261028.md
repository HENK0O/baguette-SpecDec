# Mode automatique : accélérer les reprises de texte

Mesures du 28 octobre 2026 sur Apple MPS. Les [essais bruts](lookup-mps-20261028.json) contiennent les prompts complets, les ordres de passage, les tokens et les durées. Le script de reproduction est [`benchmark_lookup.py`](../scripts/benchmark_lookup.py).

## Résultats

Le mode automatique obtient un gain sur les cas de copie et de texte répétitif testés. Les prompts libres courts utilisent la cible seule, avec un petit surcoût de gestion. Les **42 sorties automatiques correspondent à la référence token par token**, en échantillonnage et en greedy.

| Prompt | Automatique / cible, échantillonnage | Automatique / cible, greedy |
| --- | ---: | ---: |
| Bonjour, je suis | 0,999× | 0,969× |
| Explication de la pluie | 0,986× | 0,970× |
| Organisation d'un projet Python | 0,984× | 0,964× |
| Copie du paragraphe sur le vélo | 1,234× | 1,202× |
| Copie du paragraphe sur la bibliothèque | 1,474× | 3,071× |
| Code répété | 1,294× | 2,048× |
| Tableau CSV répété | 1,631× | 2,837× |

Chaque cellule est la médiane de trois rapports appariés. En échantillonnage, les graines sont 0, 1 et 42 ; en greedy, elles servent à répéter les mesures et ne changent pas le texte.

Pour la copie « bibliothèque » en échantillonnage, les débits médians sont **71,94 tokens/s pour Baguette seul** et **106,07 tokens/s en automatique**. En greedy, ils atteignent respectivement **81,82 et 251,25 tokens/s**. Le brouillon nano reste plus lent que la cible seule dans toutes les lignes de ce benchmark.

Les cas libres ne démontrent aucun gain spéculatif : le moteur y effectue zéro vérification par blocs et zéro appel au brouillon. Les rapports individuels vont de 0,958× à 1,002× sur ces cas. Ce repli évite le ralentissement important de l'ancien assistant, mais ne garantit pas une absence totale de surcoût.

## Pourquoi ce mode peut gagner

Le [prompt lookup decoding](https://huggingface.co/docs/transformers/assisted_decoding#prompt-lookup-decoding) utilise le texte disponible comme source de propositions. Ici, un index CPU retrouve des groupes de 2 à 4 tokens dans le prompt et dans la sortie déjà produite.

Deux tokens sont d'abord tirés avec des étapes ordinaires de Baguette. S'ils correspondent à une continuation connue, la suite peut être vérifiée en bloc. Sinon, aucune grosse vérification n'est lancée. Après une vérification peu productive, un délai croissant de 8 à 64 tokens suspend les tentatives. Un prompt de moins de 32 tokens sans continuation déjà disponible reste entièrement en génération classique.

Le brouillon entraîné n'est donc pas nécessaire à cette stratégie. Son chargement dans l'interface est différé jusqu'à sa sélection explicite.

Dans l'essai bibliothèque greedy de graine 42, 64 tokens nécessitent **17 passages de Baguette et zéro passage de brouillon**, contre 64 passages pour la référence. Le nano demande 54 passages cible et 105 passages brouillon sur le même prompt.

## Correction du tirage

Une proposition copiée est déterministe. Pour une proposition `c`, tirer `y` dans la distribution cible puis accepter si `y == c` équivaut à une acceptation de probabilité `p(c)`. En cas d'écart, `y` suit déjà la distribution résiduelle conditionnée par `y != c`.

Chaque token conservé provient ainsi d'un tirage cible. Les logits suivants ne sont utilisés que si tout le préfixe proposé a été confirmé ; au premier écart, le token tiré remplace la proposition et les entrées ultérieures du cache sont ignorées. La même logique fonctionne avec l'argmax en greedy.

Le moteur ne change ni les poids ni les réglages de sampling. La qualité et les erreurs de Baguette restent celles de la référence. Certains textes bruts continuent au-delà de la consigne ou se répètent : ce benchmark mesure la vitesse des mêmes sorties, sans évaluer leur qualité.

## Protocole et limites

- Baguette SFT 122 710 784 paramètres, checkpoint local `baguette-123m-sft.pt`, FP32.
- Apple MPS, PyTorch 2.8.0, au plus 64 nouveaux tokens ; cache activé pour toutes les méthodes.
- Automatique : blocs de 8 ; nano : blocs de 2 ; température 0,5, top-k 20, top-p 0,95.
- Sept prompts, deux modes, trois graines/répétitions : 42 comparaisons à trois méthodes.
- Un échauffement complet de chaque méthode par prompt et mode. L'ordre des méthodes est alterné et inversé entre les essais.
- Le chargement des modèles et les échauffements sont exclus des débits ; la recherche de correspondances et la gestion du repli font partie du temps de génération automatique.

Ces cas forment un petit jeu de développement, pas une évaluation indépendante de grande taille. Les exemples de copie et de code ont notamment servi à explorer la méthode ; le second paragraphe et le CSV ont été ajoutés ensuite. Le paragraphe « bibliothèque » a été retenu pour le bouton de démonstration après ces mesures. Les gains sur les entrées répétitives ne doivent pas être généralisés au texte libre.

Les 38 tests passent, dont les vérifications de distribution et de graine, de rejet, de token de fin, de limite du cache, de repli et de chargement différé du brouillon.

## Vérification de l'interface

Le parcours vérifié est : choix du mode automatique et du prompt → `POST /api/compare` → génération avec le checkpoint local → restitution des textes et mesures.

| Frontière | Observation |
| --- | --- |
| Page et formulaire | Automatique, bloc 8, cache actif ; température, top-k et top-p restent visibles. |
| API | `GET /api/info` et les trois `POST /api/compare` testés renvoient HTTP 200. |
| Modèles chargés | Après les essais automatiques, `draft_parameters` vaut `null`. Après sélection du nano, il vaut 16 619 136. |
| Prompt libre | La page affiche « Aucun bloc réutilisé », les mêmes tokens et 64 / 0 passages. La variation de ratio n'est pas présentée comme un gain spéculatif. |
| Exemple de copie, premier essai de la page | 67,93 → 95,14 tokens/s, soit 1,40× ; 5 blocs vérifiés, 39 / 0 passages, mêmes tokens. |
| Ancien assistant | Le choix explicite charge le modèle et affiche ses mesures séparément. |
| Navigateur | Aucune erreur de console observée. |

L'interface effectue seulement un court échauffement de 4 tokens avant la comparaison et affiche aussi la durée complète de la requête. Les premières exécutions peuvent encore inclure des coûts de compilation MPS qui n'apparaissent pas dans les mesures après échauffement complet.

## Reproduire

```bash
python scripts/benchmark_lookup.py \
  --baguette-source ../LLM \
  --target-checkpoint ../LLM/baguette-123m-sft.pt \
  --tokenizer ../LLM/tokenizer.json \
  --include-draft \
  --draft-checkpoint ../LLM/runs/specdec-draft-nano/distill-run1/draft-step3000.pt \
  --draft-tokenizer ../LLM/runs/specdec-draft-nano/tokenizer.json \
  --device mps --output results/lookup-recheck.json
```

Pour comparer seulement la référence et l'automatique, omettre `--include-draft` et les deux chemins du brouillon. Les poids restent hors du dépôt GitHub.
