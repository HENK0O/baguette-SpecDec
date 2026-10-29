# Fonctionnement du décodage

La génération classique produit un token à la fois avec le modèle cible. Le mode automatique utilise le [prompt lookup decoding](https://huggingface.co/docs/transformers/assisted_decoding#prompt-lookup-decoding) : un index de petits groupes de tokens retrouve des continuations dans le prompt et le texte déjà généré. Deux tokens sont d'abord contrôlés avec des étapes classiques pour éviter de lancer un gros bloc sur une correspondance fragile. Après deux blocs entièrement acceptés, ce contrôle préalable passe à un token : les autres tokens restent vérifiés ensemble par Baguette. Au premier écart, le moteur revient aux deux contrôles. Sur un prompt court, la recherche est relancée après 32 tokens générés, puis tous les 16 tokens tant qu'aucune suite réutilisable n'est trouvée.

Les tokens sont toujours tirés dans les distributions de Baguette. Tant qu'ils correspondent aux propositions copiées, le bloc peut être conservé ; au premier écart, le token tiré devient la correction et la suite du bloc est abandonnée. Le cache est remis à la bonne position. Ce mécanisme conserve la distribution cible, en greedy comme en échantillonnage.

Avec le brouillon entraîné, l'acceptation suit la règle `min(1, p(x) / q(x))`, où `p` est la probabilité donnée par la cible et `q` celle du brouillon. Une distribution résiduelle corrige les rejets. Les deux moteurs utilisent le cache des clés et valeurs de l'attention pour éviter de recalculer tout le préfixe.


## Où lire le code

| Fichier | Rôle |
| --- | --- |
| [`baseline.py`](../specdec/baseline.py) | Génération classique sans cache, utilisée comme référence simple. |
| [`cached_generation.py`](../specdec/cached_generation.py) | Génération classique et spéculative avec cache. |
| [`lookup.py`](../specdec/lookup.py) | Index des suites réutilisables, contrôles adaptatifs et repli. |
| [`speculative.py`](../specdec/speculative.py) | Acceptation et correction avec un modèle brouillon. |
| [`sampling.py`](../specdec/sampling.py) | Température, top-k, top-p et tirage des tokens. |
| [`cache.py`](../specdec/cache.py) | Écriture et remise à la bonne position du cache d'attention. |
| [`loader.py`](../specdec/loader.py) | Chargement des poids et vérification des tokenizers. |

## Limites actuelles

Le cache prend en charge les modèles Baguette avec `hybrid=False`. La gestion de l'état récurrent des couches DeltaNet reste à faire. Le chargeur utilise l'architecture `model.py` du dépôt Baguette ; ce projet n'est pas un moteur universel pour tous les modèles Hugging Face.

Le mode automatique dépend des suites déjà présentes dans le texte. Sans suite utile, la génération reste classique, avec un surcoût de gestion possible. Les poids du modèle ne changent pas : ses erreurs et répétitions restent présentes.

[Retour au projet](../README.md)
