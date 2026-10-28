# Décodage adaptatif : comparaison avec la version précédente

Mesures du 28 octobre 2026 sur Apple MPS, PyTorch 2.8.0, Baguette SFT 122 710 784 paramètres en FP32. Les [60 essais bruts](adaptive-lookup-mps-20261028.json) contiennent les tokens, durées, compteurs et ordres de passage de chaque méthode.

## Ce qui change

- Après deux blocs entièrement acceptés, le moteur passe de deux contrôles préalables à un seul. Tous les autres tokens sont toujours tirés et vérifiés avec Baguette dans un passage par bloc. Une divergence rétablit les deux contrôles ; les délais après les mauvais blocs restent actifs.
- Les prompts courts ne restent plus bloqués en génération classique pour toute la requête. Après 32 tokens générés, puis tous les 16 tokens, une recherche peut activer la réutilisation des répétitions apparues dans la sortie.
- L'API et l'interface indiquent les blocs avec contrôle réduit et la découverte de répétitions pendant la génération. Aucun nouveau checkpoint n'est ajouté.

## Résultats

Les 60 sorties de la nouvelle version sont identiques token par token à la cible seule et à la version précédente. La colonne « / précédente » mesure l'amélioration apportée ici ; la colonne « / cible » mesure le gain total par rapport à Baguette seul. Chaque cellule est la médiane de trois rapports appariés, pas un rapport entre deux médianes de débit.

| Prompt | Tokens au plus | Mode | Adaptatif / cible | Adaptatif / précédente |
| --- | ---: | --- | ---: | ---: |
| Bonjour, je suis | 64 | Échantillonnage | 0,978× | 1,016× |
| Bonjour, je suis | 64 | Greedy | 1,051× | 1,084× |
| Explication de la pluie | 64 | Échantillonnage | 0,993× | 1,142× |
| Explication de la pluie | 64 | Greedy | 1,418× | 1,469× |
| Organisation Python | 64 | Échantillonnage | 0,978× | 0,995× |
| Organisation Python | 64 | Greedy | 0,971× | 1,001× |
| Copie vélo | 64 | Échantillonnage | 1,239× | 1,018× |
| Copie vélo | 64 | Greedy | 1,241× | 1,042× |
| Copie bibliothèque | 64 | Échantillonnage | 1,559× | 1,143× |
| Copie bibliothèque | 64 | Greedy | 4,431× | 1,460× |
| Code répété | 64 | Échantillonnage | 1,289× | 0,996× |
| Code répété | 64 | Greedy | 2,336× | 1,148× |
| CSV répété | 64 | Échantillonnage | 1,720× | 1,036× |
| CSV répété | 64 | Greedy | 3,446× | 1,204× |
| Bonjour, sortie longue | 128 | Échantillonnage | 0,957× | 1,008× |
| Bonjour, sortie longue | 128 | Greedy | 1,077× | 1,112× |
| Pluie, sortie longue | 128 | Échantillonnage | 1,081× | 1,105× |
| Pluie, sortie longue | 128 | Greedy | 1,496× | 1,537× |
| Python, sortie longue | 128 | Échantillonnage | 0,984× | 0,993× |
| Python, sortie longue | 128 | Greedy | 1,153× | 1,186× |

Sur la copie bibliothèque en greedy, 64 tokens demandent **11 passages cible contre 17 auparavant et 64 pour la référence** : le débit gagne 46 % par rapport à l'ancien moteur. Sur la pluie en greedy, la détection des répétitions réduit les passages de 64 à 42 avec le budget de 64, et de 71 à 45 avec le budget de 128 (arrêt anticipé au token de fin). Les sorties et leurs éventuelles répétitions restent celles de Baguette.

## Cas sans gain et limites

Le mécanisme ne sait pas prédire une suite entièrement nouvelle : il dépend toujours du texte réutilisable. En échantillonnage, les trois prompts libres avec budget de 64 restent sans gain médian par rapport à la cible. Sur « Bonjour » avec budget de 128, le ratio médian est 0,957× ; un essai descend à 0,910×. Le code répété en échantillonnage ne s'améliore pas par rapport à la version précédente (0,996× médian).

Les écarts de temps sans réduction des passages comprennent le coût de gestion et la variation des mesures. La version précédente, la cible et la nouvelle version sont exécutées dans le même processus, mais cela ne supprime pas les variations du GPU. Aucune absence universelle de ralentissement n'est annoncée.

Ces dix configurations réutilisent les sept prompts du précédent jeu de développement, avec trois budgets longs supplémentaires sur les prompts libres. Elles ne forment pas un jeu indépendant ; il n'y a ni nouveaux modèles ni évaluation de qualité. Les bonnes mesures sur les répétitions de Baguette ne signifient pas que ces répétitions deviennent de meilleures réponses.

## Protocole et reproduction

- Trois méthodes : cible seule avec cache, ancien lookup au commit `6f8cc74`, nouveau lookup adaptatif ; blocs de 8 dans les deux moteurs spéculatifs.
- Sept prompts avec budget de 64 tokens, puis les trois prompts libres avec budget de 128 ; deux modes ; graines 0, 1 et 42. En greedy, les graines servent à répéter les mesures.
- Un échauffement complet de chaque méthode par prompt et mode, exclu du débit. L'ordre tourne sur les trois méthodes entre répétitions et est inversé en greedy.
- Température 0,5, top-k 20, top-p 0,95 ; ces réglages sont ignorés en greedy.
- Chargement exclu ; recherche CPU, gestion du cache, correction et découverte des répétitions incluses dans les durées.
- Le script charge seulement le fichier lookup de l'ancien commit avec les dépendances communes inchangées. Les empreintes du fichier ancien et du moteur courant sont enregistrées ; une modification du moteur pendant les mesures invalide l'exécution.
- 44 tests passent, dont les rejets avec contrôle réduit, l'arrêt EOS dans un bloc adaptatif, les limites de sortie, la découverte tardive et l'intégration avec un vrai cache Baguette.

```bash
python scripts/benchmark_lookup_revision.py \
  --baguette-source ../LLM \
  --target-checkpoint ../LLM/baguette-123m-sft.pt \
  --tokenizer ../LLM/tokenizer.json \
  --previous-ref 6f8cc74 --device mps \
  --output results/adaptive-recheck.json
```

L'historique Git contenant la référence est nécessaire. Les essais de l'ancien moteur restent documentés dans le [rapport précédent](lookup-mps-20261028.md) ; ses chiffres ne décrivent pas la nouvelle implémentation.

## Vérification de l'interface

Parcours vérifié sur un serveur temporaire au port 8767, avec les poids locaux : formulaire → `POST /api/compare` → moteur adaptatif → affichage des textes et compteurs. Le serveur habituel de l'utilisateur au port 8765 est resté actif.

| Cas | Observation |
| --- | --- |
| Premier essai de copie, greedy, après démarrage | Préparation MPS de 1,05 s ; 74,47 → 272,85 tokens/s, soit 3,66× ; mêmes tokens, 11 / 0 appels, 8 blocs vérifiés dont 6 avec contrôle préalable réduit. |
| Prompt court sur la pluie, greedy, 64 tokens | 77,20 → 115,68 tokens/s, soit 1,50× ; mêmes tokens, 42 / 0 appels, message « Répétition détectée pendant la génération ». |
| API et navigateur | Page, informations et comparaisons : HTTP 200 ; aucune erreur de console observée. |
| Réglages et documentation | Contrôles restaurés après calcul ; température, top-k et top-p désactivés uniquement en greedy ; documentation adaptative accessible. |

Ces mesures d'interface sont distinctes des médianes du benchmark après échauffement complet. La préparation MPS ne supprime pas toute variation de compilation et de débit. Le contenu de l'exemple sur la pluie est incorrect et répétitif dans les deux colonnes : l'accélération ne corrige pas le modèle.
