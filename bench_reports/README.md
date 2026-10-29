# Rapports de benchmark

Les rapports décrivent les mesures réalisées à une date et avec une version précise du moteur. Les données brutes conservent les prompts, les durées et les compteurs utilisés pour les calculs. Les poids restent à l'extérieur du dépôt.

| Rapport | Comparaison | Données |
| --- | --- | --- |
| [Lookup adaptatif — 28 octobre 2026](adaptive-lookup-mps-20261028.md) | Moteur adaptatif, ancien lookup et cible seule ; 60 essais MPS | [JSON](adaptive-lookup-mps-20261028.json) |
| [Premier mode automatique — 28 octobre 2026](lookup-mps-20261028.md) | Lookup, brouillon nano et cible seule ; 42 essais MPS | [JSON](lookup-mps-20261028.json) |
| [Optimisation du cache — 28 octobre 2026](cached-optimization-20261028.md) | Moteur avec cache avant et après optimisation | [Échantillonnage](cached-optimization-20261028-sampling.json), [greedy](cached-optimization-20261028-greedy.json) |
| [Brouillon nano distillé](nano-distilled-step3000-mps.md) | Cible seule et brouillon distillé à 3 000 étapes | [JSON](nano-distilled-step3000-mps.json), [CSV](nano-distilled-step3000-mps.csv) |

Les essais adaptatifs ont été réalisés avant le nettoyage de présentation du code, sur la révision [`4adf0e6`](https://github.com/HENK0O/baguette-SpecDec/tree/4adf0e6). Les empreintes enregistrées dans les fichiers bruts sont celles des sources mesurées. Le formatage ultérieur ne constitue pas une nouvelle mesure.

Les gains dépendent du prompt, du mode et de la machine. Les textes de ce petit jeu de développement ne représentent pas tous les usages ; les rapports conservent aussi les cas sans gain.

[Retour au projet](../README.md)
