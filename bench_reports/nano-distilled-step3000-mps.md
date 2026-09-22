# Premier brouillon compatible sur Apple MPS

Mesure du 22 septembre 2026. Les essais bruts sont dans [CSV](nano-distilled-step3000-mps.csv) et [JSON](nano-distilled-step3000-mps.json).

## Modèles et protocole

- Cible : Baguette SFT, 122 710 784 paramètres.
- Brouillon : preset `nano`, 16 619 136 paramètres, même tokenizer que la cible. Initialisation par un pilote de 50 étapes de pré-entraînement, puis 3 000 étapes de distillation sur 3 072 000 tokens. Le KL de validation du dernier point de contrôle était d'environ 2,44.
- Matériel : backend Apple MPS ; cache KV activé pour les deux modèles.
- Échantillonnage : température 0,5, top-k 20, top-p 0,95.
- Trois prompts en français, trois graines par prompt, 64 tokens maximum, un essai d'échauffement avant les mesures. Deux tailles de bloc : K=2 et K=4.

| Méthode | Débit médian (tokens/s) | Accélération médiane | Acceptation médiane |
|---|---:|---:|---:|
| Cible seule | 63,81 | 1,00× | — |
| SpecDec K=2 | 40,18 | 0,653× | 26,5 % |
| SpecDec K=4 | 30,91 | 0,536× | 13,3 % |

Le brouillon est donc **plus lent que la génération cible seule** dans ces essais. Pour K=2, les accélérations individuelles vont de 0,615× à 0,820×. L'acceptation encore faible ne compense pas les passages du brouillon et la vérification. L'écart de validation baisse avec l'entraînement, mais ce point de contrôle ne démontre aucun gain de vitesse.

Ces résultats décrivent cette machine et ce petit jeu de prompts. Un prompt s'est arrêté avant 64 tokens selon la graine ; le CSV donne le nombre exact de tokens pour chaque essai. Les ratios de vitesse sont calculés essai par essai, puis résumés par la médiane. Des mesures sur davantage de prompts, des répétitions et un brouillon mieux entraîné sont nécessaires avant de conclure sur la performance générale.
