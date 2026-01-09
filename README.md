# CamemBERT : Réimplémentation et Évaluation (Projet MLA - Groupe 5)

Ce dépôt contient les travaux de notre groupe sur le modèle **CamemBERT**, l'adaptation francophone de l'architecture RoBERTa (Martin et al., 2020). Ce projet s'articule autour de deux axes principaux : la réimplémentation de l'architecture "from scratch" et l'évaluation des performances par fine-tuning sur des tâches de TALN.

## Structure du Dépôt

L'organisation du répertoire reflète le travail collaboratif effectué sur les différentes branches du projet :

| Dossier | Responsable | Description / Tâche |
| :--- | :--- | :--- |
| `ProjetCamemBERT_Ghiles` | **Ghiles REDJDAL** | **Version de référence** : Réimplémentation finale du modèle et moteur de pré-entraînement MLM. |
| `amine` | **Amine NAIT SI AHMED** | Fine-tuning et évaluation sur les tâches **XNLI** et **NER**. |
| `Youdas_camembert` | **Youdas BEDHOUCHE** | Fine-tuning et évaluation sur le **POS Tagging**. |
| `JSRaulin (obsolete)` | **Jean-Sébastien RAULIN** | Code sourve d'une première itération du modèle. |

---

## Répartition du Travail

### Axe 1 : Réimplémentation & Pré-entraînement (Pre-training Track)
L'objectif était de reconstruire l'architecture **CamemBERT-base** (12 couches, 768 dimensions) et d'assurer sa convergence.
* **Ghiles REDJDAL:** Développement de la version finale du modèle utilisée pour les mesures, gestion de la précision 32-bit (FP32) pour la stabilité et exécution du pré-entraînement MLM.
* **Jean-Sébastien RAULIN :** Conception de la structure initiale du modèle et développement des pipelines de données, entrainement du tokenizer SentencePiece (non-utilisé)

### Axe 2 : Fine-Tuning & Évaluation (Evaluation Track)
Validation des capacités linguistiques du modèle sur des jeux de données spécialisés.
* **Amine NAIT SI AHMED:** Mise en place des protocoles d'évaluation pour la reconnaissance d'entités nommées (**NER**) et l'inférence naturelle (**XNLI**).
* **Youdas BEDHOUCHE:** Spécialisation du modèle sur la tâche de **POS Tagging** pour vérifier la compréhension grammaticale.
