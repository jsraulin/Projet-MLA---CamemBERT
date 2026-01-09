\# Résultats NER - CamemBERT



\## Configuration

\- Modèle : camembert-base

\- Dataset : WikiNER French (120,682 train / 13,410 test)

\- Epochs : 3

\- Batch size : 256

\- Learning rate : 2e-5

\- GPU : Quadro RTX 6000



\## Résultats par epoch



| Epoch | Training Loss | Validation Loss | F1-Score |

|-------|---------------|-----------------|----------|

| 1     | -             | 0.1986          | 86.02%   |

| 2     | 0.3291        | 0.1363          | 88.06%   |

| 3     | 0.3291        | 0.1240          | 88.08%   |



\## Résultat final (Test Set)



| Métrique | Notre test | Article |

|----------|------------|---------|

| \*\*F1-Score\*\* | \*\*88.07%\*\* | 89.97% |

| Precision | 87.52% | - |

| Recall | 88.62% | - |

| Accuracy | 98.46% | - |



\## Temps d'entraînement

\- Durée : ~14 minutes

\- GPU : Quadro RTX 6000 (24 GB)

