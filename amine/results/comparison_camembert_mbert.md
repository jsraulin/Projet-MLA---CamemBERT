\# Comparaison CamemBERT vs mBERT



\## Résultats



| Tâche | CamemBERT | mBERT | Gagnant |

|-------|-----------|-------|---------|

| \*\*XNLI\*\* | \*\*81.78%\*\* | 77.54% | CamemBERT +4.24% |

| \*\*NER\*\* | 88.07% | \*\*90.65%\*\* | mBERT +2.58% |



\## Analyse



\- \*\*XNLI\*\* : CamemBERT est meilleur pour la compréhension du français

\- \*\*NER\*\* : mBERT obtient de meilleurs résultats sur WikiNER



\## Configuration

\- GPU : Quadro RTX 6000 (24 GB)

\- CamemBERT : camembert-base (110M params)

\- mBERT : bert-base-multilingual-cased (175M params)

