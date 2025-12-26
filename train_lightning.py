# train_lightning.py
# Lancement de l'entraînement Lightning de CamemBERT simplifié
from pytorch_lightning import Trainer
from cmbrt_lightning_module import CamembertLightning
from cmbrt_data_module import CamembertDataModule

if __name__ == "__main__":
    # Chemin vers ton dataset texte
    dataset_path = "./dataset_g5/fr_text.txt"  # à adapter si besoin

    # Charger données + modèle
    data_module = CamembertDataModule(dataset_path, batch_size=8, max_length=128)
    model = CamembertLightning(vocab_size=32000, lr=1e-4)

    # Créer un Trainer Lightning (gère GPU, logs, checkpoints)
    trainer = Trainer(
        max_epochs=3,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        log_every_n_steps=10
    )

    # Entraînement
    trainer.fit(model, data_module)
