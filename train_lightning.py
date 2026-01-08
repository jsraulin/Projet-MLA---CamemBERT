import sys
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from cmbrt_lightning_module import CamembertLightning
from cmbrt_data_module import CamembertDataModule

if __name__ == "__main__":
    dm = CamembertDataModule(
        base_path="/home/camembert/dataset_g5",
        batch_size=128,      
        max_length=128,
        num_workers=8
    )

    model = CamembertLightning(lr=1e-4)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="camembert-{epoch:02d}-{train_loss:.3f}",
        save_top_k=-1,
        every_n_epochs=1,
        save_weights_only=True,
        monitor="train_loss",
        mode="min"
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger("lightning_logs", name="camembert_full")

    trainer = Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=1,
        precision=32,
        accumulate_grad_batches=2,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
    )

    ckpt_path = None
    if len(sys.argv) > 1 and sys.argv[1].startswith("--ckpt_path="):
        ckpt_path = sys.argv[1].split("=")[1]
        print(f"[INFO] Reprise depuis le checkpoint : {ckpt_path}")

    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
