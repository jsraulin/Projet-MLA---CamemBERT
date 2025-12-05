from pytorch_lightning.cli import LightningCLI
from cmbrt_module import CmbrtLightningModule
from cmbrt_data import CmbrtDataModule

def cli_main():
    # LightningCLI connecte automatiquement le module, les données et les arguments
    cli = LightningCLI(CmbrtLightningModule, CmbrtDataModule)

if __name__ == "__main__":
    cli_main()