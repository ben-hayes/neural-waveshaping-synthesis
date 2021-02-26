import click
import pytorch_lightning as pl

from nts.data.urmp import URMPDataModule
from nts.models.timbre_transfer import TimbreTransfer


@click.command()
@click.option("--device", default="cuda")
@click.option("--instrument", default="vn")
def main(device, instrument):
    model = TimbreTransfer(learning_rate=1e-3)
    data = URMPDataModule(
        "/import/c4dm-datasets/URMP/synth-dataset/4-second",
        instrument,
        batch_size=32,
        num_workers=32,
        shuffle=True,
    )

    logger = pl.loggers.WandbLogger(project="neural-timbre-shaping")
    logger.watch(model, log="parameters")

    trainer = pl.Trainer(logger=logger, gpus=device)
    trainer.fit(model, data)


if __name__ == "__main__":
    main()