import click
import gin
import pytorch_lightning as pl

from nts.data.urmp import URMPDataModule
from nts.models.timbre_transfer import TimbreTransfer


@click.command()
@click.option("--gin-file", prompt="Gin config file")
@click.option("--device", default="cuda")
@click.option("--instrument", default="vn")
def main(gin_file, device, instrument):
    gin.parse_config_file(gin_file)
    model = TimbreTransfer(learning_rate=1e-3)
    data = URMPDataModule(
        "/import/c4dm-datasets/URMP/synth-dataset/4-second-segments",
        instrument,
        num_workers=32,
        shuffle=True,
    )

    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    logger = pl.loggers.WandbLogger(project="neural-timbre-shaping")
    logger.watch(model, log="parameters")

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[lr_logger],
        gpus=device,
        # val_check_interval=val_check_interval,
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    main()