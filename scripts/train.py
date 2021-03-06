import click
import gin
import pytorch_lightning as pl

from nts.data.urmp import URMPDataModule
from nts.models.timbre_transfer import TimbreTransfer


@click.command()
@click.option("--gin-file", prompt="Gin config file")
@click.option("--device", default="cuda")
@click.option("--instrument", default="vn")
@click.option("--load-data-to-memory", default=True)
def main(gin_file, device, instrument, load_data_to_memory):
    gin.parse_config_file(gin_file)
    model = TimbreTransfer(learning_rate=1e-3)
    data = URMPDataModule(
        "/import/c4dm-datasets/URMP/synth-dataset/4-second-segments",
        instrument,
        load_to_memory=load_data_to_memory,
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
        # val_check_interval=100,
        # check_val_every_n_epoch=5,
        max_epochs=5000,
        # overfit_batches=1,
        # gradient_clip_val=1.0,
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    main()