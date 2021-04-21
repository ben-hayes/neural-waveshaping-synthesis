import click
import gin
import pytorch_lightning as pl

from nts.data.urmp import URMPDataModule
from nts.models.timbre_transfer import TimbreTransfer
from nts.models.timbre_transfer_newt import TimbreTransferNEWT


@gin.configurable
def get_model(model):
    return model()


@gin.configurable
def trainer_kwargs(**kwargs):
    return kwargs


@click.command()
@click.option("--gin-file", prompt="Gin config file")
@click.option("--device", default="cuda")
@click.option("--instrument", default="vn")
@click.option("--load-data-to-memory/--load-data-from-disk", default=True)
@click.option("--with-wandb/--without-wandb", default=True)
@click.option("--restore-checkpoint", default="")
def main(
    gin_file, device, instrument, load_data_to_memory, with_wandb, restore_checkpoint
):
    gin.parse_config_file(gin_file)
    model = get_model()
    data = URMPDataModule(
        "/import/c4dm-datasets/URMP/synth-dataset/4s-dataset",
        instrument,
        load_to_memory=load_data_to_memory,
        num_workers=16,
        shuffle=True,
    )

    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    logger = pl.loggers.WandbLogger(project="neural-timbre-shaping", log_model=True)
    logger.watch(model, log="parameters")

    checkpointing = pl.callbacks.ModelCheckpoint(monitor="val/loss", save_top_k=5)

    kwargs = trainer_kwargs()
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[lr_logger, checkpointing],
        gpus=device,
        resume_from_checkpoint=restore_checkpoint if restore_checkpoint != "" else None,
        profiler=profiler,
        **kwargs,
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    main()