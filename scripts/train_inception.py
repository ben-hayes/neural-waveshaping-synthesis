import click
import gin
import pytorch_lightning as pl

from nts.evaluation.inception import InceptionInstrumentClassifier
from nts.data.urmp_classification import URMPClassificationDataModule


@gin.configurable
def trainer_kwargs(**kwargs):
    return kwargs


@click.command()
@click.option("--gin-file", prompt="Gin config file")
@click.option("--device", default="cuda")
def main(gin_file, device):
    gin.parse_config_file(gin_file)

    model = InceptionInstrumentClassifier()
    data = URMPClassificationDataModule()

    checkpointing = pl.callbacks.ModelCheckpoint(monitor="val/loss", save_top_k=5)

    logger = pl.loggers.WandbLogger(project="inception-urmp", log_model=False)
    logger.watch(model, log="parameters")

    kwargs = trainer_kwargs()
    trainer = pl.Trainer(
        callbacks=[checkpointing], logger=logger, gpus=device, **kwargs
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    main()