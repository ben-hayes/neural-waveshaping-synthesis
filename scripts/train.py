import click
import gin
import pytorch_lightning as pl

from neural_waveshaping_synthesis.data.general import GeneralDataModule
from neural_waveshaping_synthesis.data.urmp import URMPDataModule
from neural_waveshaping_synthesis.models.neural_waveshaping import NeuralWaveshaping


@gin.configurable
def get_model(model, with_wandb):
    return model(log_audio=with_wandb)


@gin.configurable
def trainer_kwargs(**kwargs):
    return kwargs


@click.command()
@click.option("--gin-file", prompt="Gin config file")
@click.option("--dataset-path", prompt="Dataset root")
@click.option("--urmp", is_flag=True)
@click.option("--device", default="0")
@click.option("--instrument", default="vn")
@click.option("--load-data-to-memory", is_flag=True)
@click.option("--with-wandb", is_flag=True)
@click.option("--restore-checkpoint", default="")
def main(
    gin_file,
    dataset_path,
    urmp,
    device,
    instrument,
    load_data_to_memory,
    with_wandb,
    restore_checkpoint,
):
    gin.parse_config_file(gin_file)
    model = get_model(with_wandb=with_wandb)

    if urmp:
        data = URMPDataModule(
            dataset_path,
            instrument,
            load_to_memory=load_data_to_memory,
            num_workers=16,
            shuffle=True,
        )
    else:
        data = GeneralDataModule(
            dataset_path,
            load_to_memory=load_data_to_memory,
            num_workers=16,
            shuffle=True,
        )

    checkpointing = pl.callbacks.ModelCheckpoint(
        monitor="val/loss", save_top_k=1, save_last=True
    )
    callbacks = [checkpointing]
    if with_wandb:
        lr_logger = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_logger)
        logger = pl.loggers.WandbLogger(project="neural-waveshaping-synthesis")
        logger.watch(model, log="parameters")


    kwargs = trainer_kwargs()
    trainer = pl.Trainer(
        logger=logger if with_wandb else None,
        callbacks=callbacks,
        gpus=device,
        resume_from_checkpoint=restore_checkpoint if restore_checkpoint != "" else None,
        **kwargs
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    main()