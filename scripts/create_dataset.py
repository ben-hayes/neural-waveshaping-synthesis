import os

import click
import gin

from nts.data.utils.create_dataset import create_dataset
from nts.utils import seed_all


def get_filenames(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if ".wav" in f]


@click.command()
@click.option("--gin-file", prompt="Gin config file")
@click.option("--data-directory", prompt="Data directory")
@click.option("--output-directory", prompt="Output directory")
@click.option("--seed", default=0)
@click.option("--device", default="cpu")
def main(gin_file, data_directory, output_directory, seed=0, device="cpu"):
    gin.constant("device", device)
    gin.parse_config_file(gin_file)

    seed_all(seed)

    files = get_filenames(data_directory)
    create_dataset(files, output_directory)


if __name__ == "__main__":
    main()