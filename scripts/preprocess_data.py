import os

import click
import gin

from nts.data.utils.preprocess_audio import preprocess_audio

@gin.configurable
def load_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if ".wav" in f]

@click.command()
@click.option("--gin-file", prompt="Gin config file")
def main(gin_file):
    gin.parse_config_file(gin_file)
    files = load_files()
    preprocess_audio(
        files,
    )


if __name__ == "__main__":
    main()