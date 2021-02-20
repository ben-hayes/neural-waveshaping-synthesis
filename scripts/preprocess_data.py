import os

import click

from nts.data.utils.preprocess_audio import preprocess_audio


@click.command()
@click.option("--directory", prompt="Audio directory")
def hello(directory):
    print(directory)
    files = [os.path.join(directory, f) for f in os.listdir(directory) if ".wav" in f]
    preprocess_audio(files)

if __name__ == "__main__":
    hello()