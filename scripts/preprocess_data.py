import os

import click

from nts.data.utils.preprocess_audio import preprocess_audio


@click.command()
@click.option("--directory", prompt="Audio directory")
@click.option("--target-sr", default=16000)
@click.option("--f0-extractor", default="crepe")
def hello(directory, target_sr, f0_extractor):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if ".wav" in f]
    preprocess_audio(files, target_sr=target_sr, f0_extractor=f0_extractor)

if __name__ == "__main__":
    hello()