import os
from pathlib import Path

import click
import gin

from neural_waveshaping_synthesis.data.utils.create_dataset import create_dataset
from neural_waveshaping_synthesis.utils import seed_all

INSTRUMENTS = (
    "vn",
    "vc",
    "fl",
    "cl",
    "tpt",
    "sax",
    "tbn",
    "ob",
    "va",
    "bn",
    "hn",
    "db",
)


def get_instrument_file_list(instrument_string, directory):
    return [
        str(f)
        for f in Path(directory).glob(
            "**/*_%s_*/AuSep*_%s_*.wav" % (instrument_string, instrument_string)
        )
    ]


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

    file_lists = {
        instrument: get_instrument_file_list(instrument, data_directory)
        for instrument in INSTRUMENTS
    }
    for instrument in file_lists:
        create_dataset(
            file_lists[instrument], os.path.join(output_directory, instrument)
        )


if __name__ == "__main__":
    main()